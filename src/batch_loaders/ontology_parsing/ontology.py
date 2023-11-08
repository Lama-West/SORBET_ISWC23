from cProfile import label
from concurrent.futures import process
import itertools
import logging
from sys import prefix
from xml.dom import minidom
from urllib.request import urlopen
from typing import List

from sklearn import neighbors
from .preprocessing import preprocessor
from .synonym_generator import SynonymGenerator
import numpy as np
import json
import os
from copy import deepcopy
import random 
from nltk.corpus import wordnet as wn
import copy
from .ontology_config import OntoConfig
from globals import Globals
from numpy import dot
from numpy.linalg import norm
from enum import Enum
import re
import importlib

import owlready2

logger = logging.getLogger("onto")

flatten = lambda l: [item for sublist in l for item in sublist]

# TODO: refactor Ontology by add Node class


class Ontology():
    def __init__(self, path, prefix=None, ontology_config: OntoConfig = None, bert_vocab=None, fast=False):
        
        importlib.reload(owlready2)

        if not ontology_config:
            ontology_config = OntoConfig()

        if prefix is not None:
            self.ontology_path = os.path.join(prefix, path)
        else:
            self.ontology_path = path

        self.onto_config = ontology_config

        if self.ontology_path.startswith("https:/") or self.ontology_path.startswith("http:/") or self.ontology_path.startswith("file:/"):
            self.onto = owlready2.get_ontology(self.ontology_path).load()
        else:
            self.onto = owlready2.get_ontology("file://" + self.ontology_path).load()

        # try:
        #     owlready2.sync_reasoner(infer_property_values = True)
        # except:
        #     pass

        # get Ontology name
        self.onto_name = self.onto.name
        self.bert_vocab = Globals.tokenizer.get_vocab().keys() | [self.onto_name]

        
        self.xsd_types = ["integer",
                        "nonPositiveInteger",
                        "negativeInteger",
                        "long",
                        "int",
                        "short",
                        "byte ",
                        "nonNegativeInteger",
                        "unsignedLong",
                        "unsignedInt",
                        "unsignedShort",
                        "unsignedByte",
                        "positiveInteger",
                        "double",
                        "decimal",
                        "float",
                        "string",
                        "normalizedString",
                        "token",
                        "duration",
                        "time",
                        "binary",
                        "date",
                        "datetime",
                        "boolean"]
        self.id_to_label_dict = dict({})

        self.iris = {}
        self.construct_mapping_dict()  

        self.parse_synonyms()
        self.parse_definitions()

        self.construct_synonym_controlled_randomness()


        filepath, filename = os.path.split(self.ontology_path)
        cache_file = os.path.join(filepath, "cache", filename + ".cache")

        if self.onto_config.use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    self.id_to_label_dict = json.load(f) 
            except:
                logger.error(f"Could not load cached pre-processed ontology: {self.ontology}")
                self.construct_ID_label_dict() 
                with open(cache_file, "w+") as f:
                    f.write(json.dumps(self.id_to_label_dict, indent=4)) 

        else:
            self.construct_ID_label_dict() 

            if self.onto_config.use_cache :
                os.makedirs(os.path.join(filepath, "cache"), exist_ok=True)
                with open(cache_file, "w+") as f:
                    f.write(json.dumps(self.id_to_label_dict, indent=4)) 

        if (self.onto_config.synonym_extension) and (not fast) and (self.__len__() < 1000):
            self.extend_synonyms() 

        self.id_to_index = {id: i for i, id in enumerate(self.classes)}
        self.cousins_set = None
        self.neighbours_cache = None

        self.parse_use_in_alignment_annotation()


    def __len__(self):
        return len(self.classes)

    def get_id_index(self, id):
        return self.id_to_index[id]

    def get_classes(self):
        return self.classes

    def get_name(self):
        return self.onto_name
        
    def extract_label(self, node):
        try:
            label = node.label
            if len(label) > 0:
                return node.label[0]
            else:
                return node.name
        except:
            return node.name

    def add_to_label_dict(self, *nodes):
        for node in nodes:
            if node.name not in self.id_to_label_dict:
                self.id_to_label_dict[node.name] = self.extract_label(node)
                self.iris[node.name] = node.iri

    def get_parents(self, node):
        if node not in self.parents_map:
            return []
        else:
            return self.parents_map[node]

    def get_childs(self, node):
        if node not in self.childs_map:
            return []
        else:
            return self.childs_map[node]

    
    def get_object_properties(self, node):
        if node not in self.object_prop_map:
            return []
        else:
            return self.object_prop_map[node]

    def parse_concept(self, concept, add_to_dicts=True):

        # Check if datatype property
        if isinstance(concept, (int, float)):
            return type(concept).__name__
        if (concept is str):
            return "string"
        if (concept is int) or (concept is float):
            return "int"
        if (concept is bool) or isinstance(concept, bool):
            return "True"

        if (type(concept) is not owlready2.entity.ThingClass) and (type(concept) is not owlready2.prop.ObjectPropertyClass) \
            and (type(concept) is not owlready2.prop.DataPropertyClass):
            logger.debug(f"Problem during preprocessing, concept of type: {str(type(concept))} is not supported")
            return "[UNK]"

        if (concept.name not in self.childs_map) and add_to_dicts:

            self.childs_map[concept.name] = []
            self.parents_map[concept.name] = []
            self.object_prop_map[concept.name] = []
            self.data_prop_map[concept.name] = []

        
        self.add_to_label_dict(concept)

        return concept.name


    def get_subjects(self, concept):
        if type(concept) is owlready2.class_construct.Restriction:
            return [concept.value]

        elif (type(concept) is owlready2.class_construct.Or) or (type(concept) is owlready2.class_construct.And):
                
            return concept.get_Classes()
        
        else:
            return [concept]

    def construct_mapping_dict(self):
        '''
        Constructs ID to label mapping dict for ontologies where 
        entities are identified by IDs.
        '''

        elements = list(self.onto.classes())
        self.classes = [node.name for node in elements if node.name not in self.onto_config.exclude_classes]

        self.childs_map       = { el.name:[]  for el in elements }
        self.parents_map      = { el.name:[]  for el in elements }
        self.object_prop_map  = { el.name:[]  for el in elements }
        self.data_prop_map    = { el.name:[]  for el in elements }
        

        # Parsing all object properties
        for object_property in self.onto.object_properties():
            object_property_name = self.parse_concept(object_property, add_to_dicts=False)

            for object, subjects in object_property.get_relations():

                object_name = self.parse_concept(object)
                
                if object_name == "[UNK]":
                    continue

                for subject in self.get_subjects(subjects):
                    subject_name = self.parse_concept(subject)

                    self.object_prop_map[object_name].append((object_property_name, subject_name))


            for _domain in [d for union in object_property.domain for d in self.get_subjects(union)]:
                domain_name = self.parse_concept(_domain)
                for _range in [r for union in object_property.range for r in self.get_subjects(union)]:
                    range_name = self.parse_concept(_range)
                    
                    self.object_prop_map[domain_name].append((object_property_name, range_name))






        # Parsing all Owl:Class definitions
        for concept in self.onto.classes():
            concept_name = self.parse_concept(concept)



            for parent_classes in (self.onto.get_parents_of(concept) + concept.equivalent_to):
                
                # Not adding these relations
                if type(parent_classes) is owlready2.class_construct.Not:
                    continue

                # This is the case some object properties is in the class definition
                if "property" in parent_classes.__dict__:
                    property_parsed = self.parse_concept(parent_classes.property, add_to_dicts=False)

                    for subject in self.get_subjects(parent_classes.value):
                        c_parsed = self.parse_concept(subject)

                        if c_parsed == "[UNK]":
                            continue

                        if (self.onto_config.subclass_of_properties == "all") or (property_parsed in self.onto_config.subclass_of_properties):
                            if c_parsed in self.childs_map:
                                self.parents_map[concept_name].append(c_parsed)

                                self.childs_map[c_parsed].append(concept_name)
                        
                        else:

                            self.object_prop_map[concept_name].append((property_parsed, c_parsed))
                
                # Usual Subclass relation
                else:

                    for parent_class in self.get_subjects(parent_classes):

                        parent_name = self.parse_concept(parent_class)

                        if parent_name == "[UNK]":
                            continue

                        self.parents_map[concept_name].append(parent_name)

                        self.childs_map[parent_name].append(concept_name)
        
        # TODO: Add datatype props

    def parse_use_in_alignment_annotation(self):

        self.not_used_in_alignment = []
        for property in self.onto.annotation_properties():
            if property.name == "use_in_alignment":
                for object, subject in property.get_relations():
                    if (subject == "False") or (subject == False):
                        self.not_used_in_alignment.append(object.name)

        self.not_used_in_alignment = set(self.not_used_in_alignment)

    def parse_synonyms(self):
        synonyms_tags = {"hasRelatedSynonym", "hasExactSynonym", "P90", "exactMatch", 
                    "alternative_term", "synonym", "altLabel", "prefLabel", "definition"}

        self.synonyms = {el:[el]  for el in self.classes }

        for property in self.onto.annotation_properties():
            if property.name in synonyms_tags:
                for object, subject in property.get_relations():
                    if object.name in self.synonyms:
                        if isinstance(subject, str):
                            if not subject.startswith("http"):
                                self.synonyms[object.name].append(subject)
                                self.id_to_label_dict[subject] = subject
                                self.add_to_label_dict(object)
                        else:  
                            self.synonyms[object.name].append(subject.name)

                            self.add_to_label_dict(object, subject)

        self.synonyms["Thing"] = ["Thing"]

    def parse_definitions(self):
        definitions_tags = {"hasDefinition"}

        for property in self.onto.annotation_properties():
            if property.name in definitions_tags:
                for object, subject in property.get_relations():
                    self.object_prop_map[object.name].append((":", subject.name))

                    self.add_to_label_dict(object, subject)
        

    def get_id_label(self, id):
        return self.id_to_label_dict[id]

    def construct_ID_label_dict(self):
        '''
        constructs the id to preprocessed label dict
        '''
        preprocessor.set_params(bert_vocab=self.bert_vocab, autocorrect=self.onto_config.autocorrect)

        self.id_to_label_dict = { id:preprocessor.process(label) for id, label in self.id_to_label_dict.items()}

        # Special tokens
        self.id_to_label_dict["int"] = ["number"]
        self.id_to_label_dict["float"] = ["number"]
        self.id_to_label_dict["string"] = ["something"]
        self.id_to_label_dict["False"] = ["boolean"]
        self.id_to_label_dict["True"] = ["boolean"]
        self.id_to_label_dict["bool"] = ["boolean"]


        self.id_to_label_dict[">"] = [">"]
        self.id_to_label_dict["<"] = ["<"]
        self.id_to_label_dict[";"] = [";"]
        self.id_to_label_dict[":"] = [":"]
        self.id_to_label_dict["Thing"] = ["Thing"]
        self.id_to_label_dict["Anatomic_Structure_Is_Physical_Part_Of"] = ["part", "of"]
        self.id_to_label_dict["UNDEFINED_part_of"] = ["part", "of"]

        self.id_to_label_dict["[UNK]"] = ["[UNK]"]
        self.id_to_label_dict["[SEP]"] = ["[SEP]"]


        
                

    def construct_synonym_controlled_randomness(self):
        self.past_chosen_synonyms = {id: np.ones((len(synonyms),), dtype=np.ushort) for id, synonyms in self.synonyms.items()}


    def extend_synonyms(self):

        filepath, filename = os.path.split(self.ontology_path)
        cache_file = os.path.join(filepath, "cache", self.onto_name + ".synonyms_cache.json")

        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                self.synonyms = json.load(f) 

                for id in self.synonyms:
                    for synonym in self.synonyms[id]:
                        if synonym not in self.id_to_label_dict:
                            self.id_to_label_dict[synonym] = synonym.split(" ")
            return

        sg_instance = SynonymGenerator()

        # sg_instance.set_context(all_labels)

        for id in self.classes:
            labels = self.get_id_label(id)

            if len(labels) > 4:
                continue

            synonyms = sg_instance.generate_synonyms(labels)
            
            for synonym in synonyms:
                if id in self.synonyms:
                    self.id_to_label_dict[synonym] = synonym.split(" ")
                    self.synonyms[id].append(synonym)

                    logger.debug(f"Added synonym to: [{id}] synonym [{synonym}]")

        with open(cache_file, "w+") as f:
            f.write(json.dumps(self.synonyms, indent=4)) 
                

    def get_random_synonym(self, node):
        
        if node not in self.synonyms:
            return node

        if len(self.synonyms[node]) <= 1:
            return node

        weights = self.past_chosen_synonyms[node]/np.sum(self.past_chosen_synonyms[node])
        chosen_arg = random.choices(list(range(len(self.past_chosen_synonyms[node]))), weights=weights)[0]

        self.past_chosen_synonyms[node] = self.past_chosen_synonyms[node] + 1
        self.past_chosen_synonyms[node][chosen_arg] = 1

        return self.synonyms[node][chosen_arg]

    def node_has_neighbours(self, root_node, add_parents=True, add_childs=True, add_obj_prop=True):
        n = len(self.get_childs(root_node)) if add_parents else 0 
        n+= len(self.get_parents(root_node))  if add_childs else 0
        n+= len(self.get_object_properties(root_node)) if add_obj_prop else 0

        return n


    def build_neighbours_cache(self, A):
        for c in self.classes:
            self.neighbours_cache[c] = set(self.get_node_neighbours(c, k_hop_max=4, add_obj_prop=False, flatten=True))


    def get_cached_neighbours(self, id, A=4):
        if self.neighbours_cache is None:
            self.build_neighbours_cache(A)
            
        return self.neighbours_cache[id]

    def build_cousins_set(self):
        self.cousins_set = []

        for c in self.parents_map:
            cousins, _ = self.get_random_neighbour(c, strategy="cousins")

            if cousins is not None:
                self.cousins_set.append(c)
                continue

        self.cousins_set = set(self.cousins_set)



    def node_has_cousins(self, root_node):
        if self.cousins_set is None:
            self.build_cousins_set()

        if root_node not in self.cousins_set:
            return False

        return True



    def get_random_neighbour(self, root_node, k_hop_max=4, strategy="lineage_path"):

        if strategy == "all":

            all_neighbours = self.get_node_neighbours(root_node, k_hop_max=k_hop_max)

        if strategy == "ontologic_relation":

            all_neighbours = self.get_node_neighbours(root_node, k_hop_max=k_hop_max, add_obj_prop=False)    

        elif strategy == "lineage_path":
            
            all_neighbours = self.get_node_neighbours(root_node, k_hop_max=k_hop_max, add_childs=False, add_obj_prop=False)

        elif strategy == "lineage_path_and_childs":

            parents = self.get_node_neighbours(root_node, k_hop_max=k_hop_max, add_childs=False, add_obj_prop=False)
            childs = self.get_node_neighbours(root_node, k_hop_max=k_hop_max, add_parents=False, add_obj_prop=False)

            all_neighbours = []

            for i in range(max(len(parents), len(childs))):
                temp = []
                if i < len(parents):
                    temp.extend(parents[i])
                
                if i < len(childs):
                    temp.extend(childs[i])

                all_neighbours.append(temp)

        elif strategy == "cousins":
            parents = self.get_node_neighbours(root_node, k_hop_max=2, add_childs=False, add_obj_prop=False, flatten=True)

            if len(parents) == 0:
                return None, None

            cousins = self.get_node_neighbours(parents, k_hop_max=2, add_parents=False, add_obj_prop=False)

            if len([cousin for hop in cousins for cousin in hop]) == 0:
                return None, None
                
            all_neighbours = cousins

        k_hop = random.randint(0, len(all_neighbours) - 1)
        
        chosen_node = random.choice(list(all_neighbours[k_hop]))

        return chosen_node, k_hop + 1  


    def get_node_neighbours(self, root_node: str, k_hop_max=1, add_parents=True, add_childs=True, add_obj_prop=True, flatten=False):


        if isinstance(root_node, List):
            current_neighbors = set(root_node)
        else:
            current_neighbors = set([root_node])


        # TODO: consider or not object prop neighbours
        all_neighbours = []
        
        visited_nodes = copy.deepcopy(current_neighbors)
        for i in range(k_hop_max):
            new_neighbours = set({})
            for node in current_neighbors:
                if add_childs:
                    new_neighbours.update(self.get_childs(node))

                if add_parents:
                    new_neighbours.update(self.get_parents(node))

                if add_obj_prop:
                    new_neighbours.update([n[1] for n in self.get_object_properties(node)])
            
            new_neighbours = new_neighbours - visited_nodes

            if len(new_neighbours) == 0:
                break
            
            visited_nodes = visited_nodes.union(new_neighbours)

            if flatten:
                all_neighbours.extend(new_neighbours)
            else:
                all_neighbours.append(new_neighbours)

            current_neighbors = new_neighbours
        


        return all_neighbours

    def get_all_node_ancestors(self, root_node, max_lineage=-1, max_height=-1):
        ancestors = [root_node]
        iteration_nodes = [root_node]
        height = 0

        while (len(iteration_nodes) > 0) and ((max_height == -1) or (height < max_height)):
            height += 1

            next_iteration_nodes = []
            for parent_node in iteration_nodes:
                if (parent_node in self.parents_map) and (len(self.get_parents(parent_node)) > 0):
                    if max_lineage == -1:
                        next_iteration_nodes.extend(self.get_parents(parent_node))
                    else:
                        if height > 1 :
                            next_iteration_nodes.append(random.choice(self.get_parents(parent_node)))
                        else:
                            next_iteration_nodes.extend(random.choices(self.get_parents(parent_node), k=min(max_lineage, len(self.get_parents(parent_node)))))
            
            ancestors.extend(next_iteration_nodes)
            iteration_nodes = next_iteration_nodes

        return ancestors


