from ctypes import alignment
from typing import List
from xml.dom import minidom
import os 
import logging
import pandas as pd

class Alignment:
    def __init__(self, source_onto, target_onto, id1, id2, relation="=", label=1.0) -> None:
        self.source_onto =  source_onto
        self.target_onto = target_onto
        self.id1 = id1
        self.id2 = id2
        self.label = label
        self.relation = relation

    def __eq__(self, other):
        if isinstance(other, Alignment):
            return ((self.source_onto == other.source_onto) and (self.id1 == other.id1) and \
                    (self.target_onto == other.target_onto) and (self.id2 == other.id2) \
                    or \
                    (self.source_onto == other.target_onto) and (self.id1 == other.id2) and \
                    (self.target_onto == other.source_onto) and (self.id2 == other.id1)) 
        return False

    def is_class(self):
        "return whether the alignment is between 2 classes"

        return self.id1[0].isupper()

    def __repr__(self):
        return "({}#{}, {}#{}, {})".format(self.source_onto, self.id1, self.target_onto, self.id2, self.label)

    def __hash__(self):
        
        if self.source_onto < self.target_onto:
            id = "({}#{}, {}#{})".format(self.source_onto, self.id1, self.target_onto, self.id2)
        else:
            id = "({}#{}, {}#{})".format(self.target_onto, self.id2, self.source_onto, self.id1)
        return id.__hash__()

    def inverse(self):
        return Alignment(self.target_onto, self.source_onto, self.id2, self.id1)

class AlignmentDataset:
    def __init__(self, alignment_path, source_onto=None, target_onto=None, considered_relations=["="], type="OA") -> None:
        self.alignment_path = alignment_path
        self.considered_relations = considered_relations

        self.source_onto = source_onto
        self.target_onto = target_onto
        
        if alignment_path.endswith(".tsv") or alignment_path.endswith(".csv"):
            self.load_csv_alignments()
        else:
            self.load_alignments()

        self.type = type
        self.alignments = list(filter(lambda a: a.relation in self.considered_relations, self.alignments))
        self.user_input_candidates = None
        self.user_input_candidates_list = None


    def add_candidates(self, candidates_path):


        self.user_input_candidates = {}
        self.user_input_candidates_list = []

        if candidates_path.endswith(".tsv"):
            df = pd.read_csv(candidates_path, sep='\t')
        else:
            df = pd.read_csv(candidates_path, header=None, names=["SrcEntity", "TgtEntity", "TgtCandidates"])
        
        df = df.drop(columns="TgtEntity")



        for i, row in df.iterrows():

            chars_to_remove = [u'(', u'\'', u')']
            dd = {ord(c):None for c in chars_to_remove}

            row_parsed = row["TgtCandidates"].translate(dd).split(",")

            row_parsed = list(map(lambda s: s.split("/")[-1].split("#")[-1].split(";")[-1].split(":")[-1], row_parsed))

            src_entity = row["SrcEntity"].split("/")[-1].split("#")[-1].split(";")[-1].split(":")[-1]

            if src_entity in self.user_input_candidates:
                self.user_input_candidates[src_entity] = list(set(self.user_input_candidates[src_entity]).union(set(row_parsed)))
            else:
                self.user_input_candidates[src_entity] = row_parsed
            self.user_input_candidates_list.append(row_parsed)

        


    def split_train_test(self, all_train_alignments):
        self.train_alignments = []
        self.test_alignments = []
        for a in self.alignments:
            if a in all_train_alignments:
                self.train_alignments.append(a)
            else:
                self.test_alignments.append(a)



    def load_csv_alignments(self):
        self.alignments = []

        self.source_ontology_metadata = OntologyMetadata(id=self.source_onto)
        self.target_ontology_metadata = OntologyMetadata(id=self.target_onto)

        if self.alignment_path.endswith(".tsv"):
            df = pd.read_csv(self.alignment_path, sep='\t')
        else:
            df = pd.read_csv(self.alignment_path, header=None, names=["SrcEntity", "TgtEntity"], usecols=[0,1])
        
        for i, row in df.iterrows():

            self.alignments.append(Alignment(self.source_onto, self.target_onto
                                    ,row["SrcEntity"].split("/")[-1].split("#")[-1].split(";")[-1].split(":")[-1]
                                    ,row["TgtEntity"].split("/")[-1].split("#")[-1].split(";")[-1].split(":")[-1]))

    def get_source_filename(self):
        return self.source_ontology_metadata.path

    def get_target_filename(self):
        return self.target_ontology_metadata.path

    def _parse_alignment(self, cell, source, target):
        return Alignment(source,
                        target,
                        self._parse_entity(cell.getElementsByTagName('entity1')[0]),
                        self._parse_entity(cell.getElementsByTagName('entity2')[0]), 
                        relation = cell.getElementsByTagName('relation')[0].firstChild.nodeValue)

    def _parse_entity(self, entity):
        return entity.getAttribute('rdf:resource').split("#")[-1].split(";")[-1]

    def load_alignments(self, only_concepts = True):

        self.alignments = []

        doc = minidom.parse(self.alignment_path)

        if doc.getElementsByTagName('Ontology').length == 0:
            first_cell = doc.getElementsByTagName('Cell')[0]
            onto1= first_cell.getElementsByTagName('entity1')[0].getAttribute('rdf:resource').split("#")[0].split("/")[-1].split(".")[0]
            onto2= first_cell.getElementsByTagName('entity2')[0].getAttribute('rdf:resource').split("#")[0].split("/")[-1].split(".")[0]

            self.source_ontology_metadata = OntologyMetadata(id=onto1)
            self.target_ontology_metadata = OntologyMetadata(id=onto2)
        else:
            onto1 = doc.getElementsByTagName('Ontology')[0]
            onto2 = doc.getElementsByTagName('Ontology')[1]

            self.source_ontology_metadata = OntologyMetadata(tag=onto1)
            self.target_ontology_metadata = OntologyMetadata(tag=onto2)



        alignments = [self._parse_alignment(cell, self.source_ontology_metadata.id, self.target_ontology_metadata.id) \
                        for cell in doc.getElementsByTagName('Cell')]
        
        if only_concepts:
            alignments = [alignment for alignment in alignments if alignment.is_class()]

        
        self.alignments.extend(alignments)

    def get_prediction_metrics(self, predictions):
        tp, fp, fn = 0, 0, 0

        tp = sum([pred in self.alignments for pred in predictions])
        fp = len(predictions) - tp
        fn = len(self.alignments) - tp

        return tp, fp, fn

        

class OntologyMetadata:
    def __init__(self, tag=None, id=None) -> None:
        self.id = id


        if tag is not None:

            if len(tag.getElementsByTagName("path")) > 0:
                self.path = tag.getElementsByTagName("path")[0].firstChild.nodeValue
                self.id = self.path.split("/")[-1].split(".")[0]

            else:

                self.url = tag.getAttribute("rdf:about")
                self.id = tag.getAttribute("rdf:about").split("/")[-1].rsplit(".", 1)[0]

                if len(tag.getElementsByTagName("rdfs:label")) > 0:
                    self.label = tag.getElementsByTagName("rdfs:label")[0].firstChild.nodeValue
                else:
                    self.label = self.id

                self.path = self.id + ".owl"
