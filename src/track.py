
from batch_loaders.alignment import Alignment, AlignmentDataset
from batch_loaders.ontology_parsing.ontology import Ontology
from batch_loaders.ontology_parsing.ontology_config import OntoConfig
from batch_loaders.ontology_parsing.preprocessing import preprocessor
from metrics import Metrics, Results
from sklearn.model_selection import train_test_split

from typing import List
import os
import copy
import logging
import copy 

logger = logging.getLogger("onto")



class Track():
    def __init__(self, name, config, metrics_config) -> None:

        self.name = name

        self.metrics = Metrics(**metrics_config, track_name=name)

        track_config = config[name]
        self.track_config = track_config

        self.load_ontologies(track_config)


        self.alignment_per_dataset = []
        self.alignments = []
        self.train_alignments = []
        
        if "alignments_folder" in track_config:
            self.load_alignments(track_config)

                    
    def load_ontologies(self, track_config):

        ontologies_folder = track_config['ontologies_folder']

        ontologies_filenames = self._load_folder_or_list(track_config, "ontologies", ontologies_folder) 



        if "accronym_resolution" in track_config['parsing_parameters'] and track_config['parsing_parameters']["accronym_resolution"]:
            temp_ontologies = [Ontology(path, prefix=ontologies_folder, 
                               ontology_config=OntoConfig(track_config['parsing_parameters']),
                               fast=True) 
                    for path in ontologies_filenames]

            preprocessor.fit_to_vocab([onto.get_id_label(id) for onto in temp_ontologies for id in onto.classes])
            preprocessor.export_accronym_resolution_dict()

        self.ontologies = [Ontology(path, prefix=ontologies_folder, 
                               ontology_config=OntoConfig(track_config['parsing_parameters'])) 
                    for path in ontologies_filenames]

        self.ontologies_map = {o.get_name(): o for o in self.ontologies}


    def load_alignments(self, track_config):
        
        if "type" not in track_config:
            self.type = "OA"
        else:
            self.type = track_config["type"]

        alignments_folder = track_config["alignments_folder"]
        alignments_filenames = self._load_folder_or_list(track_config, "alignments", alignments_folder)
        
        if "ontologies_in_alignment" in track_config:
            ontologies_in_alignment = track_config["ontologies_in_alignment"]
            self.alignment_per_dataset: List[AlignmentDataset] = [AlignmentDataset(os.path.join(alignments_folder, path),
                                                                                    source_onto=ontologies_in_alignment[0],
                                                                                    target_onto=ontologies_in_alignment[1],
                                                                                    type = self.type)
                                                                                    for i, path in enumerate(alignments_filenames)]
        else:
            self.alignment_per_dataset: List[AlignmentDataset] = [AlignmentDataset(os.path.join(alignments_folder, path), type = self.type) for i, path in enumerate(alignments_filenames)]


        self.alignments = [alignment for alignment_dataset in self.alignment_per_dataset for alignment in alignment_dataset.alignments]

        self.track_config = track_config

        if "test_alignments" in track_config:
            if "ontologies_in_alignment" not in track_config:
                raise ValueError("Must specifiy ontologies_in_alignment in config.json")

            self.train_alignments = copy.deepcopy(self.alignments)        

            ontologies_in_alignment = track_config["ontologies_in_alignment"]

            self.alignment_per_dataset = AlignmentDataset(os.path.join(alignments_folder, track_config["test_alignments"]),
                                                                    source_onto=ontologies_in_alignment[0],
                                                                    target_onto=ontologies_in_alignment[1], type = self.type)

            self.alignment_per_dataset.split_train_test(set())
            self.alignment_per_dataset = [self.alignment_per_dataset]
        
        if "candidates" in track_config:
            for i, alignment_dataset in enumerate(self.alignment_per_dataset):
                alignment_dataset.add_candidates(os.path.join(alignments_folder, track_config["candidates"][i]))


    def split_train_test(self, test_size, consider_train_set=False):
        if "test_alignments" in self.track_config:
            return

        if test_size >= 1.:
            self.train_alignments = []
            self.test_alignments = self.alignments
        else:

            if not consider_train_set:
                self.train_alignments, self.test_alignments = train_test_split(self.alignments, test_size=test_size, random_state=42)
            else:
                self.train_alignments, _ = train_test_split(self.alignments, test_size=test_size, random_state=42)
                self.test_alignments = self.alignments
        
        if not consider_train_set:
            train_set = set(self.train_alignments)
        else:
            train_set = set()
            
        for alignments_dataset in self.alignment_per_dataset:
            alignments_dataset.split_train_test(train_set)

    
    def _load_folder_or_list(self, config, list_key, folder):
        if list_key in config:
            return config[list_key]
        else:
            return [path for path in os.listdir(folder) if os.path.isfile(os.path.join(folder,path)) ]

    def __add__(self, other_track):
        if not isinstance(other_track, Track):
            raise ValueError()

        new_track = copy.copy(self)

        new_track.ontologies_map = self.ontologies_map | other_track.ontologies_map
        new_track.ontologies = self.ontologies + other_track.ontologies
        new_track.alignments = self.alignments + other_track.alignments
        new_track.train_alignments = self.train_alignments + other_track.train_alignments
        new_track.alignment_per_dataset = self.alignment_per_dataset + other_track.alignment_per_dataset


        return new_track
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

