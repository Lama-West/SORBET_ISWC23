
import random, copy, os
from numpy import dtype
import torch
from math import ceil
from globals import Globals

from .random_walk import RandomWalk, RandomWalkConfig


class RandomTreeWalkBatchLoader(torch.utils.data.Dataset):
    def __init__(self, ontologies=None, batch_size = 16, walk_config: RandomWalkConfig = None, **kwargs):
        self.batch_size = batch_size
        self.ontos = ontologies

        self.walk_config = walk_config
        
        self.build_dataset()

    def build_dataset(self):
        self.nodes_dataset = [(id, onto) for onto in self.ontos for id in onto.get_classes() if onto.node_has_neighbours(id)]

        random.shuffle(self.nodes_dataset)

    def __getitem__(self, idx):


        if idx >= self.__len__():
            raise IndexError

        # Get random ontology with choice weigthed according to ontologies sizes
        batch_nodes = self.nodes_dataset[(idx*self.batch_size) : min((idx+1)*self.batch_size, len(self.nodes_dataset) )]

        # RandomWalk follows the builder pattern
        random_walks = [RandomWalk(onto, first_node=id, walk_config=self.walk_config)\
                            .build_pooling_mask()\
                            .build_mlm_mask() for id, onto in batch_nodes]

        mask_array = torch.cat([w.mlm_mask for w in random_walks], dim=0).to(Globals.device)

        inputs = Globals.tokenizer([w.sentence for w in random_walks], return_tensors='pt',
                   max_length=512, truncation=True, padding='max_length').to(Globals.device)

        inputs.input_ids = torch.where(mask_array, Globals.tokenizer.mask_token_id, inputs.input_ids)

        labels = Globals.tokenizer([w.sentence for w in random_walks], return_tensors='pt',
                   max_length=512, truncation=True, padding='max_length').input_ids.to(Globals.device)


        # The value -100 will not be calculated in the loss, therefore only loss of masked predictions are calculated
        labels[inputs.input_ids == 0] = -100 
        inputs['labels'] = labels

        

        return inputs, [w.concept_pool_masks for w in random_walks], [w.concepts for w in random_walks]

    def __len__(self):

        return len(self.nodes_dataset) // self.batch_size




########## Evaluation Batch Loader

import logging 
logger = logging.getLogger("onto")

class EvaluationTreeWalkBatchLoader():
    def __init__(self, ontology, candidates=None, batch_size = 64, walk_config = None, **kwargs):

        self.batch_size = batch_size
        self.ontology = ontology

        if candidates is None:
            self.nodes = [c for c in ontology.classes if c not in ontology.not_used_in_alignment]
        else:
            self.nodes = candidates

        self.walk_config = walk_config

        if self.walk_config is None:
            self.walk_config = RandomWalkConfig(n_branches=99, max_path_length=4)
            

    def __getitem__(self, idx):

        if idx >= self.__len__():
            raise IndexError

        batch_nodes = self.nodes[(idx*self.batch_size) : min((idx+1)*self.batch_size, len(self.nodes) )]

        if len(batch_nodes) == 0:
            raise IndexError

        # RandomWalk follows the builder pattern
        random_walks = [RandomWalk(self.ontology, first_node=id, walk_config = self.walk_config).build_pooling_mask() for id in batch_nodes]


        root_masks = torch.cat([w.root_mask for w in random_walks], dim=0).to(Globals.device)

        inputs = Globals.tokenizer([w.sentence for w in random_walks], return_tensors='pt',
                   max_length=512, truncation=True, padding='max_length').to(Globals.device)

        return inputs, batch_nodes, root_masks


    def __len__(self):
        return ceil(len(self.ontology) / self.batch_size) 


