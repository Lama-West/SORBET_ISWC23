import random, copy, os
from typing import Dict, List, Tuple
from numpy import dtype
import torch
from math import ceil
from globals import Globals

from .alignment_batch_loader import AlignmentBatchLoader, SemiNegativeSampling

from .alignment import Alignment, AlignmentDataset
from .ontology_parsing.ontology_config import OntoConfig
from .ontology_parsing.ontology import Ontology
from .random_walk import RandomWalk, PairRandomWalk, RandomWalkConfig
import logging

class PairAlignmentBatchLoader(SemiNegativeSampling):
    def __init__(self, walk_config: RandomWalkConfig=None, **kwargs):
        super().__init__(**kwargs)

        self.walk_config = walk_config

    def build_inputs(self, batch_alignments):

        random_walks_pairs = [PairRandomWalk(\
                                RandomWalk(self.ontologies_map[a.source_onto], first_node=a.id1, walk_config=self.walk_config),\
                                RandomWalk(self.ontologies_map[a.target_onto], first_node=a.id2, walk_config=self.walk_config))\
                              .build_pooling_mask()\
                              .build_mlm_mask()\
                              for a in batch_alignments]

        batch = Globals.tokenizer([w.sentence for w in random_walks_pairs], return_tensors='pt',
                   max_length=512, truncation=True, padding='max_length').to(Globals.device)


        mask_array = torch.cat([w.mlm_mask for w in random_walks_pairs], dim=0).to(Globals.device)

        batch.input_ids = torch.where(mask_array, Globals.tokenizer.mask_token_id, batch.input_ids)

        batch["labels"] = torch.tensor([int(a.label) for a in batch_alignments], dtype=torch.float).to(Globals.device)


        return batch, [w.concept_pool_masks for w in random_walks_pairs], [w.concepts for w in random_walks_pairs],\
                      [w.ontos for w in random_walks_pairs]





class SiameseEmbeddingsBatchLoader:
    def __init__(self, src_embeddings, src_keys,  trg_embeddings, trg_keys, batch_size=64) -> None:

        self.batch_size = batch_size

        self.src_embeddings = src_embeddings
        self.trg_embeddings = trg_embeddings
        self.src_keys = src_keys
        self.trg_keys = trg_keys

        self.indices = [(i,j) for i in range(len(src_keys)) for j in range(len(trg_keys))]


    def __getitem__(self, idx):
            
        if idx >= self.__len__():
            raise IndexError

        batch_indices = self.indices[(idx*self.batch_size) : min((idx+1)*self.batch_size, len(self.indices) )]
        src_indices = [i for i,j in batch_indices]
        trg_indices = [j for i,j in batch_indices]
        batch_src_embeddings = self.src_embeddings[src_indices,:]
        batch_trg_embeddings = self.trg_embeddings[trg_indices,:]

        return batch_src_embeddings, batch_trg_embeddings, batch_indices

    def __len__(self):
        return ceil(len(self.indices) / self.batch_size) 




class MatrixRefinementBatchLoader:
    def __init__(self, ontologies_map, similarities, topk, src_onto, 
                trg_onto, src_keys, trg_keys, exclude_indices=None, batch_size=64, walk_config:RandomWalkConfig = None) -> None:

        self.walk_config = walk_config
        self.batch_size = batch_size
        self.ontologies_map = ontologies_map

        self.indices = torch.topk(similarities, k=topk, sorted=False).indices.tolist()

        self.indices = [(i,j) for i, topks in enumerate(self.indices) for j in topks]
        # topk_thresholds = torch.min(torch.topk(similarities, k=topk, sorted=False).values, dim=-1).values
        # topk_thresholds = topk_thresholds.unsqueeze(1).expand(-1, similarities.shape[-1])
        # src_indexes, trg_indexes = torch.where(similarities >= topk_thresholds, similarities.double(), 0.).nonzero(as_tuple=True)

        # self.indices = list(zip(src_indexes.tolist(), trg_indexes.tolist()))
        if exclude_indices is not None:
            self.indices = list(filter(lambda idx: idx[0] not in exclude_indices, self.indices))

        self.src_onto = src_onto
        self.trg_onto = trg_onto
        self.src_keys = src_keys
        self.trg_keys = trg_keys


    def __getitem__(self, idx):
            
        if idx >= self.__len__():
            raise IndexError

        batch_indices = self.indices[(idx*self.batch_size) : min((idx+1)*self.batch_size, len(self.indices) )]
        batch_alignments = [Alignment(self.src_onto, self.trg_onto, self.src_keys[i], self.trg_keys[j]) for i, j in batch_indices]

        random_walks_pairs = [PairRandomWalk(\
                                RandomWalk(self.ontologies_map[a.source_onto], first_node=a.id1, walk_config=self.walk_config),\
                                RandomWalk(self.ontologies_map[a.target_onto], first_node=a.id2, walk_config=self.walk_config))
                              for a in batch_alignments]
        
        batch = Globals.tokenizer([w.sentence for w in random_walks_pairs], return_tensors='pt',
                   max_length=512, truncation=True, padding='max_length').to(Globals.device)

        return batch, batch_indices

    def __len__(self):
        return ceil(len(self.indices) / self.batch_size) 

