


from ctypes import alignment
import os
from typing import List
from torch import embedding
from transformers import BertTokenizer, BertForPreTraining, DistilBertTokenizer, \
                         DistilBertForMaskedLM, BertForMaskedLM,AutoTokenizer, AutoModel, BertModel
import torch
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
import gc
import numpy as np
from models.base_model import BaseModel
from losses.SentenceEmbeddingLoss import SentenceEmbeddingLoss
from losses.InitialEmbeddingLoss import InitialEmbeddingLoss
from sklearn.model_selection import train_test_split

from batch_loaders.alignment_batch_loader import SemiNegativeSampling, ReferenceAlignmentsBatchLoader
from batch_loaders.alignment import Alignment, AlignmentDataset
from batch_loaders.ontology_parsing.ontology import Ontology
from batch_loaders.ontology_parsing.ontology_config import OntoConfig
from batch_loaders.MLM_batch_loader import EvaluationTreeWalkBatchLoader, RandomTreeWalkBatchLoader
from losses.CosineSimilarityLoss import CosineSimilarityLoss
from losses.ClassificationObjectiveLoss import ClassificationObjectiveLoss
from losses.SubclassLoss import SubclassLoss
from globals import Globals
import math
import logging 
logger = logging.getLogger("onto")



class StructureSimilarityDecorator():
    def __init__(self, base_model = None) -> None:
        super().__init__()

        self.base_model = base_model

        self.child_cache = {}
        self.parent_cache = {}


    def _get_parents(self, onto, root):
        if onto not in self.parent_cache:
            self.parent_cache[onto.onto_name] = {}

        if root not in self.parent_cache[onto.onto_name]:
            self.parent_cache[onto.onto_name][root] = onto.get_node_neighbours(root, k_hop_max=4, add_childs=False, add_obj_prop=False, flatten=True)

            self.parent_cache[onto.onto_name][root] = [c for c in self.parent_cache[onto.onto_name][root] if ((c != "Thing") and (c != "[UNK]"))]

        return self.parent_cache[onto.onto_name][root]

    def _get_childs(self, onto, root):
        if onto not in self.child_cache:
            self.child_cache[onto.onto_name] = {}

        if root not in self.child_cache[onto.onto_name]:
            self.child_cache[onto.onto_name][root] = onto.get_node_neighbours(root, k_hop_max=2, add_parents=False, add_obj_prop=False, flatten=True)

            self.child_cache[onto.onto_name][root] = [c for c in self.child_cache[onto.onto_name][root] if ((c != "Thing") and (c != "[UNK]"))]

        return self.child_cache[onto.onto_name][root]


    def get_similarity_matrix(self, source: Ontology, target: Ontology, walk_config=None):
        src_embeddings, src_keys = self.base_model.get_embeddings(source, walk_config)
        trg_embeddings, trg_keys = self.base_model.get_embeddings(target, walk_config)

        similarities = F.normalize(src_embeddings,dim=1) @ F.normalize(trg_embeddings,dim=1).t()
        
        
        def max_sim_parent(key_to_index_src, key_to_index_trg, parents1, parents2, sims):

            all_sims = [sims[key_to_index_src[src_parent], key_to_index_trg[trg_parent]] for src_parent in parents1 for trg_parent in parents2]

            return max(all_sims)

        key_to_index_src = {id: i for i, id in enumerate(src_keys)}
        key_to_index_trg = {id: i for i, id in enumerate(trg_keys)}

        updated_similarities = torch.clone(similarities)

        for src_key in src_keys:
            if src_key in source.parents_map:
                for trg_key in trg_keys:
                    if trg_key in target.parents_map:
                        i_1 = key_to_index_src[src_key]
                        i_2 = key_to_index_trg[trg_key]

                        parents1 = self._get_parents(source, src_key)
                        parents2 = self._get_parents(target, trg_key)


                        if (len(parents1) > 0) and (len(parents2) > 0):
                            

                            sim_parent = max_sim_parent(key_to_index_src, key_to_index_trg, parents1, parents2, similarities)

                            updated_similarities[i_1, i_2] = min(1.0, similarities[i_1, i_2] + math.tanh(sim_parent - 0.75)/2)

                        if (len(parents1) == 0) and (len(parents2) == 0):
                            updated_similarities[i_1, i_2] = min(1.0, similarities[i_1, i_2] + math.tanh(0.8 - 0.75))
                        else:
                            updated_similarities[i_1, i_2] = min(1.0, similarities[i_1, i_2] + math.tanh(0.70 - 0.75)/2)

                        # childs1 = self.get_childs(source, src_key)
                        # childs2 = self.get_childs(target, trg_key)

                        # if (len(childs1) > 0) and (len(childs2) > 0):
                        #     sim_chilren = max_sim_parent(key_to_index_src, key_to_index_trg, childs1, childs2, similarities)

                        #     updated_similarities[i_1, i_2] = min(1.0, similarities[i_1, i_2] + math.tanh(sim_chilren - 0.75)/2)


        return updated_similarities, src_keys, trg_keys
    

    def get_embeddings(self, ontology, walk_config=None):
        return self.base_model.get_embeddings(ontology, walk_config=walk_config)

    def __getattr__(self, attr):
        # Forward all other method calls to the original model class
        return getattr(self.base_model, attr)

