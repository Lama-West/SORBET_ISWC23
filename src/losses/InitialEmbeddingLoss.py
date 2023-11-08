from typing import List
import torch
from torch import nn, Tensor
import random

from transformers import AutoTokenizer, AutoModel

from batch_loaders.ontology_parsing.ontology import Ontology
from globals import Globals
from models.base_model import BaseModel

class InitialEmbeddingLoss(nn.Module):

    id_sentence_embedding = {}
    onto_embeddings = None

    def __init__(self, loss_fct = nn.MSELoss()):
        super(InitialEmbeddingLoss, self).__init__()
        self.loss_fct = loss_fct




    def forward(self, output, concept_pool_masks, concept_ids, ontos):
        if InitialEmbeddingLoss.id_sentence_embedding is None:
            raise ValueError("Error when calculating loss: must call _precompute_sentence_embeddings first")

        loss = 0.0
        for i, ids in enumerate(concept_ids):
            for j, id in enumerate(ids):

                if id not in InitialEmbeddingLoss.id_sentence_embedding[ontos[i][j]]:
                    continue

                loss += self.loss_fct(torch.mean(output[i][concept_pool_masks[i][j].view(-1)], dim=0).view(1,-1), \
                                      InitialEmbeddingLoss.id_sentence_embedding[ontos[i][j]][id].to(Globals.device) )
        
        loss = loss / sum([len(concept) for concept in concept_ids])
        
        return loss

    def _precompute_sentence_embeddings(model: BaseModel, ontologies: List[Ontology]):
        
        for onto in ontologies:
            InitialEmbeddingLoss.id_sentence_embedding[onto.get_name()] = {}
            embeddings, keys = model.get_node_tree_embedding(onto)
            for i, id in enumerate(keys):
                InitialEmbeddingLoss.id_sentence_embedding[onto.get_name()][id] = embeddings[i]
            