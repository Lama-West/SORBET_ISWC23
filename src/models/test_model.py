


from ctypes import alignment
import os
from re import S
from typing import List
from torch import embedding
from transformers import AutoModelForSequenceClassification, BertTokenizer, BertForPreTraining, DistilBertTokenizer, \
                         DistilBertForMaskedLM, BertForMaskedLM,AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
import gc
import numpy as np
from models.base_model import BaseModel
from losses.SentenceEmbeddingLoss import SentenceEmbeddingLoss
from losses.InitialEmbeddingLoss import InitialEmbeddingLoss
from sklearn.model_selection import train_test_split
import random 

from batch_loaders.alignment_batch_loader import DAEBatchLoader, DAEBatchLoaderInference, ReferenceAlignmentsBatchLoader
from batch_loaders.alignment import Alignment, AlignmentDataset
from batch_loaders.ontology_parsing.ontology import Ontology
from batch_loaders.ontology_parsing.ontology_config import OntoConfig
from batch_loaders.MLM_batch_loader import EvaluationTreeWalkBatchLoader, RandomTreeWalkBatchLoader
from losses.CosineSimilarityLoss import CosineSimilarityLoss
from losses.ClassificationObjectiveLoss import ClassificationObjectiveLoss
from batch_loaders.pair_alignment_batch_loader import MatrixRefinementBatchLoader, PairAlignmentBatchLoader
from models.mlm_model import MLMOntoBert
from models.siamese_model import SiameseOntoBert
from globals import Globals
import math

import logging

logger = logging.getLogger("onto")

class DAE(nn.Module):
    def __init__(self):
        super(DAE,self).__init__()
        self.encoder=nn.Sequential(
                    nn.Linear(768,32),
                    nn.ReLU(),
            
                    )
        
        self.decoder=nn.Sequential(
                  nn.Linear(32,768),
                  )
    
 
    def forward(self, x):
        noise_mask = ((1/0.3)*torch.rand(*x.shape)).int().bool().to(Globals.device)

        x=self.encoder(x*noise_mask)
        x=self.decoder(x)

        return x


    def encode(self, x):
        
        x = self.encoder(x)

        return x




class TestModel(BaseModel):
    def __init__(self, from_pretrained="emilyalsentzer/Bio_ClinicalBERT", threshold_dae=0.6) -> None:
        super().__init__()
        
        self.is_supervised = True

        self.siamese_model = SiameseOntoBert(from_pretrained=from_pretrained)
        self.cache = {}

        self.loader_type = DAEBatchLoader

        self.model = DAE().to(Globals.device)
        self.threshold_dae = threshold_dae
        self.loss_fn = nn.MSELoss()

    def get_loss(self, batch_alignments: List[Alignment], onto_map):
        
        x = None
        y = None
        for a in batch_alignments:
            if random.random() < 0.5:
                a.inverse()

            if a.source_onto not in self.siamese_model.embeddings_cache:
                self.siamese_model.get_embeddings(onto_map[a.source_onto])
            if a.target_onto not in self.siamese_model.embeddings_cache:
                self.siamese_model.get_embeddings(onto_map[a.target_onto])

            if x is None:
                x = self.siamese_model.get_concept_embedding(a.source_onto, a.id1).view(1,-1)
                y = self.siamese_model.get_concept_embedding(a.target_onto, a.id2).view(1,-1)

            else:
                x = torch.cat((x , self.siamese_model.get_concept_embedding(a.source_onto, a.id1).view(1,-1)), 0)
                y = torch.cat((y , self.siamese_model.get_concept_embedding(a.target_onto, a.id2).view(1,-1)), 0)

        x_reconstructed = self.model(x)

        loss_x = self.loss_fn(x_reconstructed , x)
        loss_y = self.loss_fn(x_reconstructed , y)
        # loss_x = -torch.mean(x*torch.log(x_reconstructed) + (1-x)*torch.log(1-x_reconstructed))
        # loss_y = -torch.mean(y*torch.log(x_reconstructed) + (1-y)*torch.log(1-x_reconstructed))

        loss = loss_x + loss_y

        return {"reconstruction loss": loss}

    def get_dae_embeddings(self, ontology):
        loader = DAEBatchLoaderInference(ontology)
        embeddings = None

        with torch.no_grad():
            for batch_inputs in tqdm(loader, leave=False, total=len(loader), disable=True):
                
                inputs = torch.stack([self.siamese_model.get_concept_embedding(ontology.onto_name, id) for id in batch_inputs]).to(Globals.device)

                outputs = self.model(inputs)

                if embeddings is None:
                    embeddings = outputs
                else:
                    embeddings = torch.cat((embeddings, outputs))

        return embeddings

    def get_parents(self, onto, root):
        if onto not in self.cache:
            self.cache[onto.onto_name] = {}

        if root not in self.cache[onto.onto_name]:
            self.cache[onto.onto_name][root] = onto.get_node_neighbours(root, k_hop_max=4, add_childs=False, add_obj_prop=False, flatten=True)

            self.cache[onto.onto_name][root] = [c for c in self.cache[onto.onto_name][root] if ((c != "Thing") and (c != "[UNK]"))]

        return self.cache[onto.onto_name][root]

    def get_childs(self, onto, root):
        if onto not in self.cache:
            self.cache[onto.onto_name] = {}

        if root not in self.cache[onto.onto_name]:
            self.cache[onto.onto_name][root] = onto.get_node_neighbours(root, k_hop_max=2, add_parents=False, add_obj_prop=False, flatten=True)

            self.cache[onto.onto_name][root] = [c for c in self.cache[onto.onto_name][root] if ((c != "Thing") and (c != "[UNK]"))]

        return self.cache[onto.onto_name][root]

    def get_embeddings(self, ontology, walk_config=None):
        return self.siamese_model.get_embeddings(ontology, walk_config)



    def get_similarity_matrix(self, source: Ontology, target: Ontology, walk_config=None):
        src_embeddings, src_keys = self.siamese_model.get_embeddings(source, walk_config)
        trg_embeddings, trg_keys = self.siamese_model.get_embeddings(target, walk_config)

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

                        parents1 = self.get_parents(source, src_key)
                        parents2 = self.get_parents(target, trg_key)


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



    
