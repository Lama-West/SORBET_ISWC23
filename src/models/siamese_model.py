


from ctypes import alignment
import os
from typing import List
from torch import embedding
from transformers import BertTokenizer, BertForPreTraining, DistilBertTokenizer, \
                         DistilBertForMaskedLM, BertForMaskedLM,AutoTokenizer, AutoModel, BertModel
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
import pandas as pd

from batch_loaders.pair_alignment_batch_loader import SiameseEmbeddingsBatchLoader
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

class SiameseOntoBert(BaseModel):
    def __init__(self, from_pretrained="emilyalsentzer/Bio_ClinicalBERT",  pooling_strategy="only_concept", 
                       loss_fnc="cosine_loss", freeze_embeddings=False) -> None:
        super().__init__()

        self.is_supervised = True
        self.loader_type=SemiNegativeSampling

        self.embeddings_cache = {}
        self.freeze_embeddings = freeze_embeddings


        self.loss_fnc_name = loss_fnc
        if (loss_fnc is None) or (loss_fnc=="cosine_loss"):
            self.loss_fnc = CosineSimilarityLoss()
        elif loss_fnc=="classification":
            self.loss_fnc = ClassificationObjectiveLoss()
        else:
            raise NotImplementedError()

        self.divergence_loss = SentenceEmbeddingLoss()
        self.subclass_loss = SubclassLoss()

        if from_pretrained == "none":
            self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", add_pooling_layer=False, output_hidden_states=True).to(Globals.device)
            self.model = AutoModel.from_config(self.model.config).to(Globals.device)
            self.model.init_weights()
        else:
            if "deberta" in from_pretrained:
                self.model = AutoModel.from_pretrained(from_pretrained).to(Globals.device)
            else:
                self.model = AutoModel.from_pretrained(from_pretrained, add_pooling_layer=False, output_hidden_states=True).to(Globals.device)

        self.pretrained_model = from_pretrained

        self.pooling_strategy = pooling_strategy

        if pooling_strategy == "cls":
            self._pool_function = self._cls_pooling

        elif pooling_strategy == "only_concept":
            self._pool_function = self._only_concept_pooling

        elif pooling_strategy == "only_concept_plus_mlp":
            
            self.pool_mlp = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size).to(Globals.device)
            self._pool_function = self._only_concept_plus_mlp

        else:
            raise NotImplementedError()
        
        if self.freeze_embeddings:
            for param in self.model.parameters():
                param.requires_grad = False

        self.divergence_loss_alpha = 0.2

        self.subclass_loss_alpha = 0.0005

        self.classification_loss_alpha = 1

    def parameters(self):
        
        if self.freeze_embeddings:
            parameters = []
        else:
            parameters = list(self.model.parameters())

        if self.pooling_strategy == "only_concept_plus_mlp":
            parameters = parameters + list(self.pool_mlp.parameters())
        
        if self.loss_fnc_name=="classification":
            parameters = parameters + list(self.loss_fnc.fc1.parameters())

        return parameters

    def get_embeddings_dict(self):
        return self.embeddings_cache

    def reset_embedding_dict(self):

        if self.freeze_embeddings:
            return
        
        self.embeddings_cache = {}

    def save_model(self, path):
        self.model.save_pretrained(path)
    
    def _save_checkpoint(self, config, epoch):
        checkpoint_folder = os.path.join(config["final_path"], "checkpoints", f"epoch_{epoch}")
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        self.model.save_pretrained(checkpoint_folder)
        

    def get_loss(self, source_inputs, src_root_pooling_mask, src_subclass_pairs, src_concept_pool_masks, src_ontos, src_concept_ids, \
                       target_inputs, trg_root_pooling_mask, trg_subclass_pairs, trg_concept_pool_masks, trg_ontos, trg_concept_ids, labels):
        source_outputs = self.model(**source_inputs)
        target_outputs = self.model(**target_inputs)


        subclass_loss = 0

        source_embeddings = []
        for i, output in enumerate(source_outputs.last_hidden_state):
                source_embeddings.append(self._pool_function(output, src_root_pooling_mask[i]))
                # for subclass_pair in src_subclass_pairs[i]:
                #     subclass_loss += self.subclass_loss(output, subclass_pair)
        source_embeddings = torch.cat(source_embeddings)

        target_embeddings = []
        for i, output in enumerate(target_outputs.last_hidden_state):
                target_embeddings.append(self._pool_function(output, trg_root_pooling_mask[i]))
                # for subclass_pair in trg_subclass_pairs[i]:
                #     subclass_loss += self.subclass_loss(output, subclass_pair)
        target_embeddings = torch.cat(target_embeddings)


        # div_loss = self.divergence_loss_alpha * (self.divergence_loss(source_outputs.hidden_states[-1], src_concept_pool_masks, src_concept_ids) + \
        #                                          self.divergence_loss(target_outputs.hidden_states[-1], trg_concept_pool_masks, trg_concept_ids) )

        mse_loss = self.classification_loss_alpha * self.loss_fnc(source_embeddings, target_embeddings, labels) 
        # subclass_loss = self.subclass_loss_alpha * subclass_loss

        return {"cosine": mse_loss} 

    def _cls_pooling(self, output, masks):
        return output[0].view(1,-1)

    def _only_concept_pooling(self, output, mask):
        return torch.mean(output[mask.squeeze()], dim=0).view(1,-1)
    
    def _only_concept_plus_mlp(self, output, mask):
        embedding = torch.mean(output[mask.squeeze()], dim=0).view(1,-1)

        return self.pool_mlp(embedding)


    def _mean_pooling(self, output, masks):
        pass

    def get_embeddings(self, ontology, candidates = None, walk_config=None):

        if ontology.onto_name in self.embeddings_cache:

            return self.embeddings_cache[ontology.onto_name]["embeddings"], self.embeddings_cache[ontology.onto_name]["embeddings_key"]


        batch_loader = EvaluationTreeWalkBatchLoader(ontology, candidates=candidates, walk_config=walk_config)

        embeddings = None
        embeddings_key = []


        with torch.no_grad():

            for batch_inputs, batch_nodes, batch_nodes_masks in tqdm(batch_loader, leave=False, total=len(batch_loader), disable=True):

                outputs = self.model(**batch_inputs)

                for i, output in enumerate(outputs.last_hidden_state):
                    if embeddings == None:
                        embeddings = self._pool_function(output, batch_nodes_masks[i])
                    else:
                        embeddings = torch.cat((embeddings, self._pool_function(output, batch_nodes_masks[i])))

                embeddings_key.extend(batch_nodes)

                del outputs
                gc.collect()


        self.embeddings_cache[ontology.onto_name] = {}
        self.embeddings_cache[ontology.onto_name]["embeddings"] = embeddings
        self.embeddings_cache[ontology.onto_name]["embeddings_key"] = embeddings_key
        self.embeddings_cache[ontology.onto_name]["id_to_index"] = {id: i for i, id in enumerate(embeddings_key)}
        self.embeddings_cache[ontology.onto_name]["embeddings_label"] = [f"{ontology.onto_name}#" + " ".join(ontology.get_id_label(id)) for id in embeddings_key]


        return embeddings, embeddings_key


    def get_concept_embedding(self, onto, id):
        i = self.embeddings_cache[onto]["id_to_index"][id]
        return torch.clone(self.embeddings_cache[onto]["embeddings"][i])


    def get_similarity_matrix(self, source: Ontology, target: Ontology, walk_config=None, user_input_candidates=None):

        candidates_src=None
        candidates_trg=None

        if user_input_candidates is not None:
            candidates_src = list(set(user_input_candidates.keys()))
            candidates_trg = set()
            for candidates in user_input_candidates.values():
                candidates_trg.update(candidates)
            candidates_trg = list(candidates_trg)
            
        src_embeddings, src_keys = self.get_embeddings(source, candidates=candidates_src, walk_config=walk_config)
        trg_embeddings, trg_keys = self.get_embeddings(target, candidates=candidates_trg , walk_config=walk_config)


        if self.loss_fnc_name == "classification":
            similarities = self.get_similarity_matrix_classification(src_embeddings, src_keys, trg_embeddings, trg_keys)
        else:
            similarities = F.normalize(src_embeddings,dim=1) @ F.normalize(trg_embeddings,dim=1).t()

        if user_input_candidates is not None:
            src_key_to_idx = {key: i for i, key in enumerate(src_keys)}
            trg_key_to_idx = {key: i for i, key in enumerate(trg_keys)}


            for src_key in src_keys:
                if src_key not in user_input_candidates:
                    similarities[src_key_to_idx[src_key]] = -torch.inf
                else:
                    mask = torch.full((len(trg_keys),), -torch.inf).to(Globals.device)
                    for trg_key in user_input_candidates[src_key]:
                        mask[trg_key_to_idx[trg_key]] = 0. 

                    similarities[src_key_to_idx[src_key]] = mask+similarities[src_key_to_idx[src_key]]

                

        return similarities, src_keys, trg_keys
    
    def get_similarity_matrix_classification(self, src_embeddings, src_keys, trg_embeddings, trg_keys):
        

        similarities = torch.zeros((len(src_keys), len(trg_keys)))

        batch_loader = SiameseEmbeddingsBatchLoader(src_embeddings, src_keys,  trg_embeddings, trg_keys)


        with torch.no_grad():

            for batch_src_embeddings, batch_trg_embeddings, batch_indices in tqdm(batch_loader, leave=False, total=len(batch_loader), disable=True):

                outputs = self.loss_fnc.forward_mlp(batch_src_embeddings, batch_trg_embeddings)

                for k, index in enumerate(batch_indices):
                    similarities[index[0],index[1]] = outputs[k]

        return similarities






    
    



