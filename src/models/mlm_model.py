


from ctypes import alignment
import os
from typing import List
from torch import embedding
from transformers import BertTokenizer, BertForPreTraining, DistilBertTokenizer, \
                         DistilBertForMaskedLM, BertForMaskedLM,AutoTokenizer, AutoModel, RobertaForMaskedLM

import torch
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
import gc
import numpy as np
from globals import Globals
from models.base_model import BaseModel
from losses.SentenceEmbeddingLoss import SentenceEmbeddingLoss
from sklearn.model_selection import train_test_split

from batch_loaders.alignment_batch_loader import ReferenceAlignmentsBatchLoader
from batch_loaders.alignment import Alignment, AlignmentDataset
from batch_loaders.ontology_parsing.ontology import Ontology
from batch_loaders.ontology_parsing.ontology_config import OntoConfig
from batch_loaders.MLM_batch_loader import EvaluationTreeWalkBatchLoader, RandomTreeWalkBatchLoader
from losses.CosineSimilarityLoss import CosineSimilarityLoss
from losses.SentenceEmbeddingLoss import SentenceEmbeddingLoss
from losses.ClassificationObjectiveLoss import ClassificationObjectiveLoss



class MLMOntoBert(BaseModel):
    def __init__(self, from_pretrained="emilyalsentzer/Bio_ClinicalBERT") -> None:
        super().__init__()

        self.loader_type=RandomTreeWalkBatchLoader
        self.is_supervised = False

        self.model = BertForMaskedLM.from_pretrained(from_pretrained, output_hidden_states=True)

        self.divergence_loss = SentenceEmbeddingLoss()
        self.divergence_loss_alpha = 2

        self.mlm_loss_alpha = 0.3
        self.concept_id_map = None

        self.embeddings_cache = {}

    def get_loss(self, batch, concept_pool_masks, concept_ids):

        outputs = self.model(**batch)
        bert_embeddings = outputs.hidden_states[1]

        mlm_loss = self.mlm_loss_alpha * outputs.loss
        kl_div_loss = self.divergence_loss_alpha * self.divergence_loss(bert_embeddings, concept_pool_masks, concept_ids)

        return mlm_loss, kl_div_loss
               

    def get_embeddings_dict(self):
        return self.embeddings_cache

    def reset_embedding_dict(self):
        self.embeddings_cache = {}

    def get_embeddings(self, ontology: Ontology, walk_config=None):
        if ontology.onto_name in self.embeddings_cache:

            return self.embeddings_cache[ontology.onto_name]["embeddings"], self.embeddings_cache[ontology.onto_name]["embeddings_key"]

        batch_loader = EvaluationTreeWalkBatchLoader(ontology, walk_config=walk_config)

        embeddings = None
        embeddings_key = []

        self.model = self.model.to(Globals.device)

        with torch.no_grad():

            for batch_inputs, batch_nodes, batch_nodes_masks in tqdm(batch_loader, leave=False, total=len(batch_loader), disable=True):

                outputs = self.model(**batch_inputs)

                for i, output in enumerate(outputs.hidden_states[-1]):
                    if embeddings == None:
                        embeddings = torch.mean(output[batch_nodes_masks[i]], dim=0).view(1,-1)
                    else:
                        embeddings = torch.cat((embeddings, torch.mean(output[batch_nodes_masks[i]], dim=0).view(1,-1)))

                embeddings_key.extend(batch_nodes)


                del outputs
                gc.collect()

        self.model = self.model.cpu()

        self.embeddings_cache[ontology.onto_name] = {}
        self.embeddings_cache[ontology.onto_name]["embeddings"] = embeddings
        self.embeddings_cache[ontology.onto_name]["embeddings_key"] = embeddings_key
        self.embeddings_cache[ontology.onto_name]["embeddings_label"] = [f"{ontology.onto_name}#" + " ".join(ontology.get_id_label(id)) for id in embeddings_key]

        return embeddings, embeddings_key


    def get_similarity_matrix(self, source: Ontology, target: Ontology, walk_config=None):
        self.model = self.model.to(Globals.device)
        
        trg_embeddings, trg_keys = self.get_embeddings(target, walk_config)
        src_embeddings, src_keys = self.get_embeddings(source, walk_config)

        cosine_similarities = (F.normalize(src_embeddings,dim=1) @ F.normalize(trg_embeddings,dim=1).t()).cpu()

        self.model = self.model.cpu()

        return cosine_similarities, src_keys, trg_keys


    def linear_refinement(self, embeddings, keys, onto: Ontology):
        keys_map = {key: i for i, key in enumerate(keys)}

        new_embeddings = torch.zeros_like(embeddings)
        for i, key in enumerate(keys):
            ancestors = onto.get_all_node_ancestors(key, max_lineage=2, max_height=2)
            indices = [keys_map[node] for node in ancestors]

            new_embeddings[i] = torch.mean(embeddings[indices], dim=0)

        return new_embeddings

    
