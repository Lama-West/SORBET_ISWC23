


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

import gensim
from owlready2 import *
import gensim.downloader as api

class Owl2VecModel(BaseModel):
    def __init__(self, word2vec_only=False) -> None:
        super().__init__()

        if word2vec_only:
            embedding_file = "word2vec-google-news-300"
            self.wv = api.load(embedding_file)
        else:
            embedding_file = "/usagers/frgosb/OWL2Vec_embeddings/outputs/MONDO"
            self.wv = gensim.models.Word2Vec.load(embedding_file).wv



        

        self.is_supervised = False
        self.embeddings_cache = {}
               

    def get_embeddings_dict(self):
        return self.embeddings_cache

    def reset_embedding_dict(self):
        self.embeddings_cache = {}


    def embed_id(self, ontology: Ontology, id):
        if not any([label in self.wv for label in ontology.get_id_label(id)]):
            return np.zeros(self.wv.vector_size)


        return np.mean([self.wv.get_vector(label)  for label in ontology.get_id_label(id) if label in self.wv], axis=0)

    def get_embeddings(self, ontology: Ontology, walk_config=None):
        if ontology.onto_name in self.embeddings_cache:

            return self.embeddings_cache[ontology.onto_name]["embeddings"], self.embeddings_cache[ontology.onto_name]["embeddings_key"]


        embeddings = []
        embeddings_key = []


        embeddings = torch.tensor([self.embed_id(ontology, c) for c in ontology.classes], dtype=torch.float).to(Globals.device)
        embeddings_key = [c for c in ontology.classes]


        self.embeddings_cache[ontology.onto_name] = {}
        self.embeddings_cache[ontology.onto_name]["embeddings"] = embeddings
        self.embeddings_cache[ontology.onto_name]["embeddings_key"] = embeddings_key
        self.embeddings_cache[ontology.onto_name]["embeddings_label"] = [f"{ontology.onto_name}#" + " ".join(ontology.get_id_label(id)) for id in embeddings_key]

        return embeddings, embeddings_key


    def get_similarity_matrix(self, source: Ontology, target: Ontology, walk_config=None):
        
        trg_embeddings, trg_keys = self.get_embeddings(target, walk_config)
        src_embeddings, src_keys = self.get_embeddings(source, walk_config)

        cosine_similarities = (F.normalize(src_embeddings,dim=1) @ F.normalize(trg_embeddings,dim=1).t()).cpu()


        return cosine_similarities, src_keys, trg_keys


    def linear_refinement(self, embeddings, keys, onto: Ontology):
        keys_map = {key: i for i, key in enumerate(keys)}

        new_embeddings = torch.zeros_like(embeddings)
        for i, key in enumerate(keys):
            ancestors = onto.get_all_node_ancestors(key, max_lineage=2, max_height=2)
            indices = [keys_map[node] for node in ancestors]

            new_embeddings[i] = torch.mean(embeddings[indices], dim=0)

        return new_embeddings

    
