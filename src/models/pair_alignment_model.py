


from ctypes import alignment
import os
from re import S
from typing import List
from torch import embedding
from transformers import AutoModelForSequenceClassification, BertTokenizer, BertForPreTraining, DistilBertTokenizer, \
                         DistilBertForMaskedLM, BertForMaskedLM,AutoTokenizer, AutoModel
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
from batch_loaders.pair_alignment_batch_loader import MatrixRefinementBatchLoader, PairAlignmentBatchLoader
from models.mlm_model import MLMOntoBert
from models.siamese_model import SiameseOntoBert
from globals import Globals



class PairOntoBert(BaseModel):
    def __init__(self, from_pretrained="emilyalsentzer/Bio_ClinicalBERT") -> None:
        super().__init__()

        self.loader_type=PairAlignmentBatchLoader
        self.is_supervised = True

        self.model = AutoModelForSequenceClassification.from_pretrained(from_pretrained, num_labels=1, classifier_dropout=0.15, output_hidden_states=True)
        self.cosine_matrix = None
        self.string_alignments = None

        self.divergence_loss = InitialEmbeddingLoss()

        self.divergence_loss_alpha = 10

        self.classification_loss_alpha = 1

    def get_loss(self, batch, concept_pool_masks, concept_ids, ontos):

        outputs = self.model(**batch)
        bert_embeddings = outputs.hidden_states[-1]

        mlm_loss = self.classification_loss_alpha * outputs.loss
        # kl_div_loss = self.divergence_loss_alpha * self.divergence_loss(bert_embeddings, concept_pool_masks, concept_ids, ontos)


        return mlm_loss, torch.tensor([0]).to(Globals.device) #kl_div_loss

    def get_similarity_matrix(self, source, target, walk_config):
        src_keys = source.classes
        trg_keys = target.classes

        ontologies_map = {source.onto_name: source, target.onto_name: target}

        similarities = torch.zeros((len(src_keys), len(trg_keys)))

        refined_similarities = self.refine_similarity_matrix(similarities, len(trg_keys), ontologies_map, \
                                                            source.onto_name, target.onto_name, src_keys, trg_keys, walk_config=walk_config)


        return refined_similarities, src_keys, trg_keys

    def refine_similarity_matrix(self, similarities, topk, ontologies_map, src, trg, src_keys, trg_keys, walk_config=None):
        self.model = self.model.to(Globals.device)

        refined_similarities = torch.zeros_like(similarities)

        string_alignments_indices_src = None
        if self.string_alignments is not None:
            key_to_index_src = {key: i for i, key in enumerate(src_keys)}
            key_to_index_trg = {key: i for i, key in enumerate(trg_keys)}
            string_alignments_indices_src = [key_to_index_src[align.id1] for align in self.string_alignments]
            string_alignments_indices_trg = [key_to_index_trg[align.id2] for align in self.string_alignments]

        loader = MatrixRefinementBatchLoader(ontologies_map, similarities, topk, 
                                    src, trg, src_keys, trg_keys, exclude_indices=string_alignments_indices_src, walk_config=walk_config)

        

        with torch.no_grad():
            for batch_inputs, batch_indices in tqdm(loader, leave=True, disable=True):
                outputs = self.model(**batch_inputs)
                vals = outputs.logits

                for i, keys in enumerate(batch_indices):
                    refined_similarities[keys[0], keys[1]] = vals[i]


        refined_similarities[string_alignments_indices_src] = similarities[string_alignments_indices_src]
        

        self.model = self.model.cpu()

        return refined_similarities



    
