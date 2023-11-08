from typing import List
import torch
from torch import nn, Tensor
import random

from transformers import AutoTokenizer, AutoModel

from batch_loaders.ontology_parsing.ontology import Ontology
from globals import Globals

import json

with open("config.json", 'r') as f:
    config = json.load(f)

class SentenceEmbeddingLoss(nn.Module):

    id_sentence_embedding = None
    onto_embeddings = None

    def __init__(self, loss_fct = nn.MSELoss()):
        super(SentenceEmbeddingLoss, self).__init__()
        self.loss_fct = loss_fct




    def forward(self, output, concept_pool_masks, concept_ids):
        if SentenceEmbeddingLoss.id_sentence_embedding is None:
            raise ValueError("Error when calculating loss: must call _precompute_sentence_embeddings first")

        loss = 0.0
        for i, ids in enumerate(concept_ids):
            for j, id in enumerate(ids):
                
                if id not in SentenceEmbeddingLoss.id_sentence_embedding:
                    continue

                loss += self.loss_fct(torch.mean(output[i][concept_pool_masks[i][j].view(-1)], dim=0).view(1,-1), SentenceEmbeddingLoss.id_sentence_embedding[id] )
        
        loss = loss / sum([len(concept) for concept in concept_ids])
        
        return loss

    def _precompute_sentence_embeddings(ontologies: List[Ontology]):
        sbert_tokenizer = AutoTokenizer.from_pretrained(config["General"]["model"])
        sbert_model = AutoModel.from_pretrained(config["General"]["model"], output_hidden_states=True)
        
        def _sbert_mean_pooling(model_output):
                token_embeddings = model_output[0] #First element of model_output contains all token embeddings
                token_embeddings = token_embeddings[:,1:-1] # Remove CLS and SEP tokens
                return torch.mean(token_embeddings, 1)


        concept_id_map = {}
        onto_embeddings = {}
        with torch.no_grad():
            for onto in ontologies:
                for id in onto.id_to_label_dict:
                    if id not in concept_id_map:

                        encoded_input = sbert_tokenizer(" ".join(onto.get_id_label(id)), padding=True, truncation=True, return_tensors='pt')

                        model_output = sbert_model(**encoded_input)

                        # Perform pooling. In this case, max pooling.
                        sentence_embeddings = _sbert_mean_pooling(model_output)
                        concept_id_map[id] = sentence_embeddings.to(Globals.device)


                embeddings = None
                for id in onto.classes:
                    if embeddings == None:
                        embeddings = concept_id_map[id].view(1,-1)
                    else:
                        embeddings = torch.cat((embeddings, concept_id_map[id].view(1,-1)))
                onto_embeddings[onto.onto_name] = {}
                onto_embeddings[onto.onto_name]["embeddings"] = embeddings
                onto_embeddings[onto.onto_name]["embeddings_label"] = [f"{onto.onto_name}#" + " ".join(onto.get_id_label(id)) for id in onto.classes]
                

        SentenceEmbeddingLoss.id_sentence_embedding = concept_id_map
        SentenceEmbeddingLoss.onto_embeddings = onto_embeddings