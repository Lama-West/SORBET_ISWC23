from models.base_model import BaseModel
from losses.SentenceEmbeddingLoss import SentenceEmbeddingLoss
from batch_loaders.ontology_parsing.ontology import Ontology
from batch_loaders.MLM_batch_loader import EvaluationTreeWalkBatchLoader, RandomTreeWalkBatchLoader

import torch.nn.functional as F
import torch

class SBertModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()

        self.is_supervised = False
        self.embeddings_cache = {}
        self.loader_type=RandomTreeWalkBatchLoader


    def get_embeddings(self, ontology: Ontology):
        if ontology.onto_name in self.embeddings_cache:

            return self.embeddings_cache[ontology.onto_name]["embeddings"], self.embeddings_cache[ontology.onto_name]["embeddings_key"]


        embeddings_key = ontology.get_classes()
        embeddings = torch.cat([SentenceEmbeddingLoss.id_sentence_embedding[id].view(1,-1) for id in embeddings_key])

        self.embeddings_cache[ontology.onto_name] = {}
        self.embeddings_cache[ontology.onto_name]["embeddings"] = embeddings
        self.embeddings_cache[ontology.onto_name]["embeddings_key"] = embeddings_key
        self.embeddings_cache[ontology.onto_name]["embeddings_label"] = [f"{ontology.onto_name}#" + " ".join(ontology.get_id_label(id)) for id in embeddings_key]


        return embeddings, embeddings_key

    def get_similarity_matrix(self, source: Ontology, target: Ontology, walk_config=None):
        
        trg_embeddings, trg_keys = self.get_embeddings(target)
        src_embeddings, src_keys = self.get_embeddings(source)

        cosine_similarities = (F.normalize(src_embeddings,dim=1) @ F.normalize(trg_embeddings,dim=1).t()).cpu()

        return cosine_similarities, src_keys, trg_keys

    def get_embeddings_dict(self):
        return self.embeddings_cache

    def reset_embedding_dict(self):
        self.embeddings_cache = {}