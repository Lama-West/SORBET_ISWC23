import torch
from track import Track
from losses.SentenceEmbeddingLoss import SentenceEmbeddingLoss
from typing import List
import torch.nn.functional as F
from batch_loaders.alignment import Alignment
import logging 

logger = logging.getLogger("onto")


class IsolatedClassMatcher():
    def __init__(self, threshold=0.90) -> None:
        self.threshold = threshold

    def is_isolated(self, ontology, id):
        return (len(ontology.get_childs(id)) + len(ontology.get_parents(id))) == 0

    def find_closest_concept(self, ontologies_map, onto_name, id):
            temp_id2 = None
            temp_onto2 = None
            temp_sim = 0

            id_embedding = SentenceEmbeddingLoss.id_sentence_embedding[id]
            for onto_name_2, onto_2 in ontologies_map.items():
                if onto_name_2 != onto_name:
                    trg_embeddings = SentenceEmbeddingLoss.onto_embeddings[onto_name_2]["embeddings"]
                    similarities = F.normalize(id_embedding.view(1,-1),dim=1) @ F.normalize(trg_embeddings,dim=1).t()

                    id2_index = similarities[0].argmax().item()

                    if abs(similarities[0].max().item() - 1.0) < 1e-5:
                        # If there is a string match, and the other concept is isolated => ignore
                        if not self.is_isolated(onto_2, onto_2.classes[id2_index]):
                            return onto_2.classes[id2_index], onto_name_2, 1.0
                        else:
                            similarities[0, id2_index] = 0.0
                            id2_index = similarities[0].argmax().item()
                            

                    if (similarities[0, id2_index] > temp_sim):
                        temp_sim = similarities[0, id2_index].item()
                        temp_id2 = onto_2.classes[id2_index]
                        temp_onto2 = onto_name_2

            return temp_id2, temp_onto2, temp_sim


    def run_tracks(self, tracks: List[Track], current_alignments):

        current_alignments = set(current_alignments) 
        isc_alignments = set([])
        
        for track in tracks:
            for onto_name, onto in track.ontologies_map.items():
                for id in onto.classes:
                    if self.is_isolated(onto, id):

                        id2, onto_name_2, similarity = self.find_closest_concept(track.ontologies_map, onto_name, id) 

                        alignment = Alignment(onto_name, onto_name_2, id, id2, label=similarity)
                        if (similarity > self.threshold) and (alignment not in current_alignments):
                            isc_alignments.update([alignment])


        return list(isc_alignments)
                            




                        



