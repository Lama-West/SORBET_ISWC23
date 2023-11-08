from models.tf_idf_similarity import SubTokenSimilarity
from matchers.greedy_matcher import GreedyMatcher
from batch_loaders.alignment import Alignment
from matchers.string_matcher import StringMatcher

import torch
import gc
from globals import Globals

import logging
logger = logging.getLogger("onto")



class Results:
    def __init__(self, pred_alignments, similarities, src_keys, trg_keys, source_embeddings, target_embeddings, alignment_per_threshold=None) -> None:
        self.pred_alignments = pred_alignments  
        self.similarities = similarities  
        self.src_keys = src_keys  
        self.trg_keys = trg_keys
        self.source_embeddings = source_embeddings
        self.target_embeddings = target_embeddings

        self.alignment_per_threshold = alignment_per_threshold
    
    def exclude_alignments(self, other_alignments):
        self.pred_alignments = [pred for pred in self.pred_alignments if pred not in other_alignments]

        for t, alignments in self.alignment_per_threshold.items():
            self.alignment_per_threshold[t] = [pred for pred in alignments if pred not in other_alignments]



    


class InferencePipeline:
    def __init__(self,  string_matching_optimization = True,
                        model = None, candidate_selector = None, matcher = None, walk_config = None, thresholds = None) -> None:
                        
        self.string_matching_optimization = string_matching_optimization
        self.do_candidate_selection = candidate_selector is not None
        
        self.candidate_selector = candidate_selector
        self.matcher = matcher if matcher is not None else GreedyMatcher()
        self.model = model

        if (thresholds is None) or (len(thresholds) == 0):
            raise ValueError("thresholds array must not be None or be empty")

        self.thresholds = thresholds

        self.string_matcher = StringMatcher()

        self.candidates = {}
        self.keys_map = {}

        self.walk_config = walk_config


    def free_candidate_selector(self):
        if (self.candidate_selector is not None):
            del self.candidate_selector
            gc.collect()
            self.candidate_selector = None

    def update_similarities(self, similarities, pred_alignments, src_keys, trg_keys ):
        src_key_to_idx = {key: i for i, key in enumerate(src_keys)}
        trg_key_to_idx = {key: i for i, key in enumerate(trg_keys)}

        for alignment in pred_alignments:
            similarities[src_key_to_idx[alignment.id1], trg_key_to_idx[alignment.id2]] = 1.


    def infer(self, source, target, ontologies_map, user_input_candidates = None):
        if self.model is None:
            raise ValueError("Cannot infer without a model")

        if self.string_matching_optimization:
            string_alignments = self.string_matcher.run(ontologies_map[source], ontologies_map[target])
        else:
            string_alignments = []

        if self.do_candidate_selection:
            if (source, target) not in self.candidates:
                # Runs on iteration 0
                similarities, src_keys, trg_keys = self.candidate_selector.get_similarity_matrix(ontologies_map[source], ontologies_map[target])
                self.keys_map[(source, target)] = (src_keys, trg_keys)
                similarities = similarities.cpu()
            else:
                # Runs on iteration 1..
                src_keys, trg_keys = self.keys_map[(source, target)]
                self.model.string_alignments = string_alignments
                similarities = self.model.refine_similarity_matrix(self.candidates[(source, target)].to(Globals.device), 5, ontologies_map, \
                                                                   source, target, src_keys, trg_keys, walk_config=self.walk_config)

                
        else:
            self.model.string_alignments = string_alignments
            similarities, src_keys, trg_keys = self.model.get_similarity_matrix(ontologies_map[source], ontologies_map[target], \
                                                                                walk_config=self.walk_config, user_input_candidates=user_input_candidates)
            similarities = similarities.cpu()


        alignment_per_threshold = {}
        for t in self.thresholds:

            raw_pred_alignments = self.matcher.run(similarities, src_keys, trg_keys, threshold=t, string_alignments=string_alignments)

            pred_alignments = [Alignment(source, target, pred[0], pred[1]) for pred in raw_pred_alignments] 


            if self.string_matching_optimization:
                pred_alignments.extend(string_alignments)
                pred_alignments = list(set(pred_alignments))

            alignment_per_threshold[t] = pred_alignments

        # If there is candidate selection and it is the first run
        if ((source, target) not in self.candidates):
            self.candidates[(source, target)] = similarities

        source_embeddings, _ = self.model.get_embeddings(ontologies_map[source])
        target_embeddings, _ = self.model.get_embeddings(ontologies_map[target])

        return Results(pred_alignments, similarities, src_keys, trg_keys, source_embeddings, target_embeddings, alignment_per_threshold=alignment_per_threshold)
        