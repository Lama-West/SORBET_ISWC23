import logging
from typing import List
import gc
import json
import os
import torch
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from AlignmentFormat import write_alignments
import pandas as pd

from batch_loaders.alignment import Alignment
from inference_pipeline import Results

logger = logging.getLogger("onto")

class SubTrackMetrics:
    def __init__(self, subtrack_result: Results, onto_map, source, target, reference_alignments, user_input_candidates_list, type="OA") -> None:
        self.similarities = subtrack_result.similarities
        self.src_keys = subtrack_result.src_keys
        self.trg_keys = subtrack_result.trg_keys
        self.pred_alignments = subtrack_result.pred_alignments
        self.alignment_per_threshold = subtrack_result.alignment_per_threshold
        self.user_input_candidates_list = user_input_candidates_list

        self.reference_alignments = reference_alignments
        self.source = source
        self.source_embeddings = subtrack_result.source_embeddings
        self.target_embeddings = subtrack_result.target_embeddings
        self.target = target 
        self.onto_map = onto_map
        self.type = type


    def compute_hits(self, ks):
        self.ks = ks
        self.src_key_to_idx = {key: i for i, key in enumerate(self.src_keys)}
        self.trg_key_to_idx = {key: i for i, key in enumerate(self.trg_keys)}

        self.hit_dict = {}

        hits = [0 for _ in range(len(ks))]

        for i, alignment in enumerate(self.reference_alignments):
            
            if self.user_input_candidates_list is None:
                ranks = torch.topk(self.similarities[self.src_key_to_idx[alignment.id1]], k=ks[-1], sorted=True, largest=True).indices.cpu().numpy()
            else:
                cands = [self.trg_key_to_idx[c] for c in self.user_input_candidates_list[i]]
                cands_sim = self.similarities[self.src_key_to_idx[alignment.id1], cands]
                ranks = torch.argsort(cands_sim, descending=True)
                ranks.apply_(lambda x: cands[x])
                ranks = ranks.view((-1,))

            self.hit_dict[alignment] = ranks

            for i, k in enumerate(ks):
                if ((alignment.id2 in self.trg_key_to_idx) and (alignment.id1 in self.src_key_to_idx) and \
                   (self.trg_key_to_idx[alignment.id2] in self.hit_dict[alignment][:k])):
                    hits[i] +=1

        self.hits = hits

    def get_metrics(self, threshold):
        TP = sum([pred in self.reference_alignments for pred in self.alignment_per_threshold[threshold]])
        FP = len(self.alignment_per_threshold[threshold]) - TP
        FN = len(self.reference_alignments) - TP

        return TP, FP, FN



    def _src_label(self, id):
        return " ".join(self.onto_map[self.source].get_id_label(id))

    def _trg_label(self, id):
        return " ".join(self.onto_map[self.target].get_id_label(id))

    def get_debug_dict(self, topk=10):

        debug_dict = {"false_negatives": {},
                      "false_positives": {},
                      "true_positives": {}}
        false_negatives = [a for a in self.reference_alignments if a not in self.pred_alignments]

        for a in false_negatives:
            local_dict = {}
            src_index = self.src_key_to_idx[a.id1]

            if a.id2 in self.trg_key_to_idx:
                self.similarities[src_index, self.trg_key_to_idx[a.id2]].item()
            else:
                local_dict["predicted_value"] = 0 

            for target_idx in self.hit_dict[a][:topk]:
                local_dict[self._src_label(a.id1) + "#" + self._trg_label(self.trg_keys[target_idx])] = self.similarities[src_index, target_idx].item()

            debug_dict["false_negatives"][self._src_label(a.id1) + "#" + self._trg_label(a.id2)] = local_dict

        false_positives = [a for a in self.pred_alignments if a not in self.reference_alignments]

        for a in false_positives:
            debug_dict["false_positives"][self._src_label(a.id1) + "#" + self._trg_label(a.id2)] = self.similarities[self.src_key_to_idx[a.id1], self.trg_key_to_idx[a.id2]].item()
    
        true_positives = [a for a in self.pred_alignments if a in self.reference_alignments]

        for a in true_positives:
            debug_dict["true_positives"][self._src_label(a.id1) + "#" + self._trg_label(a.id2)] = self.similarities[self.src_key_to_idx[a.id1], self.trg_key_to_idx[a.id2]].item()
    
        

        return debug_dict

    def write_embeddings_image(self, filepath):
        folder = os.path.join(filepath, "ontologies")

        if not os.path.exists(folder):
            os.makedirs(folder)

        full_path_source = os.path.join(folder, self.source + ".png") 
        self.draw_embedding_structure(self.onto_map[self.source], self.source_embeddings, self.src_keys, self.src_key_to_idx, full_path_source)

        full_path_target = os.path.join(folder, self.target + ".png") 
        self.draw_embedding_structure(self.onto_map[self.target], self.target_embeddings, self.trg_keys, self.trg_key_to_idx, full_path_target)


    def draw_embedding_structure(self, onto, embeddings, keys, key_to_index, path):
        pca = PCA(n_components = 2)
        projected_embeddings = pca.fit_transform(embeddings.cpu().numpy())

        self.draw_projection_structure(projected_embeddings, onto, keys, key_to_index)

            
        plt.axis('off')
        plt.savefig(path, dpi=300, bbox_inches='tight')

        plt.clf()

    def draw_both_embedding_structure(self, source_emb, target_emb, folder, filename):
        
        self.draw_projection_structure(source_emb, self.onto_map[self.source], self.src_keys, self.src_key_to_idx)
        self.draw_projection_structure(target_emb, self.onto_map[self.target], self.trg_keys, self.trg_key_to_idx, color="red")

        plt.axis('off')
        plt.savefig(os.path.join(folder, filename), dpi=600, bbox_inches='tight')

        plt.clf()


    def draw_projection_structure(self, projected_embeddings, onto, keys, key_to_index, color="green"):
        plt.scatter(projected_embeddings[:,0].reshape((-1)),projected_embeddings[:,1].reshape((-1)), s = [2 for n in range(len(projected_embeddings))], c=color)

        for i in range(len(projected_embeddings)):
            plt.annotate(keys[i], (projected_embeddings[i,0], projected_embeddings[i,1]), fontsize=3)

        for i in range(len(projected_embeddings)):
            parent = keys[i]
            for child in onto.childs_map[parent]:
                point1 = projected_embeddings[key_to_index[parent]].reshape((-1))
                point2 = projected_embeddings[key_to_index[child]].reshape((-1))

                plt.plot([point1[0], point2[0]], [point1[1], point2[1]], color="black", linewidth=0.5)
   

    def write_alignments_image(self, folder):
        
        folder = os.path.join(folder, "alignments")
        if not os.path.exists(folder):
            os.makedirs(folder)

        subtrack = self.source + "_" + self.target
        folder = os.path.join(folder, subtrack)

        if not os.path.exists(folder):
            os.makedirs(folder)

        filename_reference = "reference.png"
        filename_predictions = "predictions.png"
        filename_strucutres = "structures.png"


        pca = PCA(n_components = 2)
        embeddings = pca.fit_transform(np.concatenate((self.source_embeddings.cpu().numpy(), self.target_embeddings.cpu().numpy()), axis=0))

        source_emb_np = embeddings[:self.source_embeddings.shape[0],:]
        target_emb_np = embeddings[self.source_embeddings.shape[0]:,:]

        self.draw_alignments(source_emb_np, target_emb_np, folder, filename_reference, self.reference_alignments)
        self.draw_alignments(source_emb_np, target_emb_np, folder, filename_predictions, self.pred_alignments)

        self.draw_both_embedding_structure(source_emb_np, target_emb_np, folder, filename_strucutres)




    def draw_alignments(self, source_emb_np, target_emb_np, folder, filename, alignments):

        plt.scatter(source_emb_np[:,0].reshape((-1)),source_emb_np[:,1].reshape((-1)), s = [2 for n in range(len(source_emb_np))], c="green")
        plt.scatter(target_emb_np[:,0].reshape((-1)),target_emb_np[:,1].reshape((-1)), s = [2 for n in range(len(target_emb_np))], c="red", marker="^")


        for a in alignments:
            self.draw_alignment(plt, a, source_emb_np, target_emb_np)

        # for i in range(len(source_emb_np)):
        #     plt.annotate(self.src_keys[i], (source_emb_np[i,0], source_emb_np[i,1]), fontsize=2)

        # for i in range(len(target_emb_np)):
        #     plt.annotate(self.trg_keys[i], (target_emb_np[i,0], target_emb_np[i,1]), fontsize=2)

        plt.axis('off')
        plt.savefig(os.path.join(folder, filename), dpi=300, bbox_inches='tight')

        plt.clf()



    
    def draw_alignment(self, plt, alignment, src_embeddings, trg_embeddings):
        point1 = src_embeddings[self.src_key_to_idx[alignment.id1]].reshape((-1))
        point2 = trg_embeddings[self.trg_key_to_idx[alignment.id2]].reshape((-1))

        # plt.annotate(alignment.id2, (point2[0], point2[1]), fontsize=3)

        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], color="black", linewidth=0.5)
        


    def write_alignments_tsv(self, path):
        a = [(self.onto_map[self.source].iris[alignment.id1], self.onto_map[self.target].iris[alignment.id2], 1.0)  for alignment in self.pred_alignments]

        tsv = pd.DataFrame(a, columns=["SrcEntity", "TgtEntity", "Score"])

        tsv.to_csv(path, sep="\t")

    def write_ranking(self, path):
        trg_to_iri = lambda concept: self.onto_map[self.target].iris[self.trg_keys[concept]]

        rows = []
        for a in self.reference_alignments:
            tgt_candidates = list(map(lambda c: trg_to_iri(c), self.hit_dict[a]))
            rows.append((self.onto_map[self.source].iris[a.id1], self.onto_map[self.target].iris[a.id2], tuple(tgt_candidates)))

        ranks = pd.DataFrame(rows, columns=["SrcEntity", "TgtEntity", "TgtCandidates"])
        ranks.to_csv(path, sep="\t", index=False)

    def write_alignments_rdf(self, path):
        a = [(self.onto_map[self.source].iris[alignment.id1], self.onto_map[self.target].iris[alignment.id2], "=", 1.0)  for alignment in self.pred_alignments]

        write_alignments(path, a, self.source, self.target)


class Metrics:
    def __init__(self, write_tsv=False, write_rdf=False, write_ranking=False, hits=[5,10,50], debug_files_path="./debug", \
                 results_files_path="./result_alignments", track_name="track_name") -> None:
        self.hits = hits
        self.keys_map = {}
        self.exclude_alignments = []
        self.debug_files_path = debug_files_path
        self.write_tsv = write_tsv
        self.write_rdf = write_rdf
        self.write_ranking = write_ranking
        self.track_name = track_name
        self.results_files_path = results_files_path



    def run(self, inference_pipeline, ontologies_map, alignment_per_dataset):
        
        self.subtrack_metrics = []

        equal_alignments_len = 0

        for alignment_dataset in alignment_per_dataset:
            source, target = (alignment_dataset.source_ontology_metadata.id, alignment_dataset.target_ontology_metadata.id)

            test_alignments, train_alignments = alignment_dataset.test_alignments, alignment_dataset.train_alignments


            subtrack_result: Results = inference_pipeline.infer(source, target, ontologies_map, alignment_dataset.user_input_candidates)


            subtrack_result.exclude_alignments(train_alignments)


            equal_alignments_len += len(test_alignments)

            subtrack_metric = SubTrackMetrics(subtrack_result, ontologies_map, source, target, test_alignments, alignment_dataset.user_input_candidates_list,
                                               type=alignment_dataset.type)
            subtrack_metric.compute_hits(self.hits)


            self.subtrack_metrics.append(subtrack_metric)
        

        final_result = []
        for t in inference_pipeline.thresholds:
            try:
                n_TP = 0
                n_FP = 0
                n_FN = 0

                for s in self.subtrack_metrics:
                    TP, FP, FN = s.get_metrics(t)
                    n_TP += TP
                    n_FP += FP
                    n_FN += FN

                precision = n_TP/(n_TP + n_FP)
                recall    = n_TP/(n_TP + n_FN)
                f1_score  =  2*precision*recall/(precision+recall)

                final_result.append([t, round(precision,4), round(recall,4), round(f1_score,4)])
            except:
                final_result.append([t, np.nan, np.nan, np.nan])

        final_result_string = "\n" + pd.DataFrame(final_result, columns=["Threshold", "Precision", "Recall", "F1"]).T.to_string()

        logger.info("#### Metrics ####")

        logger.info(final_result_string)


        for i, hit in enumerate(self.hits):
            logger.info(f"Hits@{hit}    : {sum([result.hits[i] for result in self.subtrack_metrics])/equal_alignments_len}")

        return self.subtrack_metrics


    def write_results(self, epoch):
        if not os.path.exists(self.results_files_path):
            os.makedirs(self.results_files_path)

        folder = os.path.join(self.results_files_path, self.track_name)

        if not os.path.exists(folder):
            os.makedirs(folder)

        for r in self.subtrack_metrics:

            filename = f"{r.source}2{r.target}_{epoch}"
            filename_rank = f"{r.source}2{r.target}_ranking_{epoch}"

            file = os.path.join(folder, filename)
            file_rank = os.path.join(folder, filename_rank)
            
            if self.write_rdf:
                r.write_alignments_rdf(file + ".rdf")

            if self.write_tsv:
                r.write_alignments_tsv(file + ".tsv")

            if self.write_ranking:
                r.write_ranking(file_rank + ".tsv")


    def write_debug(self, epoch):

        debug_dict = {}
        for result in self.subtrack_metrics:
            debug_dict[result.source + " : " + result.target] = result.get_debug_dict(topk=10)

        folder = os.path.join(self.debug_files_path, self.track_name)

        if not os.path.exists(folder):
            os.makedirs(folder)

        filename = f"_{epoch}.json"

        with open(os.path.join(folder, filename), "w") as file:
            json.dump(debug_dict, file, indent=4)

        if self.track_name == "conference":
            for r in self.subtrack_metrics:
                filepath = os.path.join(folder, "embeddings_" + str(epoch))
                
                if not os.path.exists(filepath):
                    os.mkdir(filepath)

                r.write_embeddings_image(filepath)
                r.write_alignments_image(filepath)



