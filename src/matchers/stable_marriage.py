import string
import numpy as np
import torch

class StableMarriage():
    def __init__(self) -> None:
        pass

    def run(self, similarities, src_keys, trg_keys, string_alignments = None):
        
        if string_alignments is not None:
            key_to_index_src = {key: i for i, key in enumerate(src_keys)}
            key_to_index_trg = {key: i for i, key in enumerate(trg_keys)}
            string_alignments_indices_src = set([key_to_index_src[align.id1] for align in string_alignments])
            string_alignments_indices_trg = set([key_to_index_trg[align.id2] for align in string_alignments])


        similarities_copy = torch.clone(similarities)

        threshold = 0
        alignments = []
        has_stagnated = False
        while (len(alignments) != min(similarities_copy.shape)) and not has_stagnated:
            max_concept_source = similarities_copy.argmax(axis=1)
            max_concept_target = similarities_copy.argmax(axis=0)

            correspondances = []
            # Find correspondances
            if len(max_concept_source) < len(max_concept_target):
                for i in range(len(max_concept_source)):
                    if (string_alignments is not None) and (i in string_alignments_indices_src):
                        continue

                    # If both index point at each other, there is a correspondance
                    if max_concept_target[max_concept_source[i]] == i:
                        correspondances.append((i, max_concept_source[i].item(), similarities_copy[i, max_concept_source[i]].item()))
            else:
                for j in range(len(max_concept_target)):
                    if (string_alignments is not None) and (j in string_alignments_indices_trg):
                        continue
                    # If both index point at each other, there is a correspondance
                    if max_concept_source[max_concept_target[j]] == j:
                        correspondances.append((max_concept_target[j].item(), j, similarities_copy[max_concept_target[j], j].item()))


            if len(alignments) == 0:
                values = [c[2] for c in correspondances]
                threshold = np.mean(values) - np.std(values)

            has_stagnated = True
            for i, j, value in correspondances:
                if value >= threshold:
                    alignments.append((src_keys[i], trg_keys[j]))
                    has_stagnated = False
                similarities_copy[i,:] = 0.
                similarities_copy[:,j] = 0.


        return alignments
