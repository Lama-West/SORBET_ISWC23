from scipy.optimize import linear_sum_assignment

class MaxWeightBipartiteExtractor():
    def __init__(self, ) -> None:
        pass
    
    def run(self, similarities, src_keys, trg_keys, threshold=0.8, string_alignments=None):
        chosen_indices = linear_sum_assignment(1. - similarities.cpu())

        return [(src_keys[i], trg_keys[j]) for i,j in zip(chosen_indices[0], chosen_indices[1]) if similarities[i,j] >= self.threshold]
