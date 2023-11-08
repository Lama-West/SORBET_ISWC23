import torch

class GreedyMatcher():
    def __init__(self) -> None:
        pass

    def run(self, similarities, src_keys, trg_keys, threshold=0.80, string_alignments=None):
        src_indexes, trg_indexes = torch.where(similarities > threshold, similarities.double(), 0.).nonzero(as_tuple=True)

        preds = [(src_keys[src_indexes[i]], trg_keys[trg_indexes[i]], similarities[src_indexes[i], trg_indexes[i]]) 
                                for i in range(len(src_indexes))]

        preds.sort(key=lambda x: x[2], reverse=True)
        

        aligned_terms_source = set([a.id1 for a in string_alignments])        
        aligned_terms_target = set([a.id2 for a in string_alignments])        
        alignments = []
        for pred in preds:
            if (pred[0] not in aligned_terms_source) and (pred[1] not in aligned_terms_target):
                aligned_terms_source.add(pred[0])
                aligned_terms_target.add(pred[1])

                alignments.append((pred[0], pred[1]))


        return alignments