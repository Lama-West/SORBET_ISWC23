from .random_walk import tree_sample_walk
import torch
from globals import Globals


class TreePoolBatchLoader:
    def __init__(self, use_synonyms=False, n_branches=None, max_path_length=None,
                     parent_prob = 40, child_prob=40, object_prob=20) -> None:

        self.use_synonyms = use_synonyms

        self.n_branches = n_branches
        self.max_path_length = max_path_length
        self.parent_prob = parent_prob
        self.child_prob = child_prob
        self.object_prob = object_prob

    def _prepare_input(self, batch_nodes, walks, nodes_mask):
        

        nodes_mask = torch.cat(nodes_mask, dim=0).to(Globals.device)

        walks = [ " ".join(walk) for walk in walks]

        inputs = Globals.tokenizer(walks, return_tensors='pt',
                   max_length=512, truncation=True, padding='max_length').to(Globals.device)

        return inputs, batch_nodes, nodes_mask

    def _preprocess_walk(self, node, ontology):

        walk, walk_ids, branch_indexes = tree_sample_walk(ontology, 
                                                            return_branch_indexes=True,
                                                            first_node=node, 
                                                            n_branches = self.n_branches,
                                                            max_path_length = self.max_path_length,
                                                            parent_prob = self.parent_prob,
                                                            child_prob = self.child_prob,
                                                            object_prob = self.object_prob,
                                                            use_synonyms=self.use_synonyms)


        actual_node_lengths = [len(Globals.tokenizer(" ".join(ontology.get_id_label(id))).input_ids) - 2 for id in walk_ids]


        # The node mask is the mask that indicates 1 if the input token is related to the predicted token ex:
        # Person < Thing ; Person > Student - will have the mask: 1 0 0 0 1 0 0 
        root_mask = torch.zeros((1, 512), dtype=torch.bool)

        for branch_index in branch_indexes:
            actual_index = sum(actual_node_lengths[:branch_index])
            end_index = actual_index + actual_node_lengths[branch_index]
            
            root_mask[:,actual_index+1:end_index+1] = 1


        return walk, walk_ids, root_mask