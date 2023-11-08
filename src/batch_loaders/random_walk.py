import random, copy, os
from numpy import dtype
import torch
from math import ceil
from globals import Globals
from enum import Enum
import logging


logger = logging.getLogger("onto")


def sample_walk(onto, first_node=None, path_length=None, exclude_nodes=None, parent_prob = 40, child_prob=40, object_prob=20):
    """
    Makes a walk in the ontology with paths types in probabilities
    """

    if not path_length:
        path_length = random.randint(2,6)

    if not first_node:
        first_node = random.choice(list(set(onto.classes) - {"Thing"}))
        while (len(onto.get_parents(first_node)) + 
               len(onto.get_childs(first_node)) + 
               len(onto.get_object_properties(first_node))) == 0:
            first_node = random.choice(list(set(onto.classes) - {"Thing"}))

    walk = copy.deepcopy(onto.get_id_label(first_node))
    walk_ids = [first_node]
    current_node = first_node

    visited_nodes = exclude_nodes if exclude_nodes else set(walk_ids)

    for i in range(path_length):
        
        possible_parent_next_nodes       = list(set(onto.get_parents(current_node)) - visited_nodes)
        possible_child_next_nodes        = list(set(onto.get_childs(current_node)) - visited_nodes)
        possible_object_prop_next_nodes  = list(filter(lambda x: x[0] not in visited_nodes, onto.get_object_properties(current_node)))

        true_parent_prob      = parent_prob if len(possible_parent_next_nodes) > 0 else 0
        true_child_prob       = child_prob if len(possible_child_next_nodes) > 0 else 0
        true_object_prop_prob = object_prob if len(possible_object_prop_next_nodes) > 0 else 0
        
        if (true_parent_prob + true_child_prob + true_object_prop_prob) == 0:
            break

        selected_type = random.choices(["parent", "child", "object_prop"], 
                                       weights=(true_parent_prob, true_child_prob, true_object_prop_prob))[0]

        if selected_type == "parent":
            relation = "<"
            next_node = random.choice(possible_parent_next_nodes)

        elif selected_type == "child":
            relation = ">"
            next_node = random.choice(possible_child_next_nodes)

        else:
            id   = random.choice(possible_object_prop_next_nodes)
            relation = id[0]
            next_node = id[1]

        walk_ids += [relation, next_node]

        walk += onto.get_id_label(relation) + onto.get_id_label(next_node)


        visited_nodes.update(set(walk_ids))

        if relation != ":":
            current_node = next_node

        if current_node == "Thing":
            break

    return walk, walk_ids


class WalkStrategy(Enum):
    ANY = 1
    ONTOLOGICAL_RELATIONS = 2
    LINEAGE_PATH_PLUS_CHILDS = 3
    LINEAGE_PATH = 4
    
class RandomWalkConfig():
     def __init__(self, n_branches=None, max_path_length=None, strategy: WalkStrategy = WalkStrategy.ANY, use_synonyms=False):
        
        if n_branches is None:
            n_branches = random.randint(0,3)

        self.n_branches = n_branches
        self.strategy = strategy
        self.max_path_length = max_path_length
        self.use_synonyms = use_synonyms



class RandomWalk():
    def __init__(self, onto, first_node=None, walk_config: RandomWalkConfig=None):
        """
        Makes a walk in the ontology with paths types in probabilities
        """



        if walk_config is None:
            walk_config = RandomWalkConfig()

        if first_node is None:
            first_node = random.choice(list(set(onto.classes) - {"Thing"}))
            while (len(onto.get_parents(first_node)) + 
                len(onto.get_childs(first_node)) + 
                len(onto.get_object_properties(first_node))) == 0:
                first_node = random.choice(list(set(onto.classes) - {"Thing"}))

        if walk_config.n_branches == 0:
            walk_ids = [first_node]
            self.branches_index = [0]
        else:
            walk_ids = []
            self.branches_index = []


        visited_nodes = set({})
        one_khop_neighbours = set()

        for i in range(walk_config.n_branches):
            
            # Check if adding a branch is possible
            
            unvisited_parent  = len(list(set(onto.get_parents(first_node)) - visited_nodes))
            unvisited_child   = len(list(set(onto.get_childs(first_node)) - visited_nodes))
            unvisited_object  = len(list(filter(lambda x: x[0] not in visited_nodes, onto.get_object_properties(first_node))))


            if (walk_config.strategy == WalkStrategy.ANY) and ((unvisited_parent + unvisited_child + unvisited_object) > 0):
                parent_prob = 40
                child_prob = 40
                object_prop_prob = 20
            elif (walk_config.strategy == WalkStrategy.ONTOLOGICAL_RELATIONS) and ((unvisited_parent + unvisited_child) > 0):
                object_prop_prob = 0
                parent_prob = 50
                child_prob = 50
            elif (walk_config.strategy == WalkStrategy.LINEAGE_PATH) and ((unvisited_parent + unvisited_child) > 0):
                object_prop_prob = 0
                parent_prob = 100
                child_prob = 0
            elif (walk_config.strategy == WalkStrategy.LINEAGE_PATH_PLUS_CHILDS) and ((unvisited_parent + unvisited_child) > 0):
                object_prop_prob = 0
                if unvisited_parent > 0:
                    parent_prob = 100
                    child_prob = 0
                else:
                    parent_prob = 50
                    child_prob = 100
                
            else:
                # If use case happens, it means no branch can be made so, force algo to end there
                if i==0:
                    walk_ids = [first_node]
                    self.branches_index = [0]
                break
            
            walk_ids.extend(["[SEP]"])

            # Add a branch
            self.branches_index.append(len(walk_ids))

            _, sub_walk_ids = sample_walk(onto=onto,  first_node=first_node, exclude_nodes=visited_nodes, path_length=walk_config.max_path_length, 
                            parent_prob=parent_prob, child_prob=child_prob, object_prob=object_prop_prob)

            visited_nodes.update(set(sub_walk_ids))
            walk_ids.extend(sub_walk_ids)

        if walk_config.use_synonyms:
            # Randomly replace concept by its synonyms

            # Replace root node
            new_root = onto.get_random_synonym(first_node)
            for i in self.branches_index:
                walk_ids[i] = new_root

            # Replace all other concept
            for i, id in enumerate(walk_ids):
                if id != first_node:
                    if id in onto.synonyms:
                        walk_ids[i] = onto.get_random_synonym(id)

        walk = [copy.deepcopy(onto.get_id_label(id)) for id in walk_ids]
        walk = [word for label in walk for word in label]


        self.onto = onto
        self.walk = walk
        self.walk_ids = walk_ids
        self.sentence = " ".join(self.walk)


    def _get_mask_at_index(self, i, masked_array=None):
        if masked_array is None:
            masked_array = torch.zeros((1,512), dtype=torch.bool)

        actual_index = sum(self.actual_node_lengths[:i])
        end_index = actual_index + self.actual_node_lengths[i]

        masked_array[:,actual_index+1:end_index+1] = 1

        return masked_array



    def _get_pool_masks(self):

        possible_mask_ids = list(filter(lambda index: index%2==0, range(len(self.walk_ids))))

        concept_pool_masks = []
        for id in possible_mask_ids:
            masked_array = self._get_mask_at_index(id)

            concept_pool_masks.append(masked_array)

        self.concept_pool_masks = concept_pool_masks
        self.concepts = [self.walk_ids[i] for i in possible_mask_ids]



    def _get_root_pool_mask(self):
        self.root_mask = torch.zeros((1, 512), dtype=torch.bool)

        for branch_index in self.branches_index:
            self._get_mask_at_index(branch_index, self.root_mask)

    def build_parent_child_pairs_mask(self):
        self.subclass_pairs = []
        for i, c in enumerate(self.walk_ids):

            if c == "<":
                parent_mask = self._get_mask_at_index(i+1)
                child_mask = self._get_mask_at_index(i-1)
            elif c == ">":
                parent_mask = self._get_mask_at_index(i-1)
                child_mask = self._get_mask_at_index(i+1)
            else:
                continue

            relation = self._get_mask_at_index(i)
            self.subclass_pairs.append((parent_mask, relation, child_mask))

        return self


    def build_pooling_mask(self):
        """
        Compute pooling masks for the root and other concepts
        param: Globals.tokenizer 
        """

        self.actual_node_lengths = [len(Globals.tokenizer(" ".join(self.onto.get_id_label(id))).input_ids) - 2 for id in self.walk_ids] 

        self._get_pool_masks()  
        self._get_root_pool_mask()

        self.ontos = [self.onto.get_name() for i in range(len(self.concepts))]

        return self


    def build_mlm_mask(self, apply_masks=True):
        """
        Function to compute a random MLM mask on the random walk, must have called build_pooling_mask before
        """
        self.mlm_mask = torch.zeros((1,512), dtype=torch.bool)
        
        if (len(self.walk) < 2) or not apply_masks:
            return self
            
        # TODO which tokens to mask or not?
        unwanted_masked_token = ["[SEP]"]

        possible_mask_ids = list(filter(lambda index: self.walk_ids[index] not in unwanted_masked_token, range(len(self.walk_ids))))
        masked_id_index = random.sample(possible_mask_ids,  k= ceil(len(possible_mask_ids) / 4) )

        # Get the length of input tokens for every node
        actual_node_lengths = [len(Globals.tokenizer(" ".join(self.onto.get_id_label(id))).input_ids) - 2 for id in self.walk_ids]

        for masked_index in masked_id_index:
            # Index of the masked concept
            actual_index = sum(actual_node_lengths[:masked_index])
            end_index = actual_index + actual_node_lengths[masked_index]

            masked_tokens_indices = list(range(actual_index+1,end_index+1))
            # If there are too much tokens for one node, mask
            if len(masked_tokens_indices) > 5:
                masked_tokens_indices = random.sample(masked_tokens_indices, k=int(5+len(masked_tokens_indices)*0.15))
            else:
                masked_tokens_indices = random.sample(masked_tokens_indices, k=random.randint(1, len(masked_tokens_indices)))
                

            self.mlm_mask[:,masked_tokens_indices] = 1

        return self
        



class PairRandomWalk():
    def __init__(self, src_walk: RandomWalk, trg_walk: RandomWalk) -> None:
        self.src_walk = src_walk
        self.trg_walk = trg_walk

        self.sentence = self.src_walk.sentence + " [SEP] "  + self.trg_walk.sentence

    def build_pooling_mask(self):
        self.src_walk = self.src_walk.build_pooling_mask()

        self.trg_walk = self.trg_walk.build_pooling_mask()

        self.concepts = self.src_walk.concepts + self.trg_walk.concepts


        # Shifting masks to the right
        right_shift = sum(self.src_walk.actual_node_lengths) + 1

        self.trg_walk.concept_pool_masks = [torch.roll(mask, right_shift) for mask in self.trg_walk.concept_pool_masks]

        self.concept_pool_masks = self.src_walk.concept_pool_masks + self.trg_walk.concept_pool_masks

        self.ontos = self.src_walk.ontos + self.trg_walk.ontos
        
        return self

    def build_mlm_mask(self):

        right_shift = sum(self.src_walk.actual_node_lengths) + 1

        
        if (len(self.src_walk.walk_ids) != 1) and (random.random() < 0.15):
            src_mask = self.src_walk.root_mask
        else:
            src_mask = torch.zeros_like(self.src_walk.root_mask, dtype=torch.bool)

        if (len(self.trg_walk.walk_ids) != 1) and random.random() < 0.15:
            trg_mask = torch.roll(self.trg_walk.root_mask, right_shift)
        else:
            trg_mask = torch.zeros_like(self.trg_walk.root_mask, dtype=torch.bool)

        self.mlm_mask = src_mask + trg_mask

        return self
        


