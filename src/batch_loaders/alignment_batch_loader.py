import random, copy, os
from typing import Dict, List, Tuple
import numpy as np
import torch
from math import ceil
from copy import deepcopy
import traceback

from .alignment import Alignment, AlignmentDataset
from .ontology_parsing.ontology_config import OntoConfig
from .ontology_parsing.ontology import Ontology
from models.tf_idf_similarity import SubTokenSimilarity
from .random_walk import RandomWalk, RandomWalkConfig
from globals import Globals

# device = "cpu"


class AlignmentBatchLoader:
    def __init__(self, ontologies_map, batch_size=64, n_alignments_per_batch=32, shuffle=True, apply_masks=True ,walk_config:RandomWalkConfig = None, **kwargs):

        self.batch_size = batch_size
        self.n_alignments_per_batch = n_alignments_per_batch

        self.ontologies_map: Dict[str, Ontology] = ontologies_map
        self.walk_config = walk_config
        self.shuffle = shuffle
        self.apply_masks = apply_masks

    def get_batch_alignments(self, idx):
        pass

    def build_inputs(self, batch_alignments):
        
        src_random_walks = [RandomWalk(self.ontologies_map[a.source_onto], first_node=a.id1, walk_config=self.walk_config)\
                                .build_pooling_mask().build_mlm_mask(self.apply_masks).build_parent_child_pairs_mask() for a in batch_alignments]
        trg_random_walks = [RandomWalk(self.ontologies_map[a.target_onto], first_node=a.id2, walk_config=self.walk_config)\
                                .build_pooling_mask().build_mlm_mask(self.apply_masks).build_parent_child_pairs_mask() for a in batch_alignments]
            
        source_inputs = Globals.tokenizer([w.sentence for w in src_random_walks], return_tensors='pt',
                                          truncation=True, padding='max_length').to(Globals.device)

        src_mask_array = torch.cat([w.mlm_mask for w in src_random_walks], dim=0).to(Globals.device)
        source_inputs.input_ids = torch.where(src_mask_array, Globals.tokenizer.mask_token_id, source_inputs.input_ids)
        

        target_inputs = Globals.tokenizer([w.sentence for w in trg_random_walks], return_tensors='pt',
                                            truncation=True, padding='max_length').to(Globals.device)
        

        trg_mask_array = torch.cat([w.mlm_mask for w in trg_random_walks], dim=0).to(Globals.device)
        target_inputs.input_ids = torch.where(trg_mask_array, Globals.tokenizer.mask_token_id, target_inputs.input_ids)

        src_root_masks, src_concept_pool_masks, src_concepts, src_ontos, src_subclass_pairs = zip(*[(w.root_mask, w.concept_pool_masks, w.concepts, w.ontos, w.subclass_pairs) for w in src_random_walks])
        trg_root_masks, trg_concept_pool_masks, trg_concepts, trg_ontos, trg_subclass_pairs = zip(*[(w.root_mask, w.concept_pool_masks, w.concepts, w.ontos, w.subclass_pairs) for w in trg_random_walks])

        return   source_inputs, src_root_masks, src_subclass_pairs, src_concept_pool_masks, src_ontos, src_concepts,\
                 target_inputs, trg_root_masks, trg_subclass_pairs, trg_concept_pool_masks, trg_ontos, trg_concepts, \
                 torch.tensor([a.label for a in batch_alignments]).to(Globals.device).float()                                                      


    def __getitem__(self, idx):
            
        if idx >= self.__len__():
            raise IndexError

        try:
            batch_alignments = self.get_batch_alignments(idx)

            if self.shuffle:
                random.shuffle(batch_alignments)
        except Exception as e:
            traceback.print_exc()

        return self.build_inputs(batch_alignments)

    def __len__(self):
        pass


class IntraNegativeSampler:
    def __init__(self, ontologies_map, k=10, A=4, no_hard_negative_samples=False, negative_sampling_strategy="ontologic_relation") -> None:
        self.ontologies_map = ontologies_map
        self.A = A
        self.no_hard_negative_samples = no_hard_negative_samples
        self.strategy = negative_sampling_strategy

        self.build_semi_negative_training_set()
        self.semi_negative_index = 0

    
    def build_semi_negative_training_set(self):
        self.training_data = []
        for onto in self.ontologies_map.keys():
            for id in self.ontologies_map[onto].classes:
                if self.ontologies_map[onto].node_has_neighbours(id, add_obj_prop=False):
                    self.training_data.append((id, onto))




    def _random_concept_with_cousins(self):
        onto = random.choices(list(self.ontologies_map.keys()), weights=[len(o.cousins_set) for o in self.ontologies_map.values()])[0]
        id = random.choice(list(self.ontologies_map[onto].cousins_set))

        return id, onto

    def _random_concept(self):
        onto = random.choices(list(self.ontologies_map.keys()), weights=[len(o) for o in self.ontologies_map.values()])[0]
        id = random.choice(self.ontologies_map[onto].classes)

        return id, onto


    def hard_sample(self):
        id, onto = self._random_concept_with_cousins()

        other_id, k_hop = self.ontologies_map[onto].get_random_neighbour(id, strategy="cousins")


        alignment = Alignment(onto, onto, id, other_id, label = 0)


        return alignment 
    

    def sample_distance(self):
        id, onto = self.training_data[self.semi_negative_index]

        self.semi_negative_index += 1

        if self.semi_negative_index >= len(self.training_data):
            self.semi_negative_index = 0
                
        other_id, k_hop = self.ontologies_map[onto].get_random_neighbour(id, k_hop_max=self.A, strategy=self.strategy)
        
        alignment = Alignment(onto, onto, id, other_id, label = 1 - (k_hop) / self.A)

        return alignment 

    def soft_sample(self):
        
        while True:
            id, onto = self._random_concept()
            id2 = random.choice(self.ontologies_map[onto].classes)

            if not ((self.no_hard_negative_samples) and (id in self.ontologies_map[onto].get_cached_neighbours(id2, self.A))):
                return Alignment(onto, onto, id, id2, 0.)



class InterNegativeSampler:
    def __init__(self, ontologies_map: Dict[str, Ontology], alignments: List[Alignment], k=10, A=4, no_hard_negative_samples=False, negative_sampling_strategy="ontologic_relation") -> None:
        self.ontologies_map = ontologies_map
        self.alignments = alignments
        self.A = A
        self.no_hard_negative_samples = no_hard_negative_samples
        self.strategy = negative_sampling_strategy


        self.alignments_with_cousins = []
        for a in alignments:
            if self.ontologies_map[a.source_onto].node_has_cousins(a.id1) \
               or self.ontologies_map[a.target_onto].node_has_cousins(a.id2):

                self.alignments_with_cousins.append(a)



    def sample_distance(self):
        while True:
            sample: Alignment = deepcopy(random.choice(self.alignments))

            if random.random() > 0.5:
                if self.ontologies_map[sample.source_onto].node_has_neighbours(sample.id1, add_obj_prop=False):
                    sample.id1, k_hop = self.ontologies_map[sample.source_onto].get_random_neighbour(sample.id1, k_hop_max=self.A, strategy=self.strategy)
                    
                    sample.label = 1 - (k_hop) / self.A

                    return sample
            else:
                if self.ontologies_map[sample.target_onto].node_has_neighbours(sample.id2, add_obj_prop=False):
                    sample.id2, k_hop = self.ontologies_map[sample.target_onto].get_random_neighbour(sample.id2, k_hop_max=self.A, strategy=self.strategy)
                    
                    sample.label = 1 - (k_hop) / self.A

                    return sample



    def soft_sample(self):
        sample: Alignment = deepcopy(random.choice(self.alignments))
        while True:
            if random.random() > 0.5:
                
                other = random.choice(self.ontologies_map[sample.source_onto].classes)
                if (other != sample.id1): 
                    if not ((self.no_hard_negative_samples) and (sample.id1 in self.ontologies_map[sample.source_onto].get_cached_neighbours(other, self.A))):

                        sample.id1 = other
                        sample.label = 0.
                        return sample
            else:
                other = random.choice(self.ontologies_map[sample.target_onto].classes)

                if (other != sample.id2):
                   if not ((self.no_hard_negative_samples) and (sample.id2 in self.ontologies_map[sample.target_onto].get_cached_neighbours(other, self.A))):
                    sample.id2 = other
                    sample.label = 0.
                    return sample

    def hard_sample(self):

        sample: Alignment = deepcopy(random.choice(self.alignments_with_cousins))
        decision_threshold = 0.5

        if not self.ontologies_map[sample.source_onto].node_has_cousins(sample.id1):
            decision_threshold = 1.
        
        elif not self.ontologies_map[sample.target_onto].node_has_cousins(sample.id2):
            decision_threshold = 0.


        if random.random() > decision_threshold:
            # Replace source concept
            # sample.id1, k_hop = self.ontologies_map[sample.source_onto].get_random_neighbour(sample.id1, strategy="cousins")
            sample.label = 0

            return sample
        else:
            # Replace target concept

            # sample.id2, k_hop = self.ontologies_map[sample.target_onto].get_random_neighbour(sample.id2, strategy="cousins")
            sample.label = 0
            
            return sample 






class SemiNegativeSampling(AlignmentBatchLoader):
    def __init__(self, ontologies_map, train_positive_alignments, iir=0.8, inter_soft_r=0.5, intra_soft_r=0.0,
                 no_hard_negative_samples=False, epoch_over_alignments=False, A=4, negative_sampling_strategy="ontologic_relation", **kwargs):
        """Constructor for the Inter-Intra sampler

        Args:
            ontologies_map (Dict): Dictionnary of Ontologies
            train_positive_alignments (List[Alignment]): list of positive alignments 
            iir (float, optional): inter/intra generation ratio of samples, 1.0 is only inter sampling, 0. is only intra. Defaults to 0.8.
            inter_soft_r (float, optional): distance/soft ration of sampling for inter ontology samples. Defaults to 0.5.
            intra_soft_r (float, optional): distance/soft ration of sampling for intra ontology samples. Defaults to 0.0.
            no_hard_negative_samples (bool, optional): True is the sampler should exclude neighbouring pairs in negative soft sampling. Defaults to False.
        """
        super().__init__(ontologies_map, **kwargs)
        
        self.epoch_over_alignments = epoch_over_alignments
        self.no_hard_negative_samples = no_hard_negative_samples

        self.alignments = train_positive_alignments
        self.positive_intra_dataset = [(id, onto) for onto in self.ontologies_map for id in self.ontologies_map[onto].classes]

        self.iir = iir
        self.inter_soft_r = inter_soft_r
        self.intra_soft_r = intra_soft_r
        self.idx_positive = -1
        self.idx_positive_intra = -1
        self.A = A

        self.intraNegativeSampler = IntraNegativeSampler(ontologies_map, no_hard_negative_samples, A=self.A, negative_sampling_strategy=negative_sampling_strategy)
        self.interNegativeSampler = InterNegativeSampler(ontologies_map, train_positive_alignments, no_hard_negative_samples, A=self.A, negative_sampling_strategy=negative_sampling_strategy)


    def _random_concept(self):
        onto = random.choices(list(self.ontologies_map.keys()), weights=[len(o) for o in self.ontologies_map.values()])[0]
        id = random.choice(self.ontologies_map[onto].classes)

        return id, onto

    def _generate_positive_sample(self):

        if random.random() > self.iir:

            # Intra positive sampling
            self.idx_positive_intra += 1
            if self.idx_positive_intra >= len(self.positive_intra_dataset):
                self.idx_positive_intra = 0
                random.shuffle(self.positive_intra_dataset)

            concept, onto = self.positive_intra_dataset[self.idx_positive_intra]

            return Alignment(onto, onto, concept, concept) 
        else:
            # Inter positive sampling
            self.idx_positive += 1
            if self.idx_positive >= len(self.alignments):
                self.idx_positive = 0
                random.shuffle(self.alignments)

            return self.alignments[self.idx_positive]


    def _generate_negative_sample(self):

        # Generate inter or intra sample
        if random.random() < self.iir:
            if random.random() > self.inter_soft_r:
                return self.interNegativeSampler.sample_distance()
            else:
                return self.interNegativeSampler.soft_sample()

        else:
            if random.random() > self.intra_soft_r:
                return self.intraNegativeSampler.sample_distance()
            else:
                return self.intraNegativeSampler.soft_sample()

            

    def get_batch_alignments(self, id):


        batch_positive_alignments = [self._generate_positive_sample() for i in range(self.n_alignments_per_batch)]
        batch_negative_alignments = [self._generate_negative_sample() for _ in range(self.batch_size - self.n_alignments_per_batch)]
        
        batch_alignments = batch_positive_alignments + batch_negative_alignments

        return batch_alignments


    def __len__(self):

        if self.epoch_over_alignments:
            return (len(self.alignments)//(self.n_alignments_per_batch))
        else:
            return sum(map(lambda onto: len(onto), self.ontologies_map.values()))//(self.batch_size)
 
        

class ReferenceAlignmentsBatchLoader(AlignmentBatchLoader):
    def __init__(self, ontologies_map, ontologies_in_alignments, alignments, negative_alignments=None, **kwargs):
        super().__init__(ontologies_map, **kwargs)

        self.alignments: List[Alignment] = alignments
        self.negative_alignments: List[Alignment] = negative_alignments

        self.ontologies_in_alignments: Tuple[str, str] = ontologies_in_alignments
        self._ontologies_in_alignments_weights = [len(self.ontologies_map[o1])*len(self.ontologies_map[o2]) 
                                                   for o1, o2 in self.ontologies_in_alignments]

        self.n_alignments_per_batch = self.batch_size


    def get_batch_alignments(self, idx):
        if idx == 0:
            random.shuffle(self.alignments)

        batch_positive_alignments = self.alignments[(idx*self.n_alignments_per_batch) : min((idx+1)*self.n_alignments_per_batch, len(self.alignments))]
        
        
        return batch_positive_alignments


    def __len__(self):
        return (len(self.alignments)//self.batch_size)*self.n_alignments_per_batch




class DAEBatchLoaderInference:
    def __init__(self, ontology, batch_size = 32) -> None:
        self.batch_size = batch_size
        self.ontology = ontology

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError

        return self.ontology.classes[(idx*self.batch_size) : min((idx+1)*self.batch_size, len(self.ontology.classes))]
    
    def __len__(self):
        return ceil(len(self.ontology) / self.batch_size) 


class DAEBatchLoader:
    def __init__(self, ontologies_map, alignments, batch_size=32, **kwargs):

        self.alignments: List[Alignment] = alignments
        self.ontologies_map = ontologies_map
        random.shuffle(self.alignments)
        self.batch_size = batch_size

    def get_batch_alignments(self, idx):
        if idx == 0:
            random.shuffle(self.alignments)

        batch_positive_alignments = self.alignments[(idx*self.batch_size) : min((idx+1)*self.batch_size, len(self.alignments))]
        

        return batch_positive_alignments

    def __getitem__(self, idx):
            
        if idx >= self.__len__():
            raise IndexError

        return self.get_batch_alignments(idx), self.ontologies_map

    def __len__(self):
        return (len(self.alignments)//self.batch_size)





