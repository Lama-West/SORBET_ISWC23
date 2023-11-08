import random

import random, os
from typing import List

from transformers import BertForMaskedLM, BertTokenizer, AutoModel, AutoTokenizer
from batch_loaders.MLM_batch_loader import RandomTreeWalkBatchLoader
from batch_loaders.alignment_batch_loader import AlignmentBatchLoader
from batch_loaders.pair_alignment_batch_loader import PairAlignmentBatchLoader
from matchers.greedy_matcher import GreedyMatcher
from matchers.stable_marriage import StableMarriage
from metrics import Metrics
from models.mlm_model import MLMOntoBert

from models.pair_alignment_model import PairOntoBert
from models.siamese_model import SiameseOntoBert
from models.tf_idf_similarity import SubTokenSimilarity
from models.sbert_model import SBertModel
from models.test_model import TestModel
from models.owl2vec_similarity import Owl2VecModel
from models.structure_similarity_decorator import StructureSimilarityDecorator
# from models.owl2vec_similarity import Owl2VecModel
from train_pipeline import TrainPipeline
import json
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from globals import Globals
from batch_loaders.random_walk import RandomWalkConfig, WalkStrategy
 


logger = logging.getLogger("onto")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s - %(asctime)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info(f"Using gpu: {str(torch.cuda.is_available())}")
logger.info(f"CUDA version: {torch.version.cuda}")

with open("config.json", 'r') as f:
    config = json.load(f)

Globals.tokenizer = AutoTokenizer.from_pretrained(config["General"]["model"])
Globals.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO
# pas d'apprentissage sur les embeddings apres MLM
# MLM avec moins de MASK pour largebio
# Pas de synonyms pour pairwise

def train(tracks, extra_tracks=None, pretrained=None, candidate_selector="none", matcher="greedy", loader_config=None, 
                pretrained_epoch="", train_walks=None, inference_walks=None, consider_train_set=False, test_size=0.8, save=False):

    if loader_config is None:
        loader_config = {}

    if isinstance(tracks, List) :
        fine_tuned_path = config["General"]["general_fine_tuned_path"]
        pair_tuned_path = config["General"]["general_pair_tuned_path_2"]
    else:
        fine_tuned_path = config[tracks]["fine_tuned_path"]
        pair_tuned_path = config[tracks]["pair_tuned_path"]
        tracks=[tracks]



    if pretrained is None:
        pretrained = fine_tuned_path 

    if candidate_selector=="tfidf" :
        _candidate_selector=SubTokenSimilarity()
    elif candidate_selector == "ontobert":
        _candidate_selector=MLMOntoBert(from_pretrained=fine_tuned_path+pretrained_epoch)
    elif candidate_selector == "none":
        _candidate_selector=None
    else: raise ValueError("Not a valid candidate selector")


    if (matcher is None) or (matcher == "stable marriage"):
        _matcher = StableMarriage()
    elif (matcher == "greedy"):
        _matcher = GreedyMatcher() 

    logger.info("Pairwise training started")
    logger.info(f"pretrained: {pretrained}")
    logger.info(f"candidate_selector: {candidate_selector}")
    logger.info(f"matcher: {matcher} with thresholds {str([0.6, 0.65, 0.7, 0.725, 0.75, 0.775, 0.8, 0.85])}")
    logger.info(f"Test size: {test_size}")
    logger.info(f"Extra track: {extra_tracks}")
    for key, value in loader_config.items():
        logger.info(f"{key}: {value}")

    writer = SummaryWriter(log_dir=config["General"]["metrics_folder"])

    # Pairwise Trainer
    trainer = TrainPipeline(
        train_walks = train_walks,
        inference_walks = inference_walks,
        model=SiameseOntoBert(from_pretrained=pretrained, pooling_strategy="only_concept"),
        loader_config=loader_config,
        run_metrics_bool=True,
        metrics_config={"results_files_path": "./result_alignments",
                        "write_rdf": False,
                        "write_tsv": False,
                        "write_ranking": False,
                        "hits":[1, 3, 5, 10], 
                        "debug_files_path": "./debug"},
        inference_config={"candidate_selector": _candidate_selector,
                          "string_matching_optimization": False,
                          "matcher":_matcher,
                          "thresholds": [0.6, 0.65, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85]},
                        #   "thresholds": [0.75]},
                        #   "thresholds": [0.45, 0.5, 0.55]},
        tensorboard_writer=writer
    )

    trainer.train(config, tracks, extra_tracks=extra_tracks, epochs=75, test_size=test_size, consider_train_set=consider_train_set, \
                  save_model=False, save_path=f"./store/{'_'.join(tracks)}/", lr=1e-5)

    trainer.run_metrics(trainer.tracks)



no_walk = RandomWalkConfig(n_branches=0)
infer_walk = RandomWalkConfig(strategy=WalkStrategy.ONTOLOGICAL_RELATIONS, n_branches=5)


loader_config = {
    "iir":0.8, 
    "inter_soft_r":0.0, 
    "intra_soft_r":0.0, 
    "negative_sampling_strategy": "ontologic_relation",
    "no_hard_negative_samples":False,
    "epoch_over_alignments": False,
    "A": 5,
    "batch_size":32, 
    "n_alignments_per_batch":4
}



train(["conference"], pretrained=config["General"]["model"], \
              test_size=1.0, consider_train_set=False, loader_config=loader_config, train_walks=infer_walk, inference_walks=infer_walk)

