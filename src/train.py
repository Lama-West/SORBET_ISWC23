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
from models.sorbet import SORBET
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
from batch_loaders.random_walk import TreeWalkConfig, WalkStrategy
 


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




writer = SummaryWriter(log_dir=config["General"]["metrics_folder"])

# Pairwise Trainer
trainer = TrainPipeline(
    config, 
    ["conference", "anatomy"],
    extra_tracks=None,

    # SORBET Training parameters
    epochs = 20,
    lr = 1e-4,
    save_model_checkpoints=-1,
    save_embeddings_checkpoints=-1,

    model=SORBET(from_pretrained=config["General"]["model"], pooling_strategy="only_concept"),
    train_walks = TreeWalkConfig(n_branches=(0,5), use_synonyms=True),
    loader_config = {
        "iir":0.8, 
        "inter_soft_r":0.5, 
        "intra_soft_r":0.2, 
        "negative_sampling_strategy": "ontologic_relation",
        "no_hard_negative_samples":False,
        "epoch_over_alignments": False,
        "A": 5,
        "batch_size":32, 
        "n_alignments_per_batch":8
    },

    # Inference on Ontology Alignment or Subsumption prediction tasks for testing
    run_tasks=True,
    test_size=1.0, 
    consider_train_set=False,
    inference_walks = TreeWalkConfig(strategy=WalkStrategy.ONTOLOGICAL_RELATIONS, n_branches=5),
    inference_config={"candidate_selector": None,
                        "string_matching_optimization": False,
                        "matcher": GreedyMatcher(),
                        "thresholds": [0.6, 0.65, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85]},
    metrics_config={"results_files_path": "./result_alignments",
                "write_rdf": False,
                "write_tsv": False,
                "write_ranking": False,
                "hits":[1, 3, 5, 10], 
                "debug_files_path": "./debug"},
    tensorboard_writer=writer
)


trainer.train()

