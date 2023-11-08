


from ctypes import alignment
import logging
import os
from typing import List
from torch import embedding, threshold
from transformers import BertTokenizer, BertForPreTraining, DistilBertTokenizer, \
                         DistilBertForMaskedLM, BertForMaskedLM,AutoTokenizer, AutoModel
import torch
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from models.base_model import BaseModel
from inference_pipeline import InferencePipeline

from track import Track
from batch_loaders.alignment import Alignment, AlignmentDataset
from batch_loaders.ontology_parsing.ontology import Ontology
from batch_loaders.ontology_parsing.ontology_config import OntoConfig
from losses.SentenceEmbeddingLoss import SentenceEmbeddingLoss
from losses.InitialEmbeddingLoss import InitialEmbeddingLoss
import time
import copy
import uuid
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


from matchers.string_matcher import StringMatcher
from matchers.isolated_class_matcher import IsolatedClassMatcher

from metrics import Metrics, Results


from globals import Globals

logger = logging.getLogger("onto")


class TrainPipeline:
    def __init__(self, train_walks=None, inference_walks=None, loader_config=None,
                       model:BaseModel=None, inference_config=None, metrics_config=None, run_metrics_bool=True, tensorboard_writer: SummaryWriter=None) -> None:

        self.train_walks = train_walks

        self.run_metrics_bool = run_metrics_bool
        self.metrics_config = metrics_config

        self.model = model

        self.inference_config = inference_config

        self.inference_pipeline = InferencePipeline(walk_config=inference_walks, **self.inference_config, )
        self.inference_pipeline.model = model

        self.run_id = str(uuid.uuid4())[:8]

        self.tensorboard_writer = tensorboard_writer

        if loader_config is None: 
            self.loader_config = {}
        else: 
            self.loader_config = loader_config

        logger.info("Started training with run_id: {self.run_id}")


    def save_model(self, path):
        self.model.model.save_pretrained(path)

    def run_metrics(self, tracks: List[Track], debug=False, epoch=-1):
        self.model.reset_embedding_dict()

        all_results = []

        for track in tracks:
            logger.info(f"Metrics for track: {track.name}")

            track_results = track.metrics.run(self.inference_pipeline, track.ontologies_map, track.alignment_per_dataset)
            all_results.append(track_results)

            if debug:
                self._write_onto_embeddings_tensorboard(self.model.get_embeddings_dict(), track=track, epoch=epoch, tag="MLM_" + track.name)

                track.metrics.write_debug(epoch = epoch)

            track.metrics.write_results(epoch = epoch)
        
        self.inference_pipeline.free_candidate_selector()

        return all_results


    def train(self, config, tracks, extra_tracks=None, epochs=30, lr=1e-5, test_size=0.8, consider_train_set=False, save_model=False, save_path=None, write_embeddings=False):

        tracks = [Track(track, config, metrics_config=self.metrics_config) for track in tracks]
        for t in tracks:
            t.split_train_test(test_size, consider_train_set=consider_train_set)

        self.tracks = tracks

        all_test_tracks = sum(tracks)        


        if extra_tracks is not None:    
            extra_tracks = [Track(track, config, metrics_config=self.metrics_config) for track in extra_tracks]
            all_extra_tracks = sum(extra_tracks)  
            all_tracks = all_test_tracks + all_extra_tracks
        else:
            extra_tracks = []
            all_tracks = all_test_tracks


        if self.model.is_supervised:

            positive_alignments = all_test_tracks.train_alignments

            logger.info(f"Reference train alignments size: {len(positive_alignments)}")

            if (len(extra_tracks) > 0) and (len(all_extra_tracks.alignments) > 0):
                positive_alignments.extend(np.random.choice(all_extra_tracks.alignments, size=1000))

        else:
            positive_alignments = []

        # SentenceEmbeddingLoss._precompute_sentence_embeddings(all_test_tracks.ontologies)
        

        string_matcher = StringMatcher()
        string_alignments = string_matcher.run_ontos_map(all_test_tracks.ontologies_map)
        train_alignments = positive_alignments + string_alignments
        train_alignments = list(set(train_alignments))

        # isc_matcher = IsolatedClassMatcher()
        # isolated_class_alignments = isc_matcher.run_tracks(tracks, train_alignments)

        train_alignments = positive_alignments + string_alignments #+ isolated_class_alignments
        train_alignments = list(set(train_alignments))


        logger.info(f"train alignments size: {len(train_alignments)}")

        self.loader_config["ontologies"] = all_tracks.ontologies
        self.loader_config["ontologies_map"] = all_tracks.ontologies_map
        self.loader_config["alignments"] = all_tracks.alignments
        self.loader_config["train_positive_alignments"] = train_alignments

        # InitialEmbeddingLoss._precompute_sentence_embeddings(self.inference_pipeline.candidate_selector, all_tracks.ontologies)



        if self.run_metrics_bool:

            start = time.time()
            results = self.run_metrics(tracks, debug=True, epoch=-1)

            end = time.time()
            eval_time = str(end-start)
            logger.info(f"Eval time: {eval_time}")
            
        
        if write_embeddings:
                for track in self.tracks:
                    for _, onto in track.ontologies_map.items():
                        self._write_embedding_file(onto, self.model, 0)
        
        loader = self.model.loader_type(**self.loader_config, walk_config=self.train_walks)

        optim = Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            losses = []

            logger.info(f"################################################# ---------- EPOCH {epoch} ---------- #################################################")
            self.model.model = self.model.model.to(Globals.device)
            start = time.time()
            pytorch_time = []
            # setup loop with TQDM and dataloader

            for inputs in tqdm(loader, leave=True, disable=True):
                # initialize calculated gradients (from prev step)
                optim.zero_grad()

                # process
                py_time = time.time()
                loss = self.model.get_loss(*inputs)
                # extract loss
                # TODO: this loss is temporary
                losses.append(loss)
                # calculate loss for every parameter that needs grad update
                sum(loss.values()).backward()

                # update parameters
                optim.step()

                py_time = time.time() - py_time
                pytorch_time.append(py_time)

            end = time.time()
            epoch_time = str(end-start)
            pytorch_time = str(sum(pytorch_time))

            losses = pd.DataFrame(losses)

            logger.info(f'Epoch: {epoch} time:(total= {epoch_time}, pytorch={pytorch_time}), losses: (f{" ".join([str(loss) + ": " + str(losses[loss].mean()) for loss in losses.columns])})')

            if self.tensorboard_writer is not None:

                for loss in losses.columns:
                    self.tensorboard_writer.add_scalar(loss, losses[loss].mean(), epoch)
                    logger.info(f'{loss}: {losses[loss].mean()}')
                    

            if save_model:
                self.save_model(f"{save_path}_{epoch}")

            start = time.time()
    
            if self.run_metrics_bool:
                results = self.run_metrics(tracks, debug=True, epoch=epoch)

            end = time.time()
            eval_time = str(end-start)
            logger.info(f"Eval time: {eval_time}")

            if write_embeddings:
                for track in self.tracks:
                    for _, onto in track.ontologies_map.items():
                        self._write_embedding_file(onto, self.model, epoch)
        


    def _write_embedding_file(self, ontology, model, epoch=0):

        if ontology.onto_name not in model.embeddings_cache:
            model.get_embeddings(ontology)

        embeddings = F.normalize(model.embeddings_cache[ontology.onto_name]["embeddings"])
        labels = model.embeddings_cache[ontology.onto_name]["embeddings_key"]
        labels = "\n".join([os.path.join(ontology.onto.base_iri, l) for l in labels])

        folder = os.path.join("./results", f"{ontology.onto_name}_embeddings/")
        if not os.path.exists(folder):
            os.makedirs(folder)

        torch.save(embeddings, os.path.join("./results", f"{ontology.onto_name}_embeddings/embeddings_{epoch}.pt"))
        with open(os.path.join("./results", f"{ontology.onto_name}_embeddings/labels_{epoch}.txt"), "w") as f:
            f.write(labels)





    def _write_onto_embeddings_tensorboard(self, embeddings_dict, track: Track, epoch=None, tag="default"):
        if embeddings_dict is None:
            logger.warning("Cannot get embeddings from model")
            return

        tracks_ontologies = {onto.onto_name for onto in track.ontologies}
        embeddings = None
        labels = None
        for onto_name, onto_data in embeddings_dict.items():
            if onto_name in tracks_ontologies:
                if embeddings is None:
                    embeddings = onto_data["embeddings"]
                    labels = onto_data["embeddings_label"]
                else:
                    embeddings = torch.cat((embeddings, onto_data["embeddings"]), dim=0)
                    labels = labels + onto_data["embeddings_label"]


        self.tensorboard_writer.add_embedding(embeddings, metadata=labels, global_step=epoch, tag=tag)     