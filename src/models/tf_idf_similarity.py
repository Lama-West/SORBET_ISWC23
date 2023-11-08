from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from models.base_model import BaseModel

from batch_loaders.ontology_parsing.ontology import Ontology

from globals import Globals

class SubTokenSimilarity(BaseModel):
    cache = {}

    def __init__(self, strategy = "count_vectorizer", k_hop=1) -> None:
        super().__init__()

        self.k_hop = k_hop

        self.count_strategy = strategy

        SubTokenSimilarity.cache = {}

    def _cache_key(self, source: Ontology, target: Ontology):
        return source.onto_name + "@" + target.onto_name

    def get_sorted_similarity(self, source: Ontology, target: Ontology):
        cache_key = self._cache_key(source, target)

        if cache_key not in SubTokenSimilarity.cache:
            self.get_similarity_matrix(source, target)

        return  SubTokenSimilarity.cache[cache_key]["similarity"], \
                SubTokenSimilarity.cache[cache_key]["indices"],\
                SubTokenSimilarity.cache[cache_key]["src_keys"], \
                SubTokenSimilarity.cache[cache_key]["trg_keys"]



    def get_similarity_matrix(self, source: Ontology, target: Ontology):

        cache_key = self._cache_key(source, target)
        if cache_key in SubTokenSimilarity.cache:
            return SubTokenSimilarity.cache[cache_key]["similarity"], \
                   SubTokenSimilarity.cache[cache_key]["src_keys"], \
                   SubTokenSimilarity.cache[cache_key]["trg_keys"]

        src_embeddings, src_keys = self.bert_tf_idf(source)
        trg_embeddings, trg_keys = self.bert_tf_idf(target)

        # Initialize an instance of tf-idf Vectorizer
        counter = CountVectorizer()

        if self.count_strategy == "tfidf_vectorizer":
            # Generate the tf-idf vectors for the corpus
            counter.fit(src_embeddings + trg_embeddings)

            tfidf_vectorizer = TfidfVectorizer(vocabulary=counter.vocabulary_)
            tfidf_vectorizer2 = TfidfVectorizer(vocabulary=counter.vocabulary_)
            src_tfidf = tfidf_vectorizer.fit_transform(src_embeddings)
            
            trg_count  = tfidf_vectorizer2.fit_transform(trg_embeddings).astype(np.bool)

            cosine_similarities = np.dot(src_tfidf,trg_count.T).toarray()

            division_factor = np.max(cosine_similarities, axis=1)

            division_factor = np.reshape(np.where(~division_factor.astype(np.bool), 1.,  division_factor), (-1,1))

            # capping max similarity at 1.
            cosine_similarities = cosine_similarities / division_factor

        elif self.count_strategy == "count_vectorizer":
            # Generate the tf-idf vectors for the corpus
            counter.fit(src_embeddings + trg_embeddings)
            vocab = counter.vocabulary_

            counter_2 = CountVectorizer(vocabulary = vocab)

            src_count = counter_2.fit_transform(src_embeddings)
            src_count = src_count/src_count.sum(axis=1)

            trg_count  = counter_2.fit_transform(trg_embeddings)
            trg_count = trg_count/trg_count.sum(axis=1)

            cosine_similarities = np.dot(src_count,trg_count.T)

            division_factor = np.max(src_count, axis=1)

            # capping max similarity at 1.
            cosine_similarities = cosine_similarities / division_factor

        # Set in cache
        self._save_similarities(cache_key, cosine_similarities, src_keys, trg_keys)

        cosine_similarities_torch = torch.from_numpy(cosine_similarities).to(Globals.device)

        return cosine_similarities_torch, src_keys, trg_keys

    def _save_similarities(self, cache_key, cosine_similarities, src_keys, trg_keys):
        SubTokenSimilarity.cache[cache_key] = {}
        SubTokenSimilarity.cache[cache_key]["similarity"] = cosine_similarities
        SubTokenSimilarity.cache[cache_key]["indices"] = (-cosine_similarities).argsort(axis=1) 
        SubTokenSimilarity.cache[cache_key]["src_keys"] = src_keys
        SubTokenSimilarity.cache[cache_key]["trg_keys"] = trg_keys



    def bert_tf_idf(self, onto: Ontology):
        keys = onto.classes
        terms = []
        for key in keys:
            ids = [key] + onto.get_node_neighbours(key, k_hop_max=self.k_hop)
            ids = [id for hops in ids for id in hops]
            syonyms_list = [self.get_synonyms(id, onto) for id in ids]
            syonyms_list = [synonym for l in syonyms_list for synonym in l] # Flatten list
            syonyms = " ".join(syonyms_list)
            # tokens = syonyms
            tokens = " ".join(Globals.tokenizer.tokenize(syonyms))
            terms.append(tokens)

        return terms, keys

    def get_synonyms(self, id, onto):
        return list(set([synonym for synonym_id in onto.synonyms[id] for synonym in onto.get_id_label(synonym_id)]))


