
import copy
from nltk.corpus import wordnet as wn
from numpy import dot
from numpy.linalg import norm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from globals import Globals

class SynonymGenerator():

    def __init__(self) -> None:
        self.context = None
        self.context_len = 0
        self.synonym_model = None

        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").to(Globals.device)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

    def generate_synonyms(self, labels):

        with torch.no_grad():
                concept = " ".join(labels)

                inputs = self.tokenizer('Give synonyms for the concept of "{}" in the context of a scientific conference'.format(concept), return_tensors="pt").to(Globals.device)

                outputs = self.model.generate(**inputs)

                synonym_1 = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

                inputs = self.tokenizer('Give a paraphrase for "{}" in the context of a scientific conference'.format(concept), return_tensors="pt").to(Globals.device)

                outputs = self.model.generate(**inputs)

                synonym_2 = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


                return [synonym_1, synonym_2]


    def set_context(self, all_words):
        pass
        # if not self.synonym_model:
        #     from sentence_transformers import SentenceTransformer
        #     self.synonym_model = SentenceTransformer('all-mpnet-base-v2')     

        # if self.context is None:
        #     self.context_len += len(all_words)
        #     self.context = self.synonym_model.encode(" ".join(all_words), show_progress_bar=False)
        # else:
        #     new_context = self.synonym_model.encode(" ".join(all_words), show_progress_bar=False)
        #     self.context = (self.context_len*self.context + len(all_words)*new_context) / (self.context_len+len(all_words))

        #     self.context_len = self.context_len+len(all_words)

    # def generate_synonyms_old(self,labels):

    #     if not self.synonym_model:
    #         from sentence_transformers import SentenceTransformer
    #         self.synonym_model = SentenceTransformer('all-mpnet-base-v2')        

    #     synonym_sentence = []

    #     for i, word in enumerate(labels):
                    
    #         synonyms = list(set([" ".join(synonym.lower().split("_"))  for ss in wn.synsets(word) for synonym in ss.lemma_names()]))
            
    #         words_copy = copy.copy(labels)
    #         for synonym in synonyms:
    #             words_copy[i] = synonym
                
    #             if words_copy != labels:
    #                 synonym_sentence.append(" ".join(words_copy))
        
    #     if len(synonym_sentence) == 0:
    #         return []

    #     initial_embedding = self.synonym_model.encode([" ".join(labels)], show_progress_bar=False)

    #     embeddings = self.synonym_model.encode(synonym_sentence, show_progress_bar=False)

    #     syn_cos_sim = ((initial_embedding  @ embeddings.T) /(norm(initial_embedding)*norm(embeddings, axis=1))).reshape(-1,).tolist()
    #     context_cos_sim = ((self.context  @ embeddings.T) /(norm(self.context)*norm(embeddings, axis=1))).reshape(-1,).tolist()

    #     kept_synonyms = [(x, sym_score, context_score) 
    #                         for x, sym_score, context_score
    #                              in sorted(zip(synonym_sentence, syn_cos_sim, context_cos_sim), key=lambda x: x[1], reverse=True)]

    #     kept_synonyms = [x for x, sym_score, context_score in kept_synonyms if ((context_score > 0.25) and (sym_score > 0.75))][:5]

    #     return kept_synonyms

