
import re, logging
from copy import deepcopy
import numpy as np
from w3lib.html import replace_entities
from nltk.corpus import stopwords 
from autocorrect import Speller
from nltk import ngrams
import json

logger = logging.getLogger("onto") 

class PreprocessingPipeline():
    """Class to preprocess string in ontologies"""

    def __init__(self) -> None:
        self.stop_words = set(stopwords.words('english')) 
        self.speller = Speller()
        self.prefix_list = ["non"]
        self.suffix_list = ["ed"]
        self.accronym_dict = None

    def set_params(self, bert_vocab=None, autocorrect=False):
        self.vocab = bert_vocab
        self.prefix_list = ["non"]
        self.suffix_list = ["ed"]
        self.autocorrect = autocorrect

    def export_accronym_resolution_dict(self):
        with open("accronyms.json", "w") as f:
            json.dump(self.accronym_dict, f, indent=6)

    def import_accronym_resolution_dict(self, file):
        with open(file, "r") as f:
            self.accronym_dict = json.load(f)

    def fit_to_vocab(self, labels):
        "Function to find ngrams for accronym resolution"

        labels = [self.remove_stopwords(l) for l in labels ]

        ngrams_list = []
        for l in labels:
            l = [s.capitalize() for s in l]
            for ngram in ngrams(l, 2):
                ngrams_list.append(list(ngram))
            for ngram in ngrams(l, 3):
                ngrams_list.append(list(ngram))
            for ngram in ngrams(l, 4):
                ngrams_list.append(list(ngram))

        accronym_dict = {}
        ngram_counts = {}

        for ngram in ngrams_list:
            accronym = "".join([l[0] for l in ngram])
            ngram = " ".join(ngram)
            if accronym not in accronym_dict:
                accronym_dict[accronym] = []
                
            if ngram not in ngram_counts:
                ngram_counts[ngram] = 1
                accronym_dict[accronym].append(ngram)
            else:  
                ngram_counts[ngram] += 1

        self.accronym_dict = {}

        for accronym in accronym_dict:
            if np.max([ngram_counts[ngram] for ngram in accronym_dict[accronym]]) > 3:
                best_ngram = np.argmax([ngram_counts[ngram] for ngram in accronym_dict[accronym]])

                self.accronym_dict[accronym] = accronym_dict[accronym][best_ngram]

        

    def remove_punction(self, words):
        return re.sub("[\[\]\-\.\(\),]", " ", words)

    def remove_stopwords(self, words):
        filtered = [word for word in words if word not in self.stop_words]
        return filtered if len(filtered) > 0 else words

    def camel_case_split(self, words):
        # Splits camel case strings
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', words)
        return [m.group(0) for m in matches]

    def underscore_split(self, words):
        return words.split("_")

    def space_split(self, words):
        return re.split(" +", words)

    def spell_check(self, words):
        new_words =  []
        for word in words:
            if word in self.vocab:
                new_words.append(word)
            else:
                new_word = None
                # Check if for prefix correction
                for prefix in self.prefix_list:
                    if (re.match(f"^{prefix}", word)) and (word[len(prefix):] in self.vocab):
                        new_word = prefix + "-" + word[len(prefix):] 
                        logger.debug("corrected prefix: [{}] to [{}]".format(word, new_word))

                if new_word is not None:
                    new_words.append(new_word)
                    continue

                # Remove suffix if the word without it is in vocab
                for suffix in self.suffix_list:
                    if (re.search(f"{suffix}$", word)) and (word[:-1*len(suffix)] in self.vocab):
                        new_word = word[:-1*len(suffix)]
                        logger.debug("corrected suffix: [{}] to [{}]".format(word, new_word))

                if new_word is not None:
                    new_words.append(new_word)
                    continue

                if new_word is None:
                    new_word = self.speller(word)
                    if word != new_word:
                        logger.debug("corrected spelling: [{}] to [{}]".format(word, new_word))
                new_words.append(new_word)

        return new_words

    def lowercase(self, words):
        return [word.casefold() for word in words]

    def flatten(self, tokens):
        flattened = []
        for token in tokens:
            flattened.extend(token)
        return flattened

    def parse_html_characters(self, words):
        return replace_entities(words)


    def _print_and_return_accronym(self, word):
        if word in self.accronym_dict:

            logger.debug(f"substituted accronym: [{word}] for [{self.accronym_dict[word]}]")

            return self.accronym_dict[word]
        else:
            return word

    def substitute_accronyms(self, words):
        
        return [self._print_and_return_accronym(word) for word in words]


        

    def process(self, label):

        processed_labels = deepcopy(label)

        processed_labels = self.parse_html_characters(processed_labels)

        processed_labels = [self.remove_punction(processed_labels)]

        processed_labels = self.flatten([self.space_split(token) for token in processed_labels])

        processed_labels = self.flatten([self.camel_case_split(token) for token in processed_labels])

        processed_labels = self.flatten([self.underscore_split(token) for token in processed_labels])

        if self.accronym_dict is not None:
            processed_labels = self.substitute_accronyms(processed_labels)

        processed_labels = self.lowercase(processed_labels)

        # processed_labels = self.remove_stopwords(processed_labels)

        if self.autocorrect:
            processed_labels = self.spell_check(processed_labels)

        return processed_labels

preprocessor = PreprocessingPipeline()
