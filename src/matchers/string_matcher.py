import sys

from batch_loaders.alignment import Alignment


class StringMatcher:
    def __init__(self) -> None:
        pass

    def run_ontos_map(self, onto_map):
        ontos = list(onto_map.keys())

        if len(ontos) < 2:
            return []

        alignments = []

        for i, onto_1 in enumerate(ontos[:-1]):
            for onto_2 in ontos[i+1:]:
                alignments.extend(self.run(onto_map[onto_1], onto_map[onto_2]))

        return alignments

    def run(self, source, target):
        

        alignments = set({})
        class_1 = set({})
        class_2 = set({})

        dict_label_2 = {" ".join(target.get_id_label(class_id_2)):class_id_2 for class_id_2 in target.classes}


        # First check for rdf:label matches
        for class_id in source.classes:
            label = " ".join(source.get_id_label(class_id))
            if label in dict_label_2:
                class_1.add(class_id)
                class_2.add(dict_label_2[label])
                alignments.add(Alignment(source.onto_name, target.onto_name, class_id, dict_label_2[label]))
                    

        # Second check for synonym matches
        dict_synonym_2 = {}
        for class_id_2 in target.classes:
            for synonym_2 in target.synonyms[class_id_2]:
                dict_synonym_2[" ".join(target.get_id_label(synonym_2))] = class_id_2

        for class_id in source.classes:
            if class_id in class_1:
                continue

            for synonym in source.synonyms[class_id]:
                label = " ".join(source.get_id_label(synonym))
                if  (label in dict_synonym_2) and (dict_synonym_2[label] not in class_2):

                    class_1.add(class_id)
                    class_2.add(dict_synonym_2[label])
                    alignments.add(Alignment(source.onto_name, target.onto_name, class_id, dict_synonym_2[label]))


        alignments = list(alignments)

        return alignments