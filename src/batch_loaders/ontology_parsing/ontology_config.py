
class OntoConfig:
    def __init__(self, config=None) -> None:
        
        if not config:
            config = {}

        self.use_label = config["use_label"] if "use_label" in config else False

        self.exclude_classes = config["exclude_classes"] if "exclude_classes" in config else []

        self.use_synonyms = config["use_synonyms"] if "use_synonyms" in config else False

        self.use_cache = config["use_cache"] if "use_cache" in config else False

        self.autocorrect = config["autocorrect"]  if "autocorrect" in config else False
        
        self.synonym_extension = config["synonym_extension"] if "synonym_extension" in config else False

        self.subclass_of_properties = config["subclass_of_properties"] if "subclass_of_properties" in config else []

