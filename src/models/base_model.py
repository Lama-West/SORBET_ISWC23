from batch_loaders.ontology_parsing.ontology import Ontology

class LoaderPlaceholder:
    def __init__(self, **kwargs) -> None:
        pass

class BaseModel():
    def __init__(self) -> None:
        self.is_supervised = False    
        self.loader_type = LoaderPlaceholder
    
    def get_similarity_matrix(self, source: Ontology, target: Ontology):
        return None

    def get_embeddings_dict(self):
        return None

    def reset_embedding_dict(self):
        return None

    def get_embeddings(self, ontology):
        return None