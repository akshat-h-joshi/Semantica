from .sbert_mini import SBERTMini
from .sbert_mpnet import SBERTMpnet
from .tfidf import TFIDFRecommender  

def get_model(name):
    models = {
        "mini": SBERTMini,
        "mpnet": SBERTMpnet,
        "tfidf": TFIDFRecommender,
    }
    return models[name]()