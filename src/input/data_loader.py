import json
import numpy as np
import arxiv
from sentence_transformers import SentenceTransformer
from ..constants import DATA_PATH, BASE_DIR, SECONDARY_EMBED_PATH, TFIDF_EMBED_PATH, MODEL_NAME_MPNET
from ..models.tfidf import TFIDFRecommender
from scipy.sparse import save_npz, load_npz
from scipy.sparse import csr_matrix, hstack
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def load_or_fetch_papers():
    if not DATA_PATH.exists():
        papers = fetch_from_arxiv()
        save_papers(papers, "data/arxiv_cs_ai.json")
    else:
        papers = load_papers("data/arxiv_cs_ai.json")
    
    return papers

def fetch_from_arxiv(db_query="cat:cs.AI", max_results=1000):
    """Loads papers from arxiv database and create a json file containing their details"""

    client = arxiv.Client()
    search = arxiv.Search(
        query=db_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []
    for result in client.results(search):
        papers.append({
            "id": result.entry_id,
            "title": result.title,
            "abstract": result.summary,
            "published": result.published.isoformat(),
            "updated": result.updated.isoformat(),
            "authors": [a.name for a in result.authors],
            "category": result.primary_category
        })
    
    return papers


def save_papers(papers, output_path):
    """Saves given papers list to json file of output_path"""

    with open(output_path, "w") as f:
        json.dump(papers, f, indent=2)


def load_papers(papers_path):
    """Returns the research papers from given json file path"""

    with open(papers_path) as f:
        papers = json.load(f)

    return papers


def load_or_create_embeddings(papers, model_name, embeddings_path):
    embeddings_path = BASE_DIR / embeddings_path

    if not embeddings_path.exists():
        embeddings = create_embeddings(papers, model_name, embeddings_path)
    else:
        embeddings = load_embeddings(embeddings_path)

    return embeddings


def create_embeddings(papers, model_name, embeddings_path):
    """Creates embeddings (index vectors) for papers using SentenceTransformer
    and saved the numpy array to embeddings.npy in the data folder"""

    model = SentenceTransformer(model_name)
    abstracts = [p['abstract'] for p in papers]
    embeddings = model.encode(abstracts)
    embeddings = np.array(embeddings)
    np.save(embeddings_path, embeddings)

    return embeddings


def load_embeddings(embeddings_path):
    return np.load(embeddings_path)


def load_or_create_tfidf_embeddings(papers, path_prefix):
    vec_path = BASE_DIR / f"{path_prefix}_vectorizer.pkl"
    emb_path = BASE_DIR / f"{path_prefix}_embeddings.npz"

    if vec_path.exists() and emb_path.exists():
        vectorizer = joblib.load(vec_path)
        embeddings = load_npz(emb_path)
    else:
        vectorizer, embeddings = create_tfidf_embeddings(papers, vec_path, emb_path)

    return vectorizer, embeddings


def create_tfidf_embeddings(papers, vec_path, emb_path):
    abstracts = [p["abstract"] for p in papers]

    vectorizer = TfidfVectorizer(
        max_features=30000,
        stop_words="english"
    )

    embeddings = vectorizer.fit_transform(abstracts)

    joblib.dump(vectorizer, vec_path)
    save_npz(emb_path, embeddings)

    return vectorizer, embeddings


def load_or_create_hybrid_embeddings(papers, emb_path):
    emb_path = BASE_DIR / emb_path

    mpnet_emb = load_or_create_embeddings(
        papers, MODEL_NAME_MPNET, SECONDARY_EMBED_PATH
    )
    tfidf_vec, tfidf_emb = load_or_create_tfidf_embeddings(
        papers, TFIDF_EMBED_PATH
    )

    if not emb_path.exists():
        embeddings = create_hybrid_embeddings(
            mpnet_emb, tfidf_emb, emb_path
        )
    else:
        embeddings = load_npz(emb_path)

    return embeddings


def create_hybrid_embeddings(
    mpnet_emb,
    tfidf_emb,
    emb_path,
    weights=(0.7, 0.3)
):
    w_mpnet, w_tfidf = weights

    mpnet_norm = normalize(mpnet_emb)          
    tfidf_norm = normalize(tfidf_emb)         

    mpnet_sparse = csr_matrix(mpnet_norm)

    hybrid_embs = hstack([
        w_mpnet * mpnet_sparse,
        w_tfidf * tfidf_norm
    ], format="csr")

    
    save_npz(emb_path, hybrid_embs)

    return hybrid_embs