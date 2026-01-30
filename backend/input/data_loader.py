import json
import numpy as np
import arxiv
from sentence_transformers import SentenceTransformer
from ..utils.constants import DATA_PATH, BASE_DIR
from scipy.sparse import save_npz, load_npz
from scipy.sparse import csr_matrix, hstack
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

def load_or_fetch_papers():
    FULL_PATH = BASE_DIR / DATA_PATH
    
    if not FULL_PATH.exists():
        papers = fetch_from_arxiv()
        save_papers(papers, DATA_PATH)
    else:
        papers = load_papers(DATA_PATH)
    
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
            "category": result.primary_category,
            "link": result.pdf_url,
            "arxiv_link": result.entry_id
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


def load_or_create_embeddings(papers, model_name, embeddings_dir, field):
    embeddings_dir = BASE_DIR / embeddings_dir
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    path = embeddings_dir / f"{field}.npy"

    if not path.exists():
        embeddings = create_embeddings(papers, model_name, path, field)
    else:
        embeddings = load_embeddings(path)

    return embeddings


def create_embeddings(papers, model_name, embeddings_path, field):
    """Creates embeddings (index vectors) for papers using SentenceTransformer
    and saved the numpy array to embeddings.npy in the data folder"""

    model = SentenceTransformer(model_name)

    texts = [p[field] for p in papers]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    np.save(embeddings_path, embeddings)

    return embeddings


def load_embeddings(embeddings_path):
    return np.load(embeddings_path)


def load_or_create_tfidf_embeddings(papers, path_prefix, field):
    tfidf_dir = BASE_DIR / path_prefix
    tfidf_dir.mkdir(parents=True, exist_ok=True)

    vec_path = tfidf_dir / f"{field}_vectorizer.pkl"
    emb_path = tfidf_dir / f"{field}_embeddings.npz"

    if vec_path.exists() and emb_path.exists():
        vectorizer = joblib.load(vec_path)
        embeddings = load_npz(emb_path)
    else:
        vectorizer, embeddings = create_tfidf_embeddings(papers, vec_path, emb_path, field)

    return vectorizer, embeddings


def create_tfidf_embeddings(papers, vec_path, emb_path, field):

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english"
    )

    embeddings = vectorizer.fit_transform(
        [p["title"] for p in papers] if field == "title" else [p["abstract"] for p in papers]
    )

    joblib.dump(vectorizer, vec_path)
    save_npz(emb_path, embeddings)

    return vectorizer, embeddings
