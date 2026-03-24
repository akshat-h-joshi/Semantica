# Semantica
## A Semantic Research Paper Recommender & Evaulation Framework
Semantica is a research paper recommender that leverages sentence-transformer models and the K-Nearest Neighbours (KNN) algorithm to identify papers most relevant to user queries. Alongisde this, it serves as an evaluation platform to compare the performance of different embedding techniques, such as Sentence-BERT, TF-IDF, and hybrid models using industry-standard metrics.

## Installation
```
git clone https://github.com/akshat-h-joshi/Semantica
cd Semantica

# Backend setup
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run backend
uvicorn backend.api.app:app --reload

# Frontend setup (open new terminal)
cd frontend
npm install
npm run dev

```
Then open the localhost link provided by the frontend

## Tech Stack
### Backend
- Python
- FastAPI
- Sentence Transformers (SBERT)
- FAISS (vector similarity search)

### Frontend
- React (Vite)
- Framer Motion

### Machine Learning / Algorithms
- K-Nearest Neighbours (KNN)
- TF-IDF
- Hybrid embedding models

## Features 
- Features acronym expansion in queries to accurately identify important concept (e.g. LLM expanded to Large Language Models for paper search)
- Allows the selection of various models, allowing user freedom to utilise/evaluate various models
- Highlights keywords in the abstract of the recommended paper
- Displays industry-standard evaluation metrics for unsupervised models, including categority purity and mean reciprocal rank
- Compares contributions between paper title and abstract for recommendation

## Demo

Users can type their queries in the home screen:
<img width="1919" height="900" alt="image" src="https://github.com/user-attachments/assets/95c568e4-d2fc-42f4-bd21-dac00840fb85" />

Papers are recommended based on the query, expanding any acronyms and highlighting the keywords that have high semantic relevance to the query
<img width="1912" height="894" alt="image" src="https://github.com/user-attachments/assets/84cae827-4ef9-4c53-a24d-2c5414762e43" />

Users can click on the "Why recommend" button to see the weights of the paper's title vs its abstract in the recommendation, as well as any expanded terms
<img width="1919" height="897" alt="image" src="https://github.com/user-attachments/assets/3d03d0c5-efff-4ab3-b787-250e58e663eb" />

Models can be changed with the selector for comparison
<img width="1919" height="895" alt="image" src="https://github.com/user-attachments/assets/f6efef6f-2b52-413e-b351-23fc60c4507a" />

Different models have different evaluation sections. For instance, the hybrid evaluation modal evaluates the performance of both of its composite models separately.
<img width="1919" height="887" alt="image" src="https://github.com/user-attachments/assets/71510416-b0fc-4a16-9693-78e409966486" />

## Data Source
Research papers are fetched from arxiv and indexed using fAISS index for efficient semantic retrieval


