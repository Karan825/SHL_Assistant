import re
import json
import pickle
import faiss
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CATALOG_PATH = Path("data/catalog.json")

FAISS_PATH    = Path("data/catalog.faiss")
BM25_PATH     = Path("data/bm25.pkl")
DOCS_PATH     = Path("data/docs.pkl")
LOOKUP_PATH   = Path("data/catalog_lookup.pkl")
MODEL_INFO_PATH = Path("data/model_name.pkl")

MODEL_NAME = "BAAI/bge-small-en-v1.5"


# --------------------------------------------------
# Load catalog
# --------------------------------------------------

with open(CATALOG_PATH, 'r', encoding='utf-8') as f:
    catalog = json.load(f)

print(f"Loaded {len(catalog)} assessments from catalog.")

# --------------------------------------------------
# Build embedding document
# --------------------------------------------------

def build_document(item: dict) -> str:

    keys = ', '.join(item.get('keys', []))
    return (
        f"Assessment Name: {item.get('name', '')}\n"
        f"Description: {item.get('description', '')}\n"
        f"Test Type: {keys}\n"
        f"Suitable Job Levels: {', '.join(item.get('job_levels', []))}\n"
        f"Languages: {', '.join(item.get('languages', []))}\n"
        f"Duration: {item.get('duration', '')}\n"
        f"Remote Testing: {item.get('remote', '')}\n"
        f"Adaptive Test: {item.get('adaptive', '')}"
    )

documents = [build_document(x) for x in catalog]

# --------------------------------------------------
# Tokenizer for BM25
# --------------------------------------------------

def tokenize(text: str) -> list[str]:
    text = text.lower().strip()
    return re.findall(r"[a-zA-Z0-9\+\#\.]+", text)


tokenized_docs = [tokenize(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# --------------------------------------------------
# Embeddings
# --------------------------------------------------

print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME, local_files_only=True)

embeddings = model.encode(
    documents,
    normalize_embeddings=True,  
    convert_to_numpy=True,
    show_progress_bar=True,
    batch_size=64                 
)
embeddings = embeddings.astype(np.float32)


# --------------------------------------------------
# FAISS Index
# --------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)  
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} vectors of dimension {dimension}.")


# --------------------------------------------------
# Save everything
# --------------------------------------------------

faiss.write_index(index, str(FAISS_PATH))
print(f"Saved FAISS index -> {FAISS_PATH}")

with open(BM25_PATH, "wb") as f:
    pickle.dump(bm25, f)
print(f"Saved BM25 index  -> {BM25_PATH}")

with open(DOCS_PATH, "wb") as f:
    pickle.dump(documents, f)
print(f"Saved docs        -> {DOCS_PATH}")

with open(LOOKUP_PATH, "wb") as f:
    pickle.dump(catalog, f)
print(f"Saved lookup      -> {LOOKUP_PATH}")

with open(MODEL_INFO_PATH, "wb") as f:
    pickle.dump(MODEL_NAME, f)
print(f"Saved model name  -> {MODEL_INFO_PATH}")

print("=" * 50)
print(f"Indexed {len(catalog)} assessments")
print(f"Embedding model : {MODEL_NAME}")
print(f"Embedding dim   : {dimension}")
print("=" * 50)
