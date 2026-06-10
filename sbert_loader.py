# sbert_loader.py

from sentence_transformers import SentenceTransformer

print("✓ Loading SBERT model...")

sbert_model = SentenceTransformer(
    'firqaaa/indo-sentence-bert-base'
)

print("✓ SBERT loaded successfully")