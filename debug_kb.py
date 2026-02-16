import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "knowledge_db"
COLLECTION = "satellites"

embedder = SentenceTransformer("intfloat/multilingual-e5-small")

client = chromadb.PersistentClient(path=DB_PATH)
col = client.get_collection(COLLECTION)

print("COUNT =", col.count())

q = "Что такое спутник Метеор-М?"
q_emb = embedder.encode([f"query: {q}"], normalize_embeddings=True)

res = col.query(query_embeddings=q_emb, n_results=5, include=["documents", "distances", "metadatas"])

print("\nTOP RESULTS:")
for i, (doc, dist, meta) in enumerate(zip(res["documents"][0], res["distances"][0], res["metadatas"][0])):
    print("\n----", i, "dist=", dist, "meta=", meta)
    print(doc)