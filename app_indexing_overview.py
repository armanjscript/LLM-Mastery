from langchain_community import embeddings
import numpy as np

question="What kinds of pets do I like?"
document="My favorite pet is a cat."

embed = embeddings.OllamaEmbeddings(model="nomic-embed-text:latest")
query_result = embed.embed_query(question)
document_result = embed.embed_query(document)


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

similarity = cosine_similarity(query_result, document_result)
print("Cosine Similarity:", similarity)
