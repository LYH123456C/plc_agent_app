import os
from dotenv import load_dotenv
load_dotenv()

def load_knowledge():
    file_path = os.path.join("docs", "scl_syntax.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text
def split_text(text, chunk_size=200):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

def build_vector_store(chunks):
    vectors = []
    for chunk in chunks:
        response = client.embeddings.create(
            model="embedding-3",
            input=chunk
        )
        vectors.append(response.data[0].embedding)
    return vectors

import numpy as np

def save_vectors(vectors, filename="vectors.npy"):
    np.save(filename, np.array(vectors))
    print("向量已保存到", filename)

def load_vectors(filename="vectors.npy"):
    return np.load(filename)

def search(query, chunks, vectors, top_k=1):
    # 1. 把用户问题也转成向量
    response = client.embeddings.create(
        model="embedding-3",
        input=query
    )
    query_vec = np.array(response.data[0].embedding)

    # 2. 算问题和每块文本的余弦相似度
    vectors = np.array(vectors)
    similarities = np.dot(vectors, query_vec) / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec)
    )

    # 3. 取最相似的前 top_k 块
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def load_chunks_and_vectors():
    content = load_knowledge()
    chunks = split_text(content)
    vectors = load_vectors()
    return chunks, vectors


if __name__ == "__main__":
    content = load_knowledge()
    chunks = split_text(content)
    vectors = build_vector_store(chunks)
    save_vectors(vectors)

    # 测试检索
    query = "定时器怎么用"
    results = search(query, chunks, vectors)
    print("\n查询：", query)
    print("最相关的文本块：")
    print(results[0])
    