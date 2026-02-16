import os
import json
import faiss
import ollama
from sentence_transformers import SentenceTransformer
from utils import LRUCache

INDEX_DIR = "index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4
MAX_CONTEXT_CHARS = 4000
SIM_THRESHOLD = 0.45


class AppdomeChat:
    def __init__(self, cache_size=128, embed_cache_size=512):
        print("Loading my indexing, please wait...")

        if not os.path.exists(os.path.join(INDEX_DIR, "faiss_index.bin")):
            raise RuntimeError("FAISS index not found. Run ingestion first.")

        print("Indexing loaded, loading model...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(os.path.join(INDEX_DIR, "faiss_index.bin"))

        with open(os.path.join(INDEX_DIR, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

        self.cache = LRUCache(max_size=cache_size)       # for (answer, sources)
        self.embed_cache = LRUCache(max_size=embed_cache_size)  # for query embeddings
        print("Appdome Threat-Expert here, how may I assist? Type 'exit' to quit.")

    def embed_query(self, query):
        vec = self.embed_cache.get(query)
        if vec is not None:
            return vec

        vec = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0].astype("float32").reshape(1, -1)

        self.embed_cache.set(query, vec)
        return vec

    def search(self, query, k=TOP_K):
        query_vec = self.embed_query(query)

        distances, indices = self.index.search(query_vec, k)

        context_chunks = []
        sources = set()

        total_chars = 0
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1 or score < SIM_THRESHOLD:
                continue

            item = self.metadata[idx]
            text = item["text"]

            if total_chars + len(text) > MAX_CONTEXT_CHARS:
                break

            context_chunks.append(text)
            sources.add(item["url"])
            total_chars += len(text)

        context = "\n---\n".join(context_chunks)
        return context, list(sources)

    def ask(self, query):
        # check cache first
        cached = self.cache.get(query)
        if cached:
            return cached
            
        context, sources = self.search(query)

        if not context.strip():
            return "I couldn't find relevant info.", []

        prompt = f"""
        You are an Appdome expert assistant.
        Answer ONLY from the context.
        If missing info, say you don't know.
        Be concise and technical.

        CONTEXT:
        {context}

        QUESTION:
        {query}
        """

        response = ollama.generate(
            model="llama3",
            prompt=prompt,
            options={"temperature": 0.2}
        )
        answer = response["response"].strip()

        # save in cache
        self.cache.set(query, (answer, sources))

        return answer, sources


if __name__ == "__main__":
    chat = AppdomeChat()
    while True:
        try:
            user_input = input("\n👤 Question: ").strip()
        except EOFError:
            break

        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue

        answer, links = chat.ask(user_input)

        print(f"\n🤖 AI:\n{answer}")
        if links:
            print("\n🔗 Sources:")
            for link in links:
                print(" -", link)
