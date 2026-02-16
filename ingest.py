import os
import json
import faiss
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

torch.set_num_threads(os.cpu_count()) # allow torch to use all 
os.environ["TOKENIZERS_PARALLELISM"] = "true" # enable tokenizing parallelism

INPUT_DIR = "data/raw"
INDEX_DIR = "index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
DEFAULT_BATCH_SIZE = 128


class Ingestor:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.index = None
        self.metadata = []

    def process_file(self, filepath):
        """Yield chunks from a single JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = self.text_splitter.split_text(data["content"])
        for chunk in chunks:
            yield chunk, {"url": data["url"], "text": chunk}

    def add_embeddings(self, texts, metas):
        """Embed batch and add to FAISS."""
        embeddings = self.model.encode(
            texts,
            batch_size=DEFAULT_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        embeddings = np.array(embeddings).astype("float32")

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)
        self.metadata.extend(metas)

    def save_progress(self):
        if self.index is None:
            print("Nothing to save yet.")
            return

        print("\nSaving partial index...")
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(self.index, os.path.join(INDEX_DIR, "faiss_index.bin"))

        with open(os.path.join(INDEX_DIR, "metadata.json"), "w") as f:
            json.dump(self.metadata, f)

        print(f"Saved {len(self.metadata)} chunks.")

    def run(self):
        files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
        print(f"Found {len(files)} files")

        buffer_texts = []
        buffer_meta = []

        try:
            for filename in tqdm(files, desc="Files"):
                path = os.path.join(INPUT_DIR, filename)

                for chunk, meta in self.process_file(path):
                    buffer_texts.append(chunk)
                    buffer_meta.append(meta)

                    if len(buffer_texts) >= DEFAULT_BATCH_SIZE * 4:
                        self.add_embeddings(buffer_texts, buffer_meta)
                        buffer_texts.clear()
                        buffer_meta.clear()

                # intermediate progress to avoid complete loss on errors
                if len(self.metadata) != 0 and len(self.metadata) % 1000 == 0:
                    self.save_progress()

            # last batch
            if buffer_texts:
                self.add_embeddings(buffer_texts, buffer_meta)      

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
      
        # Save index
        self.save_progress()
        print(f"Done. Indexed {len(self.metadata)} chunks.")


if __name__ == "__main__":
    Ingestor().run()
