import os
import json
import pytest
import faiss
import numpy as np
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

from ingest import Ingestor, INPUT_DIR, INDEX_DIR


class TestIngestorUnit:
    """Unit tests for Ingestor class."""

    def test_process_file(self, temp_data_dir, sample_json_file):
        """Test JSON file parsing and chunking."""
        ingestor = Ingestor()
        
        # Mock the text splitter to return predictable chunks
        mock_chunks = ["chunk 1", "chunk 2", "chunk 3"]
        ingestor.text_splitter.split_text = MagicMock(return_value=mock_chunks)
        
        chunks_and_metas = list(ingestor.process_file(sample_json_file))
        
        assert len(chunks_and_metas) == 3
        for chunk, meta in chunks_and_metas:
            assert chunk in mock_chunks
            assert 'url' in meta
            assert 'text' in meta
            assert meta['text'] == chunk

    def test_add_embeddings_creates_index(self):
        """Test embedding generation and FAISS index creation."""
        ingestor = Ingestor()
        
        # Mock the model
        mock_embeddings = np.random.rand(3, 384).astype('float32')
        ingestor.model.encode = MagicMock(return_value=mock_embeddings)
        
        texts = ["text1", "text2", "text3"]
        metas = [{"url": f"url{i}", "text": f"text{i}"} for i in range(3)]
        
        ingestor.add_embeddings(texts, metas)
        
        assert ingestor.index is not None
        assert ingestor.index.ntotal == 3
        assert len(ingestor.metadata) == 3

    def test_add_embeddings_adds_to_existing_index(self):
        """Test adding embeddings to existing index."""
        ingestor = Ingestor()
        
        # Mock the model
        mock_embeddings1 = np.random.rand(2, 384).astype('float32')
        mock_embeddings2 = np.random.rand(3, 384).astype('float32')
        ingestor.model.encode = MagicMock(side_effect=[mock_embeddings1, mock_embeddings2])
        
        # First batch
        texts1 = ["text1", "text2"]
        metas1 = [{"url": f"url{i}", "text": f"text{i}"} for i in range(2)]
        ingestor.add_embeddings(texts1, metas1)
        
        # Second batch
        texts2 = ["text3", "text4", "text5"]
        metas2 = [{"url": f"url{i}", "text": f"text{i}"} for i in range(2, 5)]
        ingestor.add_embeddings(texts2, metas2)
        
        assert ingestor.index.ntotal == 5
        assert len(ingestor.metadata) == 5

    def test_save_progress_no_index(self, temp_index_dir):
        """Test save_progress when no index exists."""
        ingestor = Ingestor()
        ingestor.index = None
        
        # Should not raise error, just return early
        ingestor.save_progress()
        
        # No files should be created
        assert not os.path.exists(os.path.join(temp_index_dir, "faiss_index.bin"))

    def test_save_progress_saves_files(self, temp_index_dir):
        """Test index and metadata saving."""
        ingestor = Ingestor()
        
        # Create a small index
        mock_embeddings = np.random.rand(2, 384).astype('float32')
        ingestor.model.encode = MagicMock(return_value=mock_embeddings)
        
        texts = ["text1", "text2"]
        metas = [{"url": "url1", "text": "text1"}, {"url": "url2", "text": "text2"}]
        ingestor.add_embeddings(texts, metas)
        
        # Override INDEX_DIR for testing
        original_index_dir = INDEX_DIR
        with patch('ingest.INDEX_DIR', temp_index_dir):
            ingestor.save_progress()
        
        # Check files were created
        index_path = os.path.join(temp_index_dir, "faiss_index.bin")
        metadata_path = os.path.join(temp_index_dir, "metadata.json")
        
        assert os.path.exists(index_path)
        assert os.path.exists(metadata_path)
        
        # Verify index can be loaded
        loaded_index = faiss.read_index(index_path)
        assert loaded_index.ntotal == 2
        
        # Verify metadata
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        assert len(loaded_metadata) == 2
        assert loaded_metadata[0]['url'] == "url1"

    def test_batch_processing_logic(self):
        """Test batch processing logic."""
        ingestor = Ingestor()
        
        # Mock model
        mock_embeddings = np.random.rand(1, 384).astype('float32')
        ingestor.model.encode = MagicMock(return_value=mock_embeddings)
        
        # Simulate batch processing
        buffer_texts = []
        buffer_meta = []
        batch_size = 128 * 4  # DEFAULT_BATCH_SIZE * 4
        
        # Add items up to batch size
        for i in range(batch_size):
            buffer_texts.append(f"text{i}")
            buffer_meta.append({"url": f"url{i}", "text": f"text{i}"})
        
        # Process batch
        ingestor.add_embeddings(buffer_texts, buffer_meta)
        
        assert ingestor.index.ntotal == batch_size
        assert len(ingestor.metadata) == batch_size

    def test_text_splitter_configuration(self):
        """Test text splitter configuration."""
        ingestor = Ingestor()
        
        assert ingestor.text_splitter is not None
        assert hasattr(ingestor.text_splitter, 'chunk_size')
        assert hasattr(ingestor.text_splitter, 'chunk_overlap')


class TestIngestorIntegration:
    """Integration tests for Ingestor class."""

    @pytest.mark.slow
    def test_ingestor_end_to_end(self, temp_data_dir, temp_index_dir, multiple_json_files):
        """Full ingestion with real SentenceTransformer (small model)."""
        # This test uses the real model, so it's marked as slow
        ingestor = Ingestor()
        
        # Override directories
        original_input_dir = INPUT_DIR
        original_index_dir = INDEX_DIR
        
        with patch('ingest.INPUT_DIR', temp_data_dir), \
             patch('ingest.INDEX_DIR', temp_index_dir):
            
            # Mock os.listdir to return our test files
            with patch('os.listdir', return_value=[os.path.basename(f) for f in multiple_json_files]):
                # Run ingestion
                ingestor.run()
        
        # Verify index was created
        index_path = os.path.join(temp_index_dir, "faiss_index.bin")
        metadata_path = os.path.join(temp_index_dir, "metadata.json")
        
        assert os.path.exists(index_path)
        assert os.path.exists(metadata_path)
        
        # Load and verify
        index = faiss.read_index(index_path)
        assert index.ntotal > 0
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        assert len(metadata) > 0
        assert all('url' in m for m in metadata)
        assert all('text' in m for m in metadata)

    def test_faiss_index_structure(self):
        """Test FAISS index creation and metadata structure."""
        ingestor = Ingestor()
        
        # Use real model but with small data
        texts = ["This is a short text.", "Another short text."]
        metas = [
            {"url": "https://example.com/1", "text": "This is a short text."},
            {"url": "https://example.com/2", "text": "Another short text."}
        ]
        
        ingestor.add_embeddings(texts, metas)
        
        assert ingestor.index is not None
        assert ingestor.index.ntotal == 2
        assert isinstance(ingestor.index, faiss.IndexFlatIP)
        
        # Verify metadata structure
        assert len(ingestor.metadata) == 2
        assert all('url' in m for m in ingestor.metadata)
        assert all('text' in m for m in ingestor.metadata)

    def test_batch_size_handling(self):
        """Test batch size handling."""
        ingestor = Ingestor()
        
        # Create many small texts
        texts = [f"Text {i}" for i in range(10)]
        metas = [{"url": f"url{i}", "text": f"Text {i}"} for i in range(10)]
        
        ingestor.add_embeddings(texts, metas)
        
        assert ingestor.index.ntotal == 10
        assert len(ingestor.metadata) == 10

    def test_progress_saving_at_intervals(self, temp_index_dir):
        """Test progress saving at intervals."""
        ingestor = Ingestor()
        
        # Mock model
        mock_embeddings = np.random.rand(1, 384).astype('float32')
        ingestor.model.encode = MagicMock(return_value=mock_embeddings)
        
        # Add 1000 items to trigger save
        texts = [f"text{i}" for i in range(1000)]
        metas = [{"url": f"url{i}", "text": f"text{i}"} for i in range(1000)]
        
        # Process in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metas = metas[i:i+batch_size]
            ingestor.add_embeddings(batch_texts, batch_metas)
            
            # Simulate save at 1000 interval
            if len(ingestor.metadata) >= 1000:
                with patch('ingest.INDEX_DIR', temp_index_dir):
                    ingestor.save_progress()
        
        # Verify save happened
        index_path = os.path.join(temp_index_dir, "faiss_index.bin")
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            assert index.ntotal >= 1000

    def test_empty_input_directory(self, temp_data_dir):
        """Test handling of empty input directory."""
        ingestor = Ingestor()
        
        # Ensure directory is empty
        for f in os.listdir(temp_data_dir):
            os.remove(os.path.join(temp_data_dir, f))
        
        with patch('ingest.INPUT_DIR', temp_data_dir), \
             patch('os.listdir', return_value=[]):
            ingestor.run()
        
        # Should handle gracefully
        assert ingestor.index is None or ingestor.index.ntotal == 0

    def test_file_processing_yields_chunks(self, sample_json_file):
        """Test that process_file correctly yields chunks."""
        ingestor = Ingestor()
        
        chunks = list(ingestor.process_file(sample_json_file))
        
        assert len(chunks) > 0
        for chunk, meta in chunks:
            assert isinstance(chunk, str)
            assert isinstance(meta, dict)
            assert 'url' in meta
            assert 'text' in meta
