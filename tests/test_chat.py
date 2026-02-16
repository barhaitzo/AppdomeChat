import os
import json
import pytest
import faiss
import numpy as np
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

from chat import AppdomeChat, INDEX_DIR, MODEL_NAME, TOP_K, MAX_CONTEXT_CHARS, SIM_THRESHOLD


class TestAppdomeChatUnit:
    """Unit tests for AppdomeChat class."""

    def test_embed_query_caching(self, faiss_index_file, metadata_file, temp_index_dir):
        """Test query embedding and caching."""
        with patch('chat.INDEX_DIR', temp_index_dir), \
             patch('chat.SentenceTransformer') as mock_model_class:
            
            # Mock model
            mock_model = MagicMock()
            mock_embedding = np.random.rand(384).astype('float32')
            mock_model.encode.return_value = [mock_embedding]
            mock_model_class.return_value = mock_model
            
            chat = AppdomeChat()
            
            # First call should use model
            vec1 = chat.embed_query("test query")
            assert mock_model.encode.called
            
            # Reset mock call count
            mock_model.encode.reset_mock()
            
            # Second call should use cache
            vec2 = chat.embed_query("test query")
            assert not mock_model.encode.called  # Should not call model again
            assert np.array_equal(vec1, vec2)

    def test_embed_query_new_query(self, faiss_index_file, metadata_file, temp_index_dir):
        """Test embedding for new queries."""
        with patch('chat.INDEX_DIR', temp_index_dir), \
             patch('chat.SentenceTransformer') as mock_model_class:
            
            mock_model = MagicMock()
            mock_embedding1 = np.random.rand(384).astype('float32')
            mock_embedding2 = np.random.rand(384).astype('float32')
            mock_model.encode.side_effect = [[mock_embedding1], [mock_embedding2]]
            mock_model_class.return_value = mock_model
            
            chat = AppdomeChat()
            
            vec1 = chat.embed_query("query 1")
            vec2 = chat.embed_query("query 2")
            
            # Should call model twice for different queries
            assert mock_model.encode.call_count == 2
            assert not np.array_equal(vec1, vec2)

    def test_search_with_mocked_index(self, temp_index_dir, sample_metadata):
        """Test FAISS search with mocked index."""
        # Create a real index for testing
        dimension = 384
        index = faiss.IndexFlatIP(dimension)
        embeddings = np.random.rand(len(sample_metadata), dimension).astype('float32')
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        index_path = os.path.join(temp_index_dir, "faiss_index.bin")
        faiss.write_index(index, index_path)
        
        metadata_path = os.path.join(temp_index_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(sample_metadata, f)
        
        with patch('chat.INDEX_DIR', temp_index_dir), \
             patch('chat.SentenceTransformer') as mock_model_class:
            
            mock_model = MagicMock()
            query_embedding = np.random.rand(384).astype('float32')
            mock_model.encode.return_value = [query_embedding]
            mock_model_class.return_value = mock_model
            
            chat = AppdomeChat()
            context, sources = chat.search("test query", k=2)
            
            assert isinstance(context, str)
            assert isinstance(sources, list)

    def test_search_similarity_threshold(self, temp_index_dir, sample_metadata):
        """Test similarity threshold filtering."""
        dimension = 384
        index = faiss.IndexFlatIP(dimension)
        
        # Create embeddings with known distances
        embeddings = np.random.rand(len(sample_metadata), dimension).astype('float32')
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        index_path = os.path.join(temp_index_dir, "faiss_index.bin")
        faiss.write_index(index, index_path)
        
        metadata_path = os.path.join(temp_index_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(sample_metadata, f)
        
        with patch('chat.INDEX_DIR', temp_index_dir), \
             patch('chat.SentenceTransformer') as mock_model_class, \
             patch('chat.SIM_THRESHOLD', 0.9):  # High threshold
            
            mock_model = MagicMock()
            query_embedding = np.random.rand(384).astype('float32')
            mock_model.encode.return_value = [query_embedding]
            mock_model_class.return_value = mock_model
            
            chat = AppdomeChat()
            context, sources = chat.search("test query")
            
            # With high threshold, might get fewer results
            assert isinstance(context, str)

    def test_search_context_character_limits(self, temp_index_dir, sample_metadata):
        """Test context character limits."""
        dimension = 384
        index = faiss.IndexFlatIP(dimension)
        embeddings = np.random.rand(len(sample_metadata), dimension).astype('float32')
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        index_path = os.path.join(temp_index_dir, "faiss_index.bin")
        faiss.write_index(index, index_path)
        
        # Create metadata with very long text
        long_metadata = [
            {"url": f"url{i}", "text": "x" * 2000}  # 2000 chars each
            for i in range(5)
        ]
        metadata_path = os.path.join(temp_index_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(long_metadata, f)
        
        with patch('chat.INDEX_DIR', temp_index_dir), \
             patch('chat.SentenceTransformer') as mock_model_class, \
             patch('chat.MAX_CONTEXT_CHARS', 3000):  # Lower limit for testing
            
            mock_model = MagicMock()
            query_embedding = np.random.rand(384).astype('float32')
            mock_model.encode.return_value = [query_embedding]
            mock_model_class.return_value = mock_model
            
            chat = AppdomeChat()
            context, sources = chat.search("test query", k=5)
            
            # Context should be limited
            assert len(context) <= 3000

    def test_search_source_extraction(self, temp_index_dir, sample_metadata):
        """Test source extraction from search results."""
        dimension = 384
        index = faiss.IndexFlatIP(dimension)
        embeddings = np.random.rand(len(sample_metadata), dimension).astype('float32')
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        index_path = os.path.join(temp_index_dir, "faiss_index.bin")
        faiss.write_index(index, index_path)
        
        metadata_path = os.path.join(temp_index_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(sample_metadata, f)
        
        with patch('chat.INDEX_DIR', temp_index_dir), \
             patch('chat.SentenceTransformer') as mock_model_class:
            
            mock_model = MagicMock()
            query_embedding = np.random.rand(384).astype('float32')
            mock_model.encode.return_value = [query_embedding]
            mock_model_class.return_value = mock_model
            
            chat = AppdomeChat()
            context, sources = chat.search("test query")
            
            # Sources should be unique URLs
            assert isinstance(sources, list)
            assert all(isinstance(s, str) for s in sources)
            # Should have unique sources
            assert len(sources) == len(set(sources))

    def test_ask_with_mocked_ollama(self, faiss_index_file, metadata_file, temp_index_dir):
        """Test prompt construction and Ollama integration (mocked)."""
        with patch('chat.INDEX_DIR', temp_index_dir), \
             patch('chat.SentenceTransformer') as mock_model_class, \
             patch('chat.ollama.generate') as mock_ollama:
            
            mock_model = MagicMock()
            query_embedding = np.random.rand(384).astype('float32')
            mock_model.encode.return_value = [query_embedding]
            mock_model_class.return_value = mock_model
            
            mock_ollama.return_value = {"response": "This is a test response."}
            
            chat = AppdomeChat()
            answer, sources = chat.ask("What is mobile app security?")
            
            assert isinstance(answer, str)
            assert answer == "This is a test response."
            assert isinstance(sources, list)
            
            # Verify ollama.generate was called
            mock_ollama.assert_called_once()
            call_args = mock_ollama.call_args
            assert "model" in call_args.kwargs
            assert "prompt" in call_args.kwargs
            assert "What is mobile app security?" in call_args.kwargs["prompt"]

    def test_ask_no_context(self, faiss_index_file, metadata_file, temp_index_dir):
        """Test ask when no relevant context is found."""
        with patch('chat.INDEX_DIR', temp_index_dir), \
             patch('chat.SentenceTransformer') as mock_model_class, \
             patch('chat.ollama.generate') as mock_ollama:
            
            mock_model = MagicMock()
            query_embedding = np.random.rand(384).astype('float32')
            mock_model.encode.return_value = [query_embedding]
            mock_model_class.return_value = mock_model
            
            chat = AppdomeChat()
            
            # Mock search to return empty context
            chat.search = MagicMock(return_value=("", []))
            
            answer, sources = chat.ask("unrelated query")
            
            assert answer == "I couldn't find relevant info."
            assert sources == []
            # Should not call ollama when no context
            mock_ollama.assert_not_called()


class TestAppdomeChatIntegration:
    """Integration tests for AppdomeChat class."""

    @pytest.mark.slow
    def test_chat_end_to_end(self, faiss_index_file, metadata_file, temp_index_dir):
        """Full chat flow with real FAISS index (from fixtures)."""
        with patch('chat.INDEX_DIR', temp_index_dir), \
             patch('chat.ollama.generate') as mock_ollama:
            
            mock_ollama.return_value = {"response": "Based on the context, mobile app security involves..."}
            
            chat = AppdomeChat()
            
            # Test search
            context, sources = chat.search("mobile app security", k=2)
            assert isinstance(context, str)
            assert isinstance(sources, list)
            
            # Test ask
            answer, answer_sources = chat.ask("What is mobile app security?")
            assert isinstance(answer, str)
            assert len(answer) > 0
            assert isinstance(answer_sources, list)

    def test_search_result_ranking(self, temp_index_dir, sample_metadata):
        """Test search result ranking."""
        dimension = 384
        index = faiss.IndexFlatIP(dimension)
        
        # Create embeddings
        embeddings = np.random.rand(len(sample_metadata), dimension).astype('float32')
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        index_path = os.path.join(temp_index_dir, "faiss_index.bin")
        faiss.write_index(index, index_path)
        
        metadata_path = os.path.join(temp_index_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(sample_metadata, f)
        
        with patch('chat.INDEX_DIR', temp_index_dir), \
             patch('chat.SentenceTransformer') as mock_model_class:
            
            mock_model = MagicMock()
            query_embedding = np.random.rand(384).astype('float32')
            mock_model.encode.return_value = [query_embedding]
            mock_model_class.return_value = mock_model
            
            chat = AppdomeChat()
            context, sources = chat.search("test query", k=TOP_K)
            
            # Results should be returned
            assert isinstance(context, str)
            assert len(sources) <= TOP_K

    def test_initialization_missing_index(self, temp_index_dir):
        """Test initialization when index is missing."""
        # Ensure index doesn't exist
        index_path = os.path.join(temp_index_dir, "faiss_index.bin")
        if os.path.exists(index_path):
            os.remove(index_path)
        
        with patch('chat.INDEX_DIR', temp_index_dir):
            with pytest.raises(RuntimeError, match="FAISS index not found"):
                AppdomeChat()

    def test_cache_functionality(self, faiss_index_file, metadata_file, temp_index_dir):
        """Test that query cache works correctly."""
        with patch('chat.INDEX_DIR', temp_index_dir), \
             patch('chat.SentenceTransformer') as mock_model_class:
            
            mock_model = MagicMock()
            query_embedding = np.random.rand(384).astype('float32')
            mock_model.encode.return_value = [query_embedding]
            mock_model_class.return_value = mock_model
            
            chat = AppdomeChat()
            
            # First search
            chat.search("test query")
            initial_call_count = mock_model.encode.call_count
            
            # Second search with same query
            chat.search("test query")
            
            # Should use cache, so call count should be same
            assert mock_model.encode.call_count == initial_call_count
