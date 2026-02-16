import os
import json
import tempfile
import shutil
import pytest
import faiss
import numpy as np
from pathlib import Path


@pytest.fixture
def temp_data_dir():
    """Create a temporary data/raw directory."""
    temp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(temp_dir, "data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    yield data_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_index_dir():
    """Create a temporary index directory."""
    temp_dir = tempfile.mkdtemp()
    index_dir = os.path.join(temp_dir, "index")
    os.makedirs(index_dir, exist_ok=True)
    yield index_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_json_file(temp_data_dir):
    """Create a sample JSON file for testing crawler output."""
    sample_data = {
        "url": "https://www.appdome.com/how-to/test-article",
        "title": "Test Article Title",
        "content": "# Test Article\n\nThis is a test article with some content.\n\n## Section 1\n\nMore content here."
    }
    filepath = os.path.join(temp_data_dir, "test-article.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=4)
    return filepath


@pytest.fixture
def multiple_json_files(temp_data_dir):
    """Create multiple sample JSON files for testing ingestion."""
    files = []
    for i in range(3):
        sample_data = {
            "url": f"https://www.appdome.com/how-to/article-{i}",
            "title": f"Article {i} Title",
            "content": f"# Article {i}\n\nThis is article {i} with some content. " * 50  # Make it long enough to chunk
        }
        filepath = os.path.join(temp_data_dir, f"article-{i}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=4)
        files.append(filepath)
    return files


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing chat/search functionality."""
    return [
        {
            "url": "https://www.appdome.com/how-to/article-1",
            "text": "This is a sample text chunk about mobile app security."
        },
        {
            "url": "https://www.appdome.com/how-to/article-2",
            "text": "Another chunk discussing threat prevention techniques."
        },
        {
            "url": "https://www.appdome.com/how-to/article-1",
            "text": "More content about security best practices."
        }
    ]


@pytest.fixture
def mock_faiss_index(sample_metadata):
    """Create a mock FAISS index for testing."""
    # Create a small index with 3 vectors of dimension 384 (MiniLM-L6-v2)
    dimension = 384
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Generate random normalized embeddings
    np.random.seed(42)  # For reproducibility
    embeddings = np.random.rand(len(sample_metadata), dimension).astype('float32')
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    return index, embeddings


@pytest.fixture
def faiss_index_file(temp_index_dir, mock_faiss_index):
    """Create a FAISS index file for testing."""
    index, _ = mock_faiss_index
    index_path = os.path.join(temp_index_dir, "faiss_index.bin")
    faiss.write_index(index, index_path)
    return index_path


@pytest.fixture
def metadata_file(temp_index_dir, sample_metadata):
    """Create a metadata.json file for testing."""
    metadata_path = os.path.join(temp_index_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(sample_metadata, f, indent=4)
    return metadata_path


@pytest.fixture
def sample_html():
    """Sample HTML content for testing crawler parsing."""
    return """
    <html>
    <head><title>Test Page</title></head>
    <body>
        <nav>Navigation</nav>
        <header>Header</header>
        <main>
            <h1>Main Content</h1>
            <p>This is the main content of the page.</p>
            <a href="/how-to/article-1">Link 1</a>
            <a href="/how-to/article-2">Link 2</a>
        </main>
        <footer>Footer</footer>
        <script>console.log('test');</script>
    </body>
    </html>
    """


@pytest.fixture
def sample_html_with_links():
    """Sample HTML with multiple links for testing link extraction."""
    return """
    <html>
    <head><title>Hub Page</title></head>
    <body>
        <a href="/how-to/article-1">Article 1</a>
        <a href="/how-to/article-2">Article 2</a>
        <a href="https://www.appdome.com/how-to/article-3">Article 3</a>
        <a href="/other-page">Other Page</a>
        <a href="https://external.com/page">External</a>
    </body>
    </html>
    """
