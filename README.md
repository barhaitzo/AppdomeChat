# Appdome RAG Project

A Retrieval-Augmented Generation (RAG) system for querying Appdome documentation. This project crawls Appdome's how-to articles, creates vector embeddings, and provides an interactive Q&A interface powered by Ollama.

## Table of Contents

- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
  - [Running the Crawler](#running-the-crawler)
  - [Running the Ingestor](#running-the-ingestor)
  - [Running the Chat Interface](#running-the-chat-interface)
- [Testing](#testing)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## Project Overview

This RAG system consists of three main components:

1. **Crawler** (`crawler.py`): Asynchronously crawls Appdome's how-to documentation, extracts content, and saves it as JSON files.

2. **Ingestor** (`ingest.py`): Processes the crawled JSON files, splits content into chunks, generates embeddings using sentence transformers, and creates a FAISS vector index for efficient similarity search.

3. **Chat** (`chat.py`): Provides an interactive Q&A interface that uses the vector index to retrieve relevant context and generates answers using Ollama's LLM.

## Prerequisites

- **Python 3.10+**: Required for async/await support and modern Python features
- **Virtual Environment**: Recommended for dependency isolation
- **Ollama**: Required for running the LLM locally
  - Download from: https://ollama.ai
  - Install the `llama3` model: `ollama pull llama3`
- **System Requirements**:
  - Minimum 4GB RAM (8GB+ recommended)
  - CPU with multiple cores (for parallel processing)
  - ~2GB disk space for models and indices

## Environment Setup

### 1. Create Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Check Python version
python --version  # Should be 3.8+

# Verify key packages
python -c "import httpx, faiss, sentence_transformers, ollama; print('All packages installed!')"
```

## Installation

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd appdome_rag_project

# Or download and extract the project files
```

### Step 2: Set Up Ollama

1. **Install Ollama**:
   - Visit https://ollama.ai and download for your platform
   - Follow installation instructions

2. **Pull the LLM Model**:
   ```bash
   ollama pull llama3
   ```

3. **Verify Ollama is Running**:
   ```bash
   ollama list  # Should show llama3 model
   ```

### Step 3: Install Python Dependencies

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

This will install:
- `httpx` - Async HTTP client for web crawling
- `beautifulsoup4` - HTML parsing
- `markdownify` - HTML to Markdown conversion
- `python-slugify` - URL slug generation
- `faiss-cpu` - Vector similarity search
- `numpy` - Numerical operations
- `sentence-transformers` - Embedding generation
- `langchain_text_splitters` - Text chunking
- `ollama` - Ollama API client
- `torch` - PyTorch for model operations
- `tqdm` - Progress bars
- `pytest` & `pytest-asyncio` - Testing framework

## Project Structure

```
appdome_rag_project/
├── crawler.py              # Async web crawler
├── ingest.py               # Embedding generation and indexing
├── chat.py                 # Interactive Q&A interface
├── utils.py                # Utility functions (LRU cache)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/
│   └── raw/                # Crawled JSON files (created by crawler)
├── index/
│   ├── faiss_index.bin     # FAISS vector index (created by ingest)
│   └── metadata.json       # Chunk metadata (created by ingest)
└── tests/
    ├── conftest.py         # Shared pytest fixtures
    ├── test_crawler.py     # Crawler tests
    ├── test_ingest.py      # Ingestor tests
    └── test_chat.py        # Chat tests
```

## Usage Guide

### Running the Crawler

The crawler fetches Appdome how-to articles and saves them as JSON files.

```bash
python crawler.py
```

**What it does:**
- Starts from: `https://www.appdome.com/how-to/#`
- Crawls all linked articles on the same domain containing "how-to"
- Extracts page content, converts to Markdown
- Saves JSON files to `data/raw/` directory

**Configuration Options:**

You can modify the crawler settings in `crawler.py`:

```python
crawler = AsyncCrawler(
    start_url="https://www.appdome.com/how-to/#",
    max_concurrency=10,  # Number of parallel requests
    max_pages=4000       # Maximum pages to crawl
)
```

**Expected Output:**
- Progress messages showing pages processed
- JSON files created in `data/raw/` with format: `{slugified-url}.json`
- Each JSON contains: `url`, `title`, `content`

**Runtime:** Depends on number of pages and network speed (typically 10-30 minutes for full crawl)

### Running the Ingestor

The ingestor processes crawled JSON files and creates a searchable vector index.

```bash
python ingest.py
```

**Prerequisites:**
- Must have JSON files in `data/raw/` directory (run crawler first)

**What it does:**
- Reads all JSON files from `data/raw/`
- Splits content into chunks (600 chars with 100 char overlap)
- Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Creates FAISS index for fast similarity search
- Saves index to `index/faiss_index.bin` and metadata to `index/metadata.json`

**Configuration Options:**

In `ingest.py`, you can modify:

```python
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
CHUNK_SIZE = 600          # Characters per chunk
CHUNK_OVERLAP = 100       # Overlap between chunks
DEFAULT_BATCH_SIZE = 128  # Batch size for embeddings
```

**Expected Output:**
- Progress bar showing files processed
- "Saving partial index..." messages every 1000 chunks
- Final message: "Done. Indexed X chunks."

**Runtime:** Depends on number of files and content size (typically 5-15 minutes)

### Running the Chat Interface

The chat interface provides an interactive Q&A experience.

```bash
python chat.py
```

**Prerequisites:**
- Must have index files in `index/` directory (run ingestor first)
- Ollama must be running with `llama3` model installed

**What it does:**
- Loads the FAISS index and metadata
- Loads the embedding model
- Provides interactive prompt for questions
- Retrieves relevant context from index
- Generates answers using Ollama's llama3 model

**Usage Example:**

```
Loading my indexing, please wait...
Indexing loaded, loading model...
Appdome Threat-Expert here, how may I assist? Type 'exit' to quit.

👤 Question: What is mobile app security?

🤖 AI:
Mobile app security involves protecting applications from threats...

🔗 Sources:
 - https://www.appdome.com/how-to/article-1
 - https://www.appdome.com/how-to/article-2
```

**Exit Commands:**
- Type `exit` or `quit` to exit the chat interface
- Press `Ctrl+C` to force quit

**Configuration Options:**

In `chat.py`, you can modify:

```python
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
TOP_K = 4                    # Number of chunks to retrieve
MAX_CONTEXT_CHARS = 4000      # Maximum context length
SIM_THRESHOLD = 0.45          # Minimum similarity score
```

And in the `ask()` method:

```python
response = ollama.generate(
    model="llama3",           # LLM model name
    prompt=prompt,
    options={"temperature": 0.2}  # Lower = more focused
)
```

## Testing

The project includes comprehensive tests using pytest.

### Running All Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with output capture disabled (see print statements)
pytest tests/ -s
```

### Running Specific Test Files

```bash
# Test crawler only
pytest tests/test_crawler.py

# Test ingestor only
pytest tests/test_ingest.py

# Test chat only
pytest tests/test_chat.py
```

### Running Specific Test Classes or Functions

```bash
# Run a specific test class
pytest tests/test_crawler.py::TestAsyncCrawlerUnit

# Run a specific test function
pytest tests/test_crawler.py::TestAsyncCrawlerUnit::test_is_valid_link
```

### Test Markers

Some tests are marked as slow (they use real models):

```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Run only slow tests
pytest tests/ -m "slow"
```

### Understanding Test Output

- `.` = Test passed
- `F` = Test failed
- `s` = Test skipped
- `E` = Test error

### Test Coverage

The test suite includes:
- **Unit tests**: Test individual functions with mocked dependencies
- **Integration tests**: Test full workflows with real dependencies
- **Edge cases**: Empty inputs, missing files, error handling

## Configuration

### Key Constants

**Crawler** (`crawler.py`):
- `max_concurrency`: Number of parallel HTTP requests (default: 10)
- `max_pages`: Maximum pages to crawl (default: 4000)
- `output_dir`: Where to save JSON files (default: "data/raw")

**Ingestor** (`ingest.py`):
- `MODEL_NAME`: Embedding model (default: "sentence-transformers/all-MiniLM-L6-v2")
- `CHUNK_SIZE`: Characters per chunk (default: 600)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 100)
- `DEFAULT_BATCH_SIZE`: Batch size for embeddings (default: 128)
- `INPUT_DIR`: Source directory for JSON files (default: "data/raw")
- `INDEX_DIR`: Output directory for index files (default: "index")

**Chat** (`chat.py`):
- `MODEL_NAME`: Embedding model (default: "sentence-transformers/all-MiniLM-L6-v2")
- `TOP_K`: Number of chunks to retrieve (default: 4)
- `MAX_CONTEXT_CHARS`: Maximum context length (default: 4000)
- `SIM_THRESHOLD`: Minimum similarity score (default: 0.45)
- `INDEX_DIR`: Directory containing index files (default: "index")

### Modifying Settings

1. **For Crawler**: Edit the `AsyncCrawler` instantiation in `crawler.py`:
   ```python
   crawler = AsyncCrawler(
       start_url="https://www.appdome.com/how-to/#",
       max_concurrency=5,   # Reduce for slower networks
       max_pages=100        # Limit for testing
   )
   ```

2. **For Ingestor**: Modify constants at the top of `ingest.py`

3. **For Chat**: Modify constants at the top of `chat.py` and the `ollama.generate()` call

## Troubleshooting

### Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'X'`

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### FAISS Index Not Found

**Error**: `RuntimeError: FAISS index not found. Run ingestion first.`

**Solution**:
1. Ensure you've run the ingestor: `python ingest.py`
2. Check that `index/faiss_index.bin` exists
3. Verify you're running chat from the project root directory

### Ollama Connection Issues

**Error**: Connection refused or model not found

**Solution**:
```bash
# Check if Ollama is running
ollama list

# If not, start Ollama service (varies by OS)
# Then pull the model
ollama pull llama3

# Test connection
ollama run llama3 "Hello"
```

### Import Errors

**Error**: `ImportError: cannot import name 'X'`

**Solution**:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)
- Try reinstalling problematic package: `pip install --force-reinstall <package>`

### Crawler Not Finding Pages

**Issue**: Crawler finds 0 pages or very few pages

**Solution**:
- Check if the start URL is accessible
- Verify network connection
- Check if the website structure has changed
- Review `is_valid_link()` method in `crawler.py` for filtering logic

### Out of Memory Errors

**Error**: Memory errors during ingestion

**Solution**:
- Reduce `DEFAULT_BATCH_SIZE` in `ingest.py`
- Process files in smaller batches
- Close other applications to free memory

### Slow Performance

**Issue**: Crawling or ingestion is very slow

**Solution**:
- For crawler: Adjust `max_concurrency` (lower if network is slow)
- For ingestor: The first run downloads the model (~80MB), subsequent runs are faster
- Ensure you're using CPU-optimized packages (faiss-cpu, not faiss-gpu unless you have CUDA)

## Development

### Project Structure for Contributors

- **Main modules**: `crawler.py`, `ingest.py`, `chat.py`
- **Utilities**: `utils.py` (LRU cache implementation)
- **Tests**: `tests/` directory with comprehensive test coverage
- **Fixtures**: `tests/conftest.py` contains shared test fixtures

### Adding New Tests

1. Create test functions in the appropriate test file
2. Use fixtures from `conftest.py` for common setup
3. Mock external dependencies (HTTP, Ollama, etc.) for unit tests
4. Use real dependencies for integration tests (mark with `@pytest.mark.slow` if needed)

Example:
```python
def test_new_feature(temp_data_dir):
    """Test description."""
    # Your test code here
    assert expected == actual
```

### Code Style

- Follow PEP 8 Python style guide
- Use type hints where appropriate
- Add docstrings to classes and functions
- Keep functions focused and testable

### Running Tests During Development

```bash
# Run tests in watch mode (requires pytest-watch)
ptw tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

- Uses [sentence-transformers](https://www.sbert.net/) for embeddings
- Uses [FAISS](https://github.com/facebookresearch/faiss) for vector search
- Uses [Ollama](https://ollama.ai/) for local LLM inference
- Uses [LangChain](https://www.langchain.com/) text splitters
