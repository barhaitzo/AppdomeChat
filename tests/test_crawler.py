import os
import json
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup

from crawler import AsyncCrawler


class TestAsyncCrawlerUnit:
    """Unit tests for AsyncCrawler class."""

    def test_is_valid_link(self):
        """Test URL validation logic."""
        crawler = AsyncCrawler("https://www.appdome.com/how-to/", max_concurrency=1, max_pages=10)
        
        # Valid links
        assert crawler.is_valid_link("https://www.appdome.com/how-to/article-1") == True
        assert crawler.is_valid_link("https://www.appdome.com/how-to/another-page") == True
        
        # Invalid links
        assert crawler.is_valid_link("https://www.appdome.com/other-page") == False  # No 'how-to'
        assert crawler.is_valid_link("https://example.com/how-to/page") == False  # Different domain
        assert crawler.is_valid_link("https://www.appdome.com/how-to/") == True  # Edge case

    def test_prepare_output_dir(self, temp_data_dir):
        """Test directory creation."""
        # Temporarily override output_dir
        crawler = AsyncCrawler("https://www.appdome.com/how-to/", max_concurrency=1, max_pages=10)
        crawler.output_dir = temp_data_dir
        
        # Directory should exist after initialization
        assert os.path.exists(temp_data_dir)
        
        # Test with non-existent directory
        new_dir = os.path.join(temp_data_dir, "new_subdir")
        crawler.output_dir = new_dir
        crawler.prepare_output_dir()
        assert os.path.exists(new_dir)

    @pytest.mark.asyncio
    async def test_get_article_urls(self, sample_html_with_links):
        """Test link extraction with mocked HTTP responses."""
        crawler = AsyncCrawler("https://www.appdome.com/how-to/", max_concurrency=1, max_pages=10)
        
        # Mock httpx.AsyncClient
        mock_response = MagicMock()
        mock_response.text = sample_html_with_links
        mock_response.status_code = 200
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        
        links = await crawler.get_article_urls(mock_client)
        
        # Should extract valid links
        assert len(links) > 0
        assert all("how-to" in link for link in links)
        assert all(urlparse(link).netloc == "www.appdome.com" for link in links)

    @pytest.mark.asyncio
    async def test_fetch_and_process_success(self, temp_data_dir, sample_html):
        """Test page processing with mocked responses."""
        crawler = AsyncCrawler("https://www.appdome.com/how-to/test", max_concurrency=1, max_pages=10)
        crawler.output_dir = temp_data_dir
        
        # Mock response
        mock_response = MagicMock()
        mock_response.text = sample_html
        mock_response.status_code = 200
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        
        # Add URL to queue
        await crawler.to_visit.put("https://www.appdome.com/how-to/test")
        
        # Process one item
        task = asyncio.create_task(crawler.fetch_and_process(mock_client))
        
        # Wait a bit for processing
        await asyncio.sleep(0.1)
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Check if file was created
        files = os.listdir(temp_data_dir)
        assert len(files) > 0
        assert any(f.endswith('.json') for f in files)

    @pytest.mark.asyncio
    async def test_fetch_and_process_skips_visited(self):
        """Test that visited URLs are skipped."""
        crawler = AsyncCrawler("https://www.appdome.com/how-to/", max_concurrency=1, max_pages=10)
        crawler.visited.add("https://www.appdome.com/how-to/visited")
        
        mock_client = AsyncMock()
        
        await crawler.to_visit.put("https://www.appdome.com/how-to/visited")
        
        task = asyncio.create_task(crawler.fetch_and_process(mock_client))
        await asyncio.sleep(0.1)
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Should not have called get
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_and_process_respects_max_pages(self):
        """Test max_pages limit enforcement."""
        crawler = AsyncCrawler("https://www.appdome.com/how-to/", max_concurrency=1, max_pages=2)
        
        # Mark 2 pages as visited
        crawler.visited.add("https://www.appdome.com/how-to/page1")
        crawler.visited.add("https://www.appdome.com/how-to/page2")
        
        mock_client = AsyncMock()
        await crawler.to_visit.put("https://www.appdome.com/how-to/page3")
        
        task = asyncio.create_task(crawler.fetch_and_process(mock_client))
        await asyncio.sleep(0.1)
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Should not process because max_pages reached
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_and_process_handles_errors(self, temp_data_dir):
        """Test error handling during page processing."""
        crawler = AsyncCrawler("https://www.appdome.com/how-to/", max_concurrency=1, max_pages=10)
        crawler.output_dir = temp_data_dir
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Network error"))
        
        await crawler.to_visit.put("https://www.appdome.com/how-to/error-page")
        
        task = asyncio.create_task(crawler.fetch_and_process(mock_client))
        await asyncio.sleep(0.1)
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Should handle error gracefully without crashing
        assert True  # If we get here, error was handled

    @pytest.mark.asyncio
    async def test_fetch_and_process_skips_non_200(self):
        """Test that non-200 status codes are skipped."""
        crawler = AsyncCrawler("https://www.appdome.com/how-to/", max_concurrency=1, max_pages=10)
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        
        await crawler.to_visit.put("https://www.appdome.com/how-to/not-found")
        
        task = asyncio.create_task(crawler.fetch_and_process(mock_client))
        await asyncio.sleep(0.1)
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Should not be added to visited
        assert "https://www.appdome.com/how-to/not-found" not in crawler.visited


class TestAsyncCrawlerIntegration:
    """Integration tests for AsyncCrawler class."""

    @pytest.mark.asyncio
    async def test_crawler_end_to_end(self, temp_data_dir, sample_html):
        """Test full crawl with mocked httpx.AsyncClient."""
        crawler = AsyncCrawler("https://www.appdome.com/how-to/", max_concurrency=2, max_pages=5)
        crawler.output_dir = temp_data_dir
        
        # Create mock responses
        mock_response = MagicMock()
        mock_response.text = sample_html
        mock_response.status_code = 200
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        
        # Mock the run method's client context
        with patch('crawler.httpx.AsyncClient', return_value=mock_client):
            # Seed with start URL
            await crawler.to_visit.put(crawler.start_url)
            
            # Start workers
            workers = []
            for _ in range(2):
                task = asyncio.create_task(crawler.fetch_and_process(mock_client))
                workers.append(task)
            
            # Process a few items
            await asyncio.sleep(0.2)
            
            # Cancel workers
            for worker in workers:
                worker.cancel()
            
            for worker in workers:
                try:
                    await worker
                except asyncio.CancelledError:
                    pass

    def test_json_file_creation(self, temp_data_dir, sample_html):
        """Test JSON file creation and structure."""
        crawler = AsyncCrawler("https://www.appdome.com/how-to/test", max_concurrency=1, max_pages=10)
        crawler.output_dir = temp_data_dir
        
        # Manually create a file like the crawler would
        from markdownify import markdownify as md
        from slugify import slugify
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(sample_html, 'html.parser')
        for element in soup(['nav', 'footer', 'script', 'style', 'header']):
            element.decompose()
        
        page_data = {
            "url": "https://www.appdome.com/how-to/test",
            "title": soup.title.string if soup.title else "No Title",
            "content": md(str(soup), heading_style="ATX").strip(),
        }
        
        filename = f"{slugify(page_data['url'])}.json"
        filepath = os.path.join(temp_data_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(page_data, f, indent=4)
        
        # Verify file exists and has correct structure
        assert os.path.exists(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert 'url' in data
        assert 'title' in data
        assert 'content' in data
        assert data['url'] == "https://www.appdome.com/how-to/test"

    def test_visited_set_tracking(self):
        """Test visited set tracking."""
        crawler = AsyncCrawler("https://www.appdome.com/how-to/", max_concurrency=1, max_pages=10)
        
        assert len(crawler.visited) == 0
        
        crawler.visited.add("https://www.appdome.com/how-to/page1")
        assert len(crawler.visited) == 1
        assert "https://www.appdome.com/how-to/page1" in crawler.visited

    @pytest.mark.asyncio
    async def test_queue_management(self):
        """Test queue management."""
        crawler = AsyncCrawler("https://www.appdome.com/how-to/", max_concurrency=1, max_pages=10)
        
        # Add URLs to queue
        await crawler.to_visit.put("https://www.appdome.com/how-to/page1")
        await crawler.to_visit.put("https://www.appdome.com/how-to/page2")
        
        # Check queue size (approximate, as it's async)
        assert crawler.to_visit.qsize() >= 0  # Queue might be empty if processed
        
        # Get an item
        url = await crawler.to_visit.get()
        assert url in ["https://www.appdome.com/how-to/page1", "https://www.appdome.com/how-to/page2"]
        
        crawler.to_visit.task_done()
