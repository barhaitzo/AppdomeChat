import asyncio
import httpx
import os
import json
import time
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from slugify import slugify
from urllib.parse import urljoin, urlparse


class AsyncCrawler:
    def __init__(self, start_url, max_concurrency=10, max_pages=4000):
        self.start_url = start_url
        self.output_dir = "data/raw"
        self.headers = {"User-Agent": "Appdome Crawler/1.0"}
        self.max_pages = max_pages
        self.max_concurrency = max_concurrency
        
        # The Semaphore: Only X tasks can run the 'protected' block at once
        self.semaphore = asyncio.Semaphore(max_concurrency)
        
        self.visited = set()
        self.to_visit = asyncio.Queue() # Using a Queue for async coordination
        self.prepare_output_dir()

    def prepare_output_dir(self):
        """Prepares the output directory by creating it if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    async def get_article_urls(self, client):
        """Phase 1: Visit the hub and extract every article link."""
        print(f"Harvesting links from Hub: {self.start_url}")
        resp = await client.get(self.start_url, follow_redirects=True)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        links = set()
        base_domain = urlparse(self.start_url).netloc
        
        for a in soup.find_all('a', href=True):
            full_url = urljoin(self.start_url, a['href']).split('#')[0].rstrip('/')
            # ONLY harvest links that stay on domain and contain 'how-to'
            # (And aren't the hub page itself)
            if urlparse(full_url).netloc == base_domain and "how-to" in full_url:
                if full_url != self.start_url:
                    links.add(full_url)
        
        print(f"Found {len(links)} articles to scrape.")
        return list(links)

    def is_valid_link(self, url):
        """Filters for same-domain and 'how-to' path."""
        base_domain = urlparse(self.start_url).netloc
        parsed = urlparse(url)
        return parsed.netloc == base_domain and "how-to" in parsed.path

    async def fetch_and_process(self, client):
        """
        The worker loop. It pulls from the queue indefinitely 
        until the queue is empty and joined.
        """
        while True:
            url = await self.to_visit.get()
            
            try:
                # Check constraints
                if url in self.visited or len(self.visited) >= self.max_pages:
                    continue

                async with self.semaphore:
                    print(f"Processing [{len(self.visited)}/{self.max_pages}]: {url}")
                    
                    try:
                        response = await client.get(url, timeout=15, follow_redirects=True)
                        if response.status_code != 200:
                            continue

                        self.visited.add(url)
                        soup = BeautifulSoup(response.text, 'html.parser')

                        # 1. EXTRACT LINKS (Inner Discovery)
                        # This allows the crawler to find Level 2, 3, 4... links
                        for a_tag in soup.find_all('a', href=True):
                            full_url = urljoin(url, a_tag['href']).split('#')[0].rstrip('/')
                            if self.is_valid_link(full_url) and full_url not in self.visited:
                                await self.to_visit.put(full_url)

                        # 2. CLEAN CONTENT
                        for element in soup(['nav', 'footer', 'script', 'style', 'header', 'aside']):
                            element.decompose()

                        # 3. SAVE JSON
                        page_data = {
                            "url": url,
                            "title": soup.title.string if soup.title else "No Title",
                            "content": md(str(soup), heading_style="ATX").strip(),
                        }

                        filename = f"{slugify(url)}.json"
                        with open(os.path.join(self.output_dir, filename), 'w', encoding='utf-8') as f:
                            json.dump(page_data, f, indent=4)

                    except Exception as e:
                        print(f"Error scraping {url}: {e}")
            
            finally:
                # Tell the queue that this specific item is done
                self.to_visit.task_done()

    async def run(self):
        """Initializes the worker pool and starts the process."""
        async with httpx.AsyncClient(headers=self.headers, follow_redirects=True) as client:
            # Seed the queue with the main page
            await self.to_visit.put(self.start_url)

            # Start a pool of workers. 
            # We create X workers where X is your max_concurrency.
            workers = []
            for _ in range(self.max_concurrency):
                task = asyncio.create_task(self.fetch_and_process(client))
                workers.append(task)

            # Wait until the queue is fully exhausted (all articles + discovered links)
            await self.to_visit.join()

            # Clean up: stop the workers
            for worker in workers:
                worker.cancel()

        print(f"Finished! Scraped {len(self.visited)} unique articles.")


if __name__ == "__main__":
    target_url = "https://www.appdome.com/how-to/#"
    crawler = AsyncCrawler(target_url)
    asyncio.run(crawler.run())