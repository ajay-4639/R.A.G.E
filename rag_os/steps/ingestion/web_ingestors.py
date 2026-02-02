"""Web and API-based ingestion steps."""

from typing import Any
from urllib.parse import urlparse
import time

from rag_os.core.types import StepType
from rag_os.core.registry import register_step
from rag_os.models.document import Document, SourceType
from rag_os.steps.ingestion.base import BaseIngestionStep


@register_step(
    name="URLIngestionStep",
    step_type=StepType.INGESTION,
    description="Ingests content from a single URL",
    version="1.0.0",
)
class URLIngestionStep(BaseIngestionStep):
    """Ingestion step for fetching content from URLs."""

    def ingest(self, source_config: dict[str, Any] | None = None) -> list[Document]:
        """Ingest content from URL."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests library required: pip install requests")

        config = self.get_config()

        # Get URL from config or source_config
        url = config.source_uri
        if source_config:
            url = source_config.get("source_uri") or source_config.get("url") or url

        if not url:
            raise ValueError("URL is required for web ingestion")

        # Fetch configuration
        timeout = self.config.get("timeout", 30)
        headers = self.config.get("headers", {})
        verify_ssl = self.config.get("verify_ssl", True)

        # Add default user agent if not provided
        if "User-Agent" not in headers:
            headers["User-Agent"] = "RAG-OS/1.0 (Document Ingestion)"

        # Fetch the URL
        response = requests.get(
            url,
            timeout=timeout,
            headers=headers,
            verify=verify_ssl,
        )
        response.raise_for_status()

        # Extract content
        content_type = response.headers.get("Content-Type", "")
        content = response.text

        # Convert HTML to plain text if needed
        if "text/html" in content_type:
            content = self._html_to_text(content)
            source_type = SourceType.WEBPAGE
        else:
            source_type = SourceType.URL

        # Extract title from URL or content
        parsed_url = urlparse(url)
        title = self.config.get("title") or parsed_url.path.split("/")[-1] or parsed_url.netloc

        metadata = {
            "url": url,
            "content_type": content_type,
            "status_code": response.status_code,
            "content_length": len(content),
        }

        doc = self.create_document(
            content=content,
            source_uri=url,
            source_type=source_type,
            title=title,
            metadata=metadata,
        )

        return [doc]

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Get text
            text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n".join(lines)

        except ImportError:
            # Fallback: basic HTML stripping
            import re
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text)
            return text.strip()


@register_step(
    name="WebCrawlerIngestionStep",
    step_type=StepType.INGESTION,
    description="Crawls and ingests multiple pages from a website",
    version="1.0.0",
)
class WebCrawlerIngestionStep(URLIngestionStep):
    """Ingestion step for crawling multiple pages."""

    def ingest(self, source_config: dict[str, Any] | None = None) -> list[Document]:
        """Crawl and ingest multiple pages."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests library required: pip install requests")

        config = self.get_config()

        # Get starting URL
        start_url = config.source_uri
        if source_config:
            start_url = source_config.get("source_uri") or source_config.get("url") or start_url

        if not start_url:
            raise ValueError("URL is required for web crawling")

        # Crawl configuration
        max_pages = self.config.get("max_pages", 10)
        max_depth = self.config.get("max_depth", 2)
        delay_seconds = self.config.get("delay_seconds", 1.0)
        same_domain_only = self.config.get("same_domain_only", True)
        respect_robots = self.config.get("respect_robots", True)

        # Parse starting domain
        parsed_start = urlparse(start_url)
        start_domain = parsed_start.netloc

        # Track visited URLs and pages to process
        visited: set[str] = set()
        to_visit: list[tuple[str, int]] = [(start_url, 0)]  # (url, depth)
        documents: list[Document] = []

        while to_visit and len(documents) < max_pages:
            url, depth = to_visit.pop(0)

            if url in visited:
                continue

            if depth > max_depth:
                continue

            visited.add(url)

            # Check domain restriction
            if same_domain_only:
                parsed = urlparse(url)
                if parsed.netloc != start_domain:
                    continue

            try:
                # Fetch page
                doc_list = super().ingest({"source_uri": url})
                if doc_list:
                    doc = doc_list[0]
                    doc.metadata["crawl_depth"] = depth
                    documents.append(doc)

                    # Extract links for further crawling (if not at max depth)
                    if depth < max_depth:
                        new_links = self._extract_links(doc.content, url)
                        for link in new_links:
                            if link not in visited:
                                to_visit.append((link, depth + 1))

                # Rate limiting
                if delay_seconds > 0:
                    time.sleep(delay_seconds)

            except Exception as e:
                if not config.skip_errors:
                    raise
                # Log and continue

        return documents

    def _extract_links(self, content: str, base_url: str) -> list[str]:
        """Extract links from HTML content."""
        from urllib.parse import urljoin
        import re

        links: list[str] = []

        # Simple regex to find href attributes
        href_pattern = r'href=["\']([^"\']+)["\']'
        matches = re.findall(href_pattern, content)

        for href in matches:
            # Skip anchors, javascript, mailto, etc.
            if href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            # Convert relative URLs to absolute
            full_url = urljoin(base_url, href)

            # Only HTTP(S) URLs
            if full_url.startswith(("http://", "https://")):
                links.append(full_url)

        return list(set(links))


@register_step(
    name="APIIngestionStep",
    step_type=StepType.INGESTION,
    description="Ingests data from REST APIs",
    version="1.0.0",
)
class APIIngestionStep(BaseIngestionStep):
    """Ingestion step for fetching data from REST APIs."""

    def ingest(self, source_config: dict[str, Any] | None = None) -> list[Document]:
        """Ingest data from API."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests library required: pip install requests")

        config = self.get_config()

        # Get API URL
        api_url = config.source_uri
        if source_config:
            api_url = source_config.get("source_uri") or source_config.get("url") or api_url

        if not api_url:
            raise ValueError("API URL is required")

        # API configuration
        method = self.config.get("method", "GET").upper()
        headers = self.config.get("headers", {})
        params = self.config.get("params", {})
        body = self.config.get("body")
        auth = self.config.get("auth")  # (username, password) tuple
        timeout = self.config.get("timeout", 30)

        # Add common headers
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        # Make request
        response = requests.request(
            method=method,
            url=api_url,
            headers=headers,
            params=params,
            json=body,
            auth=tuple(auth) if auth else None,
            timeout=timeout,
        )
        response.raise_for_status()

        # Parse response
        data = response.json()

        # Extract documents from response
        documents = self._extract_documents(data, api_url)

        return documents

    def _extract_documents(self, data: Any, source_uri: str) -> list[Document]:
        """Extract documents from API response."""
        import json

        documents: list[Document] = []

        # Configuration for extraction
        data_path = self.config.get("data_path")  # e.g., "results.items"
        text_field = self.config.get("text_field")  # Field containing text content
        id_field = self.config.get("id_field")  # Field for document ID
        one_doc_per_item = self.config.get("one_doc_per_item", True)

        # Navigate to data path if specified
        if data_path:
            for key in data_path.split("."):
                if isinstance(data, dict):
                    data = data.get(key, {})
                elif isinstance(data, list) and key.isdigit():
                    data = data[int(key)]

        # Convert to list for uniform processing
        items = data if isinstance(data, list) else [data]

        if one_doc_per_item:
            # Create one document per item
            for i, item in enumerate(items):
                if text_field and isinstance(item, dict):
                    content = str(item.get(text_field, ""))
                else:
                    content = json.dumps(item, indent=2)

                doc_id = None
                if id_field and isinstance(item, dict):
                    doc_id = str(item.get(id_field))

                metadata = {"api_index": i}
                if isinstance(item, dict):
                    # Include non-text fields as metadata
                    for k, v in item.items():
                        if k not in (text_field, id_field) and not isinstance(v, (dict, list)):
                            metadata[k] = v

                doc = self.create_document(
                    content=content,
                    source_uri=f"{source_uri}#item{i}",
                    source_type=SourceType.API,
                    title=doc_id or f"api_item_{i}",
                    metadata=metadata,
                )
                documents.append(doc)
        else:
            # Create single document for all data
            content = json.dumps(data, indent=2)
            doc = self.create_document(
                content=content,
                source_uri=source_uri,
                source_type=SourceType.API,
                title="api_response",
                metadata={"item_count": len(items)},
            )
            documents.append(doc)

        return documents
