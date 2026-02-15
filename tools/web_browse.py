"""Web browsing and search tools for the LLM."""

import logging
import re
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("tools.web")

MAX_CONTENT_CHARS = 6000
REQUEST_TIMEOUT = 15.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


# --- Tool Definitions ---

WEB_SEARCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the internet for current information, facts, or events. "
            "Use this when the user asks about something that requires up-to-date data "
            "(e.g., news, sports scores, weather, stock prices). "
            "Do NOT use this for simple greetings or common knowledge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string",
                }
            },
            "required": ["query"],
        },
    },
}

WEB_BROWSE_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "web_browse",
        "description": (
            "Fetch and read the text content of a specific web page URL. "
            "Returns the main readable text extracted from the page. "
            "Use this when you have a specific URL to read."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to fetch (must start with http:// or https://)",
                }
            },
            "required": ["url"],
        },
    },
}


# --- Search Implementation ---

import random

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) Gecko/20100101 Firefox/125.0",
]

def get_random_header():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    }


async def web_search(query: str) -> str:
    """Search the web using DuckDuckGo Lite and return results."""
    logger.info(f"[WEB_SEARCH] query: {query}")

    # Use Lite version which is less prone to bot detection
    search_url = f"https://lite.duckduckgo.com/lite/"
    
    # Form data for POST request to Lite version is sometimes more reliable,
    # but GET with params works too. Let's try GET first with specific params.
    # q={query} & kl=us-en (region)
    params = {
        "q": query,
        "kl": "us-en"
    }

    try:
        async with httpx.AsyncClient(
            follow_redirects=True, 
            timeout=REQUEST_TIMEOUT,
            http2=True, # HTTP/2 can sometimes bypass simple fingerprinting
        ) as client:
            response = await client.post(
                search_url, 
                data=params, 
                headers=get_random_header()
            )
            
            # If 403/429, try GET
            if response.status_code in (403, 418, 429):
                logger.warning(f"[WEB_SEARCH] POST failed with {response.status_code}, retrying GET")
                response = await client.get(
                    search_url, 
                    params=params, 
                    headers=get_random_header()
                )
            
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        # DuckDuckGo Lite structure:
        # It's a table. Rows alternate between title/link and snippet.
        # But sometimes they are not strictly alternating if there are ads or other things.
        # We look for .result-link, start a new item. Then look for .result-snippet to populate the current item.
        
        # Select all rows in the main table
        # usually it's the 3rd table, but let's be more specific or generic:
        # Select all 'tr' that contain either .result-link or .result-snippet
        rows = soup.select("tr")
        
        current_result = {}
        
        count = 0
        for row in rows:
            if count >= 8:
                break
                
            link_a = row.select_one("a.result-link")
            snippet_td = row.select_one("td.result-snippet")
            
            if link_a:
                # Start of a new result. If we had a previous one pending snippet, we might skip it or push it?
                # Usually snippet comes after.
                
                title = link_a.get_text(strip=True)
                href = link_a.get("href", "")
                
                current_result = {
                    "title": title,
                    "url": href,
                    "snippet": ""
                }
            
            elif snippet_td and current_result:
                # Snippet for the current result
                current_result["snippet"] = snippet_td.get_text(strip=True)
                
                # Complete result
                results.append(f"**{current_result['title']}**\nURL: {current_result['url']}\n{current_result['snippet']}")
                current_result = {} # Reset
                count += 1

        if results:
            output = f"Search results for '{query}':\n\n" + "\n\n---\n\n".join(results)
        else:
            # Fallback: try to extract any text from the page
            # Maybe the selector failed or layout changed
            # html.duckduckgo.com fallback?
            logger.warning("[WEB_SEARCH] No results found on Lite, trying HTML version fallback")
            return await _web_search_html_fallback(query)

        logger.info(f"[WEB_SEARCH] Found {len(results)} results")
        return output[:MAX_CONTENT_CHARS]

    except Exception as e:
        logger.error(f"[WEB_SEARCH] Lite Error: {e}")
        # Try fallback
        return await _web_search_html_fallback(query)


async def _web_search_html_fallback(query: str) -> str:
    """Fallback to standard HTML DuckDuckGo."""
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=REQUEST_TIMEOUT
        ) as client:
            response = await client.get(search_url, headers=get_random_header())
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for i, result in enumerate(soup.select(".result")):
            if i >= 6:
                break
            title_el = result.select_one(".result__title a")
            snippet_el = result.select_one(".result__snippet")
            url_el = result.select_one(".result__url")

            title = title_el.get_text(strip=True) if title_el else ""
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            url = url_el.get_text(strip=True) if url_el else ""

            if title or snippet:
                results.append(f"**{title}**\nURL: {url}\n{snippet}")

        if results:
            return f"Search results for '{query}':\n\n" + "\n\n---\n\n".join(results)
        else:
            return "No search results found."
    except Exception as e:
        return f"Search failed: {str(e)}"


# --- Browse Implementation ---

async def web_browse(url: str) -> str:
    """Fetch a URL and return cleaned text content."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    logger.info(f"[WEB_BROWSE] url: {url}")

    # If URL looks like a Google search, redirect to web_search
    if "google.com/search" in url:
        import re
        match = re.search(r'[?&]q=([^&]+)', url)
        if match:
            query = match.group(1).replace('+', ' ')
            return await web_search(query)

    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=REQUEST_TIMEOUT
        ) as client:
            response = await client.get(url, headers=HEADERS)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "iframe", "noscript", "svg", "form"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Collapse excessive blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        if len(text) > MAX_CONTENT_CHARS:
            text = text[:MAX_CONTENT_CHARS] + "\n\n[... content truncated ...]"

        logger.info(f"[WEB_BROWSE] Got {len(text)} chars")
        return f"Content from {url}:\n\n{text}"

    except Exception as e:
        logger.error(f"[WEB_BROWSE] Error: {e}")
        return f"Failed to fetch {url}: {str(e)}"
