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

# --- Search Implementation ---

async def web_search(query: str) -> str:
    """Search the web using duckduckgo-search and return results."""
    logger.info(f"[WEB_SEARCH] query: {query}")

    try:
        from ddgs import DDGS
        
        # synchronous DDGS usage
        results = []
        with DDGS() as ddgs:
            # Use text search with a limit
            search_results = list(ddgs.text(query, max_results=5))
            
            for r in search_results:
                title = r.get("title", "")
                url = r.get("href", "")
                snippet = r.get("body", "")
                if title or snippet:
                    results.append(f"**{title}**\nURL: {url}\n{snippet}")

        if results:
            raw_output = f"Search results for '{query}':\n\n" + "\n\n---\n\n".join(results)
            
            # --- Synthesis with Phi-3 ---
            try:
                from server_manager import UtilityServerManager
                utility_manager = UtilityServerManager()
                
                if not utility_manager.is_ready:
                    logger.info("[WEB_SEARCH] Starting utility server for synthesis...")
                    # We can't await in a non-async function if this wasn't async, but web_search IS async.
                    await utility_manager.start()

                synthesis_prompt = (
                    f"You are a strict research assistant. Your task is to summarize the provided Search Results for the query '{query}'.\n"
                    "Rules:\n"
                    "1. Only use information present in the Search Results.\n"
                    "2. Do not invent facts, stories, or external information.\n"
                    "3. If the answer is not in the results, state that.\n"
                    "4. Be concise.\n\n"
                    f"<search_results>\n{raw_output[:6000]}\n</search_results>"  # Truncate to 6k
                )
                
                logger.info(f"[WEB_SEARCH] Sending results to Phi-3 for synthesis...")
                # Use low temperature to reduce hallucinations/creativity
                synthesis = await utility_manager.infer(synthesis_prompt, temperature=0.0)
                logger.info(f"[WEB_SEARCH] Synthesis complete ({len(synthesis)} chars).")
                return f"Synthesis of Search Results for '{query}':\n\n{synthesis}\n\n[Source Data: {len(results)} results]"

            except Exception as e:
                logger.error(f"[WEB_SEARCH] Synthesis failed: {e}")
                return raw_output[:MAX_CONTENT_CHARS]

        else:
            output = f"No results found for '{query}'."

        logger.info(f"[WEB_SEARCH] Found {len(results)} results")
        return output[:MAX_CONTENT_CHARS]

    except Exception as e:
        logger.error(f"[WEB_SEARCH] Error: {e}")
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

        logger.info(f"[WEB_BROWSE] Got {len(text)} chars from {url}")
        
        # --- Summarization with Phi-3 ---
        from server_manager import UtilityServerManager
        utility_manager = UtilityServerManager()
        
        if not utility_manager.is_ready:
            # If not started, try to start it (non-blocking if possible, but here we await)
            logger.info("[WEB_BROWSE] Starting utility server for summarization...")
            await utility_manager.start()

        summary_prompt = (
            f"Please summarize the following web page content concisely for a researcher. "
            f"Focus on the main facts and details relevant to the page title.\n\n"
            f"Content:\n{text}"
        )

        try:
            logger.info(f"[WEB_BROWSE] Sending {len(text)} chars to Phi-3 for summarization...")
            summary = await utility_manager.infer(summary_prompt)
            logger.info(f"[WEB_BROWSE] Summarization complete ({len(summary)} chars).")
            return f"Summary of {url}:\n\n{summary}\n\n[Original URL: {url}]"
        except Exception as e:
            logger.error(f"[WEB_BROWSE] Summarization failed: {e}")
            return f"Content from {url} (Summarization failed):\n\n{text}"

    except Exception as e:
        logger.error(f"[WEB_BROWSE] Error: {e}")
        return f"Failed to fetch {url}: {str(e)}"
