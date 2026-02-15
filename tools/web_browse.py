import logging
import httpx
import asyncio
from bs4 import BeautifulSoup
from typing import List, Optional, Dict, Any, AsyncGenerator
import json
import os

logger = logging.getLogger(__name__)

MAX_CONTENT_CHARS = 10000

WEB_SEARCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web using DuckDuckGo and get a synthesized answer with Deep Search.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"]
        }
    }
}

WEB_BROWSE_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "web_browse",
        "description": "Fetch and return the page content of a specific URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to browse."}
            },
            "required": ["url"]
        }
    }
}

async def _fetch_and_clean_url(url: str) -> str:
    """Fetch URL and clean HTML to get plain text."""
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
                
            text = soup.get_text(separator=' ')
            
            # Basic cleanup: remove extra whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text[:12000] # Limit per page
    except Exception as e:
        logger.warning(f"[WEB_BROWSE] Error fetching {url}: {e}")
        return f"Error fetching page content: {str(e)}"

async def web_browse(url: str) -> str:
    """Browse a specific URL and return its text content."""
    logger.info(f"[WEB_BROWSE] url: {url}")
    return await _fetch_and_clean_url(url)

async def web_search(query: str) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Search DuckDuckGo and synthesize results using LLM.
    Yields event dicts for progress and final result.
    """
    try:
        from ddgs import DDGS
        import asyncio
        
        # --- PHASE 1: SEARCH ---
        yield {"type": "progress", "msg": "Searching the web...", "model": "DuckDuckGo", "t_s": 0}
        
        results = []
        top_urls = []
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=5))
            for r in search_results:
                title = r.get("title", "")
                url = r.get("href", "")
                snippet = r.get("body", "")
                if title or snippet:
                    results.append(f"**{title}**\nURL: {url}\n{snippet}")
                    if len(top_urls) < 2:
                        top_urls.append(url)

        if not results:
            yield {"type": "result", "content": f"No results found for '{query}'."}
            return

        raw_snippets = "\n\n---\n\n".join(results)
        
        # --- PHASE 2: DEEP SEARCH (Browse & Summarize) ---
        deep_summaries = []
        if top_urls:
            logger.info(f"[WEB_SEARCH] Deep Search: Browsing and summarizing {len(top_urls)} URLs...")
            
            from server_manager import UtilityServerManager
            utility_manager = UtilityServerManager()
            if not utility_manager.is_ready:
                await utility_manager.start()

            browse_tasks = [_fetch_and_clean_url(url) for url in top_urls]
            page_texts = await asyncio.gather(*browse_tasks, return_exceptions=True)
            
            for i, text in enumerate(page_texts):
                if isinstance(text, str) and len(text) > 300:
                    url = top_urls[i]
                    yield {"type": "progress", "msg": f"Reading {url}...", "model": "Phi-3", "t_s": 0}
                    
                    summary_prompt = (
                        f"Summarize the key information from this page for the query '{query}'. "
                        f"Keep it to one dense paragraph.\n\n"
                        f"Page: {url}\nContent:\n{text[:4000]}"
                    )
                    try:
                        res = await utility_manager.infer(summary_prompt, n_predict=300)
                        summary = res.get("content", "")
                        t_s = res.get("t_s", 0)
                        if summary:
                            deep_summaries.append(f"Summary from {url}:\n{summary}")
                            yield {"type": "progress", "msg": f"Read {url}", "model": "Phi-3", "t_s": t_s}
                    except Exception as e:
                        logger.warning(f"[WEB_SEARCH] Failed to summarize {url}: {e}")

        full_context = f"Search Results for '{query}':\n\n{raw_snippets}\n\n" + "\n\n".join(deep_summaries)
        
        # --- PHASE 3: FINAL SYNTHESIS ---
        yield {"type": "progress", "msg": "Synthesizing final answer...", "model": "Nemotron", "t_s": 0}
        
        try:
            from server_manager import ServerManager
            main_manager = ServerManager()
            if not main_manager._info.state == "running":
                # Fallback to utility if main is not running
                logger.warning("[WEB_SEARCH] Main model not running, falling back to utility for synthesis")
                res = await utility_manager.infer(
                    f"Answer query: {query}\nContext: {full_context[:3000]}", 
                    temperature=0.0
                )
                yield {"type": "result", "content": res.get("content", "")}
                return

            synthesis_prompt = (
                "You are a helpful assistant providing a direct answer based on search results. "
                f"Please provide a comprehensive and concise answer for the query '{query}' based on the context below.\n\n"
                "CRITICAL: Do not include prefixes like 'Answer:', 'Summary:', 'Verification:', or 'AI:'. "
                "Start your response immediately with the information.\n\n"
                f"Context:\n{full_context[:6000]}"
            )
            
            logger.info(f"[WEB_SEARCH] Performing final synthesis with main model...")
            res = await main_manager.infer(synthesis_prompt, temperature=0.0)
            synthesis = res.get("content", "")
            t_s = res.get("t_s", 0)
            
            # Cleanup common prefixes
            synthesis = synthesis.strip()
            prefixes_to_strip = ["Answer:", "Summary:", "Verification:", "AI:", "Response:", "Detailed Answer:"]
            for p in prefixes_to_strip:
                if synthesis.lower().startswith(p.lower()):
                    synthesis = synthesis[len(p):].strip()
            
            yield {"type": "progress", "msg": "Finalizing answer", "model": "Nemotron", "t_s": t_s}
            yield {"type": "result", "content": synthesis}

        except Exception as e:
            logger.error(f"[WEB_SEARCH] Final synthesis failed: {e}")
            # Fallback to utility model
            try:
                res = await utility_manager.infer(
                    f"Answer query: {query}\nContext: {full_context[:3000]}", 
                    temperature=0.0
                )
                yield {"type": "result", "content": res.get("content", "")}
            except:
                yield {"type": "result", "content": "Search synthesis failed. Please try again."}

    except Exception as e:
        logger.error(f"[WEB_SEARCH] Error: {e}")
        yield {"type": "result", "content": f"Search failed: {str(e)}"}
