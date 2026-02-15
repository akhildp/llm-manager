
import asyncio
import logging
from tools.web_browse import web_search, web_browse

logger = logging.getLogger("tools.deep_research")

DEEP_RESEARCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "deep_research",
        "description": (
            "Perform a deep, multi-step research on a topic. "
            "1. Searches for the query. "
            "2. Identifies top relevant URLs. "
            "3. Browses the content of multiple URLs in parallel. "
            "4. Returns a comprehensive aggregated report. "
            "Use this for complex questions requiring broad context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The research topic or question",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of pages to browse (default: 3)",
                    "default": 3,
                }
            },
            "required": ["query"],
        },
    },
}

async def deep_research(query: str, max_results: int = 3) -> str:
    """
    Perform deep research by searching and browsing multiple pages.
    """
    logger.info(f"[DEEP_RESEARCH] Starting for: {query}")
    
    # 1. Search
    search_results = await web_search(query)
    
    # 2. Extract URLs
    import re
    urls = re.findall(r'URL: ([^\s]+)', search_results)
    
    # Filter constraints
    unique_urls = []
    seen = set()
    for u in urls:
        if u not in seen and "duckduckgo" not in u and "google" not in u:
            unique_urls.append(u)
            seen.add(u)
            if len(unique_urls) >= max_results:
                break
                
    if not unique_urls:
        return f"Deep research could not find valid URLs for '{query}'.\nSearch results:\n{search_results}"
        
    logger.info(f"[DEEP_RESEARCH] Browsing {len(unique_urls)} URLs: {unique_urls}")
    
    # 3. Fetch in parallel
    tasks = [web_browse(url) for url in unique_urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 4. Aggregate
    report = [f"# Deep Research Report: {query}\n"]
    report.append(f"Based on {len(unique_urls)} sources:\n")
    
    for url, content in zip(unique_urls, results):
        report.append(f"## Source: {url}")
        if isinstance(content, Exception):
            report.append(f"Error fetching: {content}")
        else:
            # Truncate to avoid context overflow (approx 2000 chars per source)
            truncated = content[:2000] + ("\n[...]" if len(content) > 2000 else "")
            report.append(truncated)
        report.append("\n---\n")
        
    return "\n".join(report)
