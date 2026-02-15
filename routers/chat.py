"""API router for chat — proxies to llama-server with tool-call loop."""

import json
import logging
import random
import re
import string
from typing import Optional

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from server_manager import ServerManager, ServerState
from tools import get_tool_definitions, execute_tool
from tools.chart_recognizer import classify_chart
from agents import ResearcherAgent, AnalystAgent

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger("chat")
logging.basicConfig(level=logging.DEBUG)


@router.post("")
async def chat(request: Request):
    """
    Chat endpoint that proxies to llama-server's /v1/chat/completions.
    Supports streaming via SSE. Implements a tool-call loop: if the model
    requests a tool call, executes it and re-sends the updated conversation.
    """
    body = await request.json()
    messages = body.get("messages", [])
    enable_tools = body.get("enable_tools", True)
    system_prompt = body.get("system_prompt", "")
    agent_type = body.get("agent", "researcher")

    # Initialize the selected agent
    if agent_type == "analyst":
        agent = AnalystAgent()
    else:
        agent = ResearcherAgent()

    manager = ServerManager()
    if manager.info.state != ServerState.RUNNING:
        return {"error": "No model is currently running. Start a model first."}

    # Use Agent's logic for prompt and tool enablement
    agent_config = await agent.process(messages, tools_enabled=enable_tools)
    effective_system = agent_config["system_prompt"]
    
    # Enable tools based on agent preference (Analyst might disable them)
    enable_tools = agent_config.get("tool_choice", "auto") != "none"

    if system_prompt:
        effective_system = f"{effective_system}\n\nUser specific instructions: {system_prompt}"

    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": effective_system})

    # Auto-detect images and run chart-recognizer
    messages, chart_event = _preprocess_images(messages, manager)

    # Auto-detect search intent (Phi-3)
    # Only run if no chart event (chart takes precedence) and agent is Researcher
    search_query = None
    if not chart_event and agent_type == "researcher":
        search_query = await _detect_search_intent(messages, manager)

    async def _generate():
        # If chart analysis was performed, emit it as an event first
        if chart_event:
            logger.info("[CHAT] Chart event detected. Short-circuiting LLM.")
            yield f"data: {json.dumps(chart_event)}\n\n"
            
            # Short-circuit: Skip LLM entirely as requested
            msg = {
                "choices": [{
                    "delta": {"content": "\n\n**Analysis Complete (YOLOv8 - FINAL).**\nSee the detected patterns and annotated chart above."},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(msg)}\n\n"
            yield "data: [DONE]\n\n"
            logger.info("[CHAT] Exiting _generate (short-circuit).")
            return

        # If search intent detected, execute search with UI feedback
        if search_query:
            # 1. Emit tool_start event
            yield f"data: {json.dumps({'type': 'tool_start', 'tool': 'web_search', 'args': {'query': search_query}})}\n\n"
            
            # 2. Execute search and stream progress
            from tools.web_browse import web_search
            result = ""
            async for event in web_search(search_query):
                if event["type"] == "progress":
                    yield f"data: {json.dumps({
                        'type': 'tool_update', 
                        'tool': 'web_search', 
                        'status': event['msg'],
                        'model': event.get('model'),
                        't_s': event.get('t_s', 0)
                    })}\n\n"
                elif event["type"] == "result":
                    result = event["content"]
            
            # 3. Emit tool_result event
            yield f"data: {json.dumps({'type': 'tool_result', 'tool': 'web_search', 'result': result})}\n\n"
            
            # 4. Stream result as content
            yield f"data: {json.dumps({'type': 'content', 'content': result})}\n\n"
            yield "data: [DONE]\n\n"
            logger.info("[CHAT] Exiting _generate (search short-circuit).")
            return

        logger.info("[CHAT] No chart or search event. Proceeding to LLM.")
        effective_tools = enable_tools
        async for chunk in _stream_chat(messages, effective_tools, manager):
            yield chunk

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _preprocess_images(messages: list, manager: ServerManager) -> tuple[list, dict | None]:
    """
    Scan messages for images.
    1. For the LATEST message: if found, run YOLOv8 and create chart_event.
    2. For ALL messages: if model is text-only, strip images to save context/prevent errors.
    Returns (modified_messages, chart_event_or_None).
    """
    chart_event = None
    is_multimodal = getattr(manager.info, 'is_multimodal', False)

    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        
        if role == "user" and isinstance(content, list):
            # Extract text and image parts
            image_data = None
            has_image = False
            user_text = ""
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image_url":
                        has_image = True
                        url = part.get("image_url", {}).get("url", "")
                        if url: image_data = url
                    elif part.get("type") == "text":
                        user_text += part.get("text", "")

            # If this is the LATEST message and has an image, run YOLO
            if i == len(messages) - 1 and image_data:
                try:
                    result = classify_chart(image_data, top_k=10)
                    patterns = result.get("patterns", [])
                    chart_event = {
                        "type": "chart_analysis",
                        "is_chart": result.get("is_chart", False),
                        "patterns": patterns,
                        "summary": result.get("summary", ""),
                        "annotated_image": result.get("annotated_image"),
                    }
                    logger.info(f"[CHART] Detected {len(patterns)} patterns for latest message.")
                    
                    # Prepend pattern summary to user text for the LLM
                    if patterns:
                        pattern_lines = [f"- {p['label']} ({p['probability']}% confidence)" for p in patterns]
                        pattern_text = "\n".join(pattern_lines)
                        
                        # We'll prepend this to the user text so the model knows what was found
                        if user_text:
                            user_text = f"[Image Analysis: {pattern_text}]\n{user_text}"
                        else:
                            user_text = f"Analyze this chart. Detected: {pattern_text}"

                except Exception as e:
                    logger.error(f"[CHART] Detection failed: {e}")

            # STRIP images if model is text-only or for processed history turns
            if has_image and (not is_multimodal or i < len(messages) - 1):
                msg["content"] = user_text if user_text else "Image input (vision analysis disabled)"
                logger.info(f"[STRIP] Removed image from message {i} (multimodal={is_multimodal}).")

    return messages, chart_event


async def _detect_search_intent(messages: list, manager: ServerManager) -> str | None:
    """
    Check if the latest message requires a web search using Phi-3.
    Returns AN OPTIMIZED search_query string if YES, else None.
    """
    if not messages:
        return None
        
    last_msg = messages[-1]
    if last_msg.get("role") != "user":
        return None
        
    content = last_msg.get("content", "")
    if isinstance(content, list):
        # Flatten content for text analysis
        text_content = ""
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_content += part.get("text", "")
        content = text_content
    
    if not content or len(content) < 5:
        return None

    # Use UtilityServerManager (Phi-3) for classification and optimization
    try:
        from server_manager import UtilityServerManager
        utility_manager = UtilityServerManager()
        
        if not utility_manager.is_ready:
            await utility_manager.start()
            
        import datetime
        current_date = datetime.date.today().isoformat()
        
        prompt = (
            f"Current Date: {current_date}\n"
            "Determine if the user message below requires a web search to answer accurately (e.g. for news, dates, prices, or recent events).\n"
            "If YES, respond with 'YES: [Concise Search Query]'. Rewrite the query to be professional and effective for a search engine.\n"
            "Expand informal terms like 'tomo' or 'tomorrow' into the actual date if possible, or use 'upcoming' for recent context.\n"
            "If NO, respond with 'NO'.\n"
            "CRITICAL: Do not provide an answer. Do not include 'AI:' or 'Response:' prefixes. Only provide the query.\n\n"
            f"User Message: {content}\n"
            "Response:"
        )
        
        # Use low temperature and stop on newline to prevent over-generation
        res = await utility_manager.infer(prompt, temperature=0.0, stop=["\n", "<|end|>", "AI:", "Response:"])
        response = res.get("content", "").strip().split("\n")[0].strip()
        
        if response.upper().startswith("YES"):
            # Extract query after "YES:" or "YES "
            query = response
            if ":" in query:
                query = query.split(":", 1)[1].strip()
            elif query.upper().startswith("YES "):
                query = query[4:].strip()
            else:
                # Fallback to original content if format is weird
                query = content
            
            # Final safety strip for common hallucinated prefixes
            for prefix in ["AI:", "Response:", "Search Query:", "[", "{"]:
                if query.startswith(prefix):
                    query = query[len(prefix):].strip()
                
            logger.info(f"[INTENT] Search required. Optimized query: {query}")
            return query
        else:
            logger.debug(f"[INTENT] No search required for: {content[:30]}...")

    except Exception as e:
        logger.error(f"[INTENT] Classification/Optimization failed: {e}")
        
    return None
def _parse_functional_args(args_str: str) -> dict:
    """Parse k=\"v\" or k='v' pairs from a functional-style string."""
    pairs = re.findall(r'(\w+)\s*=\s*["\'](.*?)["\']', args_str)
    return {k: v for k, v in pairs}


def _extract_tool_calls_from_text(text: str) -> list[dict] | None:
    """
    Fallback: detect tool calls embedded as text in the content.
    Supports JSON tags, markdown blocks, and functional-style XML tags.
    """
    patterns = [
        # <tool_call> JSON tags
        (r'<tool_call>\s*(\{.*?\})\s*</tool_call>', "json"),
        # ```json blocks
        (r'```(?:json)?\s*(\{[^`]*?"name"\s*:\s*"web_(?:browse|search)"[^`]*?\})\s*```', "json"),
        # Bare JSON objects
        (r'(\{"name"\s*:\s*"web_(?:browse|search)"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\})', "json"),
        # <TOOLCALL>[name(args)]</TOOLCALL> format (used by Nemotron/Command-R)
        (r'<TOOLCALL>\[(.*?)\((.*?)\)\]</TOOLCALL>', "functional"),
    ]

    results = []
    for pattern, ptype in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                if ptype == "json":
                    obj = json.loads(match)
                    name = obj.get("name", "")
                    args = obj.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                else:
                    # functional: match[0] is name, match[1] is args string
                    name = match[0].strip()
                    if not name: continue
                    args = _parse_functional_args(match[1])

                # Skip empty/invalid calls
                if not name or (not args and ptype == "functional" and "(" not in match[0]):
                    # If it's just <TOOLCALL>[]</TOOLCALL> or similar
                    continue

                # Remapping: if model confusingly calls web_browse with a query, treat as search
                if name == "web_browse" and "query" in args and "url" not in args:
                    name = "web_search"
                
                # Standardize arguments: map 'q' to 'query' for web_search
                if name == "web_search" and "q" in args and "query" not in args:
                    args["query"] = args.pop("q")

                tc_id = ''.join(random.choices(string.ascii_letters + string.digits, k=9))
                results.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                })
            except (json.JSONDecodeError, TypeError):
                continue

    return results if results else None


async def _stream_chat(messages: list, enable_tools: bool, manager: ServerManager):
    """Generator that streams chat responses and handles tool calls."""
    base_url = f"http://127.0.0.1:{manager.info.port}"
    max_tool_rounds = 5  # prevent infinite tool-call loops

    for round_num in range(max_tool_rounds + 1):
        payload = {
            "model": manager.info.model_name,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        if enable_tools and round_num < max_tool_rounds:
            tool_defs = get_tool_definitions()
            if tool_defs:
                payload["tools"] = tool_defs
                # Suggest tools, but let the model decide if they are needed
                payload["tool_choice"] = "auto"

        # Collect the full response to detect tool calls
        full_content = ""
        tool_calls_collected = []
        current_tool_call = None

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{base_url}/v1/chat/completions",
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        error_text = ""
                        async for chunk in response.aiter_text():
                            error_text += chunk
                        yield f"data: {json.dumps({'type': 'error', 'content': error_text})}\n\n"
                        return

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = data.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        finish_reason = choices[0].get("finish_reason")

                        # Handle text content
                        content = delta.get("content", "")
                        if content:
                            full_content += content
                            
                            # Suppress "pseudo code" (tool call tags) from streaming to user
                            # We buffer if a tag might be starting
                            suppress_patterns = [r'<tool_call>', r'</tool_call>', r'<TOOLCALL>', r'</TOOLCALL>']
                            display_content = content
                            for p in suppress_patterns:
                                if p in full_content:
                                    # This is a bit simplistic; ideally we'd use a real buffer
                                    # but for now we just strip known tags from the delta
                                    display_content = display_content.replace(p, "")
                            
                            if display_content.strip() or not content.startswith('<'):
                                yield f"data: {json.dumps({'type': 'content', 'content': display_content})}\n\n"

                        # Handle tool calls in the delta
                        if "tool_calls" in delta:
                            logger.info(f"[TOOL_CALL DELTA] {delta['tool_calls']}")
                            for tc in delta["tool_calls"]:
                                idx = tc.get("index", 0)
                                if tc.get("id"):
                                    # New tool call starting
                                    current_tool_call = {
                                        "id": tc["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tc.get("function", {}).get("name", ""),
                                            "arguments": tc.get("function", {}).get("arguments", ""),
                                        },
                                    }
                                    while len(tool_calls_collected) <= idx:
                                        tool_calls_collected.append(None)
                                    tool_calls_collected[idx] = current_tool_call
                                elif current_tool_call:
                                    # Append to existing tool call arguments
                                    current_tool_call["function"]["arguments"] += (
                                        tc.get("function", {}).get("arguments", "")
                                    )

        except httpx.ConnectError:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Cannot connect to llama-server. Is a model running?'})}\n\n"
            return
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return

        # Filter out None entries
        tool_calls_collected = [tc for tc in tool_calls_collected if tc is not None]

        # Fallback: if no tool calls detected via API but text contains tool-like JSON
        if not tool_calls_collected and enable_tools and full_content:
            fallback = _extract_tool_calls_from_text(full_content)
            if fallback:
                logger.info(f"[FALLBACK] Detected tool calls in text: {fallback}")
                tool_calls_collected = fallback
                # Strip the tool call tags from full_content so they don't appear in history
                for pattern in [r'<tool_call>.*?</tool_call>', r'<TOOLCALL>.*?</TOOLCALL>']:
                    full_content = re.sub(pattern, '', full_content, flags=re.DOTALL).strip()

        # Normalize tool call IDs to exactly 9 alphanumeric chars
        # (required by some model chat templates like Qwen)
        for tc in tool_calls_collected:
            tid = tc.get("id", "")
            # Keep only alphanumeric chars, then pad/trim to 9
            alnum = ''.join(c for c in tid if c.isalnum())
            if len(alnum) >= 9:
                tc["id"] = alnum[:9]
            else:
                tc["id"] = ''.join(random.choices(string.ascii_letters + string.digits, k=9))

        logger.info(f"[ROUND {round_num}] content_len={len(full_content)}, tool_calls={len(tool_calls_collected)}")

        # If no tool calls, we're done
        if not tool_calls_collected:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Process tool calls
        assistant_msg = {"role": "assistant", "content": full_content or None, "tool_calls": tool_calls_collected}
        messages.append(assistant_msg)

        for tc in tool_calls_collected:
            func_name = tc["function"]["name"]
            try:
                func_args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
            except json.JSONDecodeError:
                func_args = {}

            # Notify frontend about tool execution
            yield f"data: {json.dumps({'type': 'tool_start', 'tool': func_name, 'args': func_args})}\n\n"

            # Execute tool with heartbeat to prevent timeouts
            # Run tool in a separate task so we can yield keep-alives while waiting
            import asyncio
            tool_task = asyncio.create_task(execute_tool(func_name, func_args))
            
            while not tool_task.done():
                try:
                    # Wait for 1 second at a time
                    await asyncio.wait_for(asyncio.shield(tool_task), timeout=1.0)
                except asyncio.TimeoutError:
                    # Yield a comment or empty event to keep connection alive
                    yield ": keep-alive\n\n"
            
            result = await tool_task
            
            yield f"data: {json.dumps({'type': 'tool_result', 'tool': func_name, 'result': result})}\n\n"

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

        # SPECIAL HANDLING: If web_search returns a synthesis, use it as the FINAL answer.
        # This prevents the main model from re-summarizing the summary.
        synthesis_result = None
        for msg in messages[-len(tool_calls_collected):]:
            if msg.get("role") == "tool" and "Synthesis of Search Results" in msg.get("content", ""):
                synthesis_result = msg["content"]
                break
        
        if synthesis_result:
            logger.info("[CHAT] Synthesis detected. Short-circuiting LLM.")
            # Remove the "Synthesis of..." prefix for a cleaner chat response if desired, 
            # or keep it as is. User wanted "routed as is".
            yield f"data: {json.dumps({'type': 'content', 'content': synthesis_result})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Continue the loop — the model will now generate a response using the tool results

    # If we exhausted rounds
    yield f"data: {json.dumps({'type': 'done'})}\n\n"
