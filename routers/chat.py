"""API router for chat — proxies to llama-server with tool-call loop."""

import json
import logging
import random
import re
import string
import time
from datetime import datetime
from typing import Optional

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from server_manager import ServerManager, ServerState
from tools import get_tool_definitions, execute_tool

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

    manager = ServerManager()
    if manager.info.state != ServerState.RUNNING:
        return {"error": "No model is currently running. Start a model first."}

    # Guidance for balanced tool use
    default_guidance = (
        f"Current Date: {datetime.now().strftime('%Y-%m-%d')}\n"
        "You are a helpful AI assistant. Use tools like 'web_search' proactively when "
        "the user asks for real-world facts, current events, sports scores, or news. "
        "However, do NOT use tools for simple greetings like 'Hello' or casual small talk."
    )
    
    if system_prompt:
        effective_system = f"{default_guidance}\n\nUser specific instructions: {system_prompt}"
    else:
        effective_system = default_guidance

    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": effective_system})

    async def _generate():
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




def _extract_tool_calls_from_text(text: str) -> list[dict] | None:
    """
    Fallback: detect tool calls embedded as text in the content.
    Many models emit tool calls as JSON blocks like:
      <tool_call>{"name": "web_browse", "arguments": {"url": "..."}}</tool_call>
    or as bare JSON objects.
    """
    patterns = [
        # <tool_call>...</tool_call> tags
        r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
        # ```json ... ``` blocks containing tool call shapes
        r'```(?:json)?\s*(\{[^`]*?"name"\s*:\s*"web_(?:browse|search)"[^`]*?\})\s*```',
        # Bare JSON objects with "name" key matching our tools
        r'(\{"name"\s*:\s*"web_(?:browse|search)"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\})',
    ]

    results = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                obj = json.loads(match)
                name = obj.get("name", "")
                args = obj.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
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
        
        # Metrics tracking
        start_time = time.time()
        token_count = 0

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
                            token_count += 1 # Rough estimate: 1 chunk ~= 1 token
                            yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"

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
            # Emit metrics before done
            duration = time.time() - start_time
            if duration > 0 and token_count > 0:
                t_s = round(token_count / duration, 1)
                yield f"data: {json.dumps({'type': 'metrics', 'tokens_per_sec': t_s})}\n\n"
            
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

            result = await execute_tool(func_name, func_args)

            yield f"data: {json.dumps({'type': 'tool_result', 'tool': func_name, 'result': result})}\n\n"

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

        # Continue the loop — the model will now generate a response using the tool results

    # If we exhausted rounds
    yield f"data: {json.dumps({'type': 'done'})}\n\n"
