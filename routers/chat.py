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

    # Auto-detect images and run chart-recognizer
    messages, chart_event = _preprocess_images(messages)

    async def _generate():
        # If chart analysis was performed, emit it as an event first
        # If chart analysis was performed, emit it as an event first
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

        logger.info("[CHAT] No chart event. Proceeding to LLM.")
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


def _preprocess_images(messages: list) -> tuple[list, dict | None]:
    """
    Scan messages for images.  If found:
      1. Run through YOLOv8 chart pattern detector.
      2. Inject detected patterns into a system prompt for the LLM.
      3. Keep the image in the message so vision models can see it.
    Returns (modified_messages, chart_event_or_None).
    """
    chart_event = None

    # Only check the LATEST message for charts to avoid persistent short-circuiting
    if messages:
        msg = messages[-1]
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                # Look for an image_url part
                image_data = None
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url:
                            image_data = url
                            break

                if image_data:
                    # Run YOLOv8 chart pattern detection
                    patterns = []
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
                        logger.info(f"[CHART] Detected {len(patterns)} patterns: {patterns}")
                    except Exception as e:
                        logger.error(f"[CHART] Detection failed: {e}")

                    # Build pattern context for the LLM
                    if patterns:
                        pattern_lines = []
                        for p in patterns:
                            # Generate location description from bbox_norm
                            loc = ""
                            bn = p.get('bbox_norm')
                            if bn:
                                cx = (bn[0] + bn[2]) / 2
                                cy = (bn[1] + bn[3]) / 2
                                w = bn[2] - bn[0]
                                h = bn[3] - bn[1]
                                
                                h_pos = "left" if cx < 0.33 else "right" if cx > 0.66 else "center"
                                v_pos = "top" if cy < 0.33 else "bottom" if cy > 0.66 else "middle"
                                
                                if w > 0.8: h_pos = "full width"
                                if h > 0.8: v_pos = "full height"
                                
                                loc = f"[{v_pos}-{h_pos}]"

                            pattern_lines.append(f"- {p['label']} ({p['probability']}% confidence) {loc}")
                        pattern_text = "\n".join(pattern_lines)

                        chart_system = {
                            "role": "system",
                            "content": (
                                "You are a professional Technical Analyst AI. \n"
                                "The user has provided raw pattern data from a specialized chart analysis tool.\n"
                                "Detected Patterns:\n"
                                f"{pattern_text}\n\n"
                                "Your Task:\n"
                                "1. Analyze these patterns as FACTUAL DATA points.\n"
                                "2. Provide trading recommendations based SOLELY on standard technical analysis theory for these patterns.\n"
                                "3. Do NOT mention that you cannot see the image. Assume the patterns are correct.\n"
                                "5. Discuss the probability and typical outcomes for these setups (e.g. M-Head implies bearish reversal)."
                            ),
                        }
                    else:
                        chart_system = {
                            "role": "system",
                            "content": (
                                "The user uploaded a trading chart, but the AI pattern detector found NO specific patterns.\n"
                                "Since vision analysis is disabled, you cannot see the chart.\n"
                                "Provide general trading advice or ask the user to describe the chart."
                            ),
                        }

                    # STRIP THE IMAGE so the LLM doesn't hallucinate or waste compute
                    # We replace the complex content list with just the text part
                    user_text = ""
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            user_text += part.get("text", "")
                    
                    # Update the user message to be text-only
                    msg["content"] = user_text if user_text else "Analyze this chart based on the detected patterns."

                    # Insert chart system prompt right before the user message
                    idx = messages.index(msg)
                    messages.insert(idx, chart_system)

    return messages, chart_event


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
