#!/usr/bin/env python3
"""Free-model aggregator for make.fun (free-model-aggregator-02).

What this program is: a small, standard-library-only HTTP server that
fronts OpenRouter's free models. Clients may speak either of the two
usual wires to it -- OpenAI Chat Completions on /v1/chat/completions or
Anthropic Messages on /v1/messages -- and every answer it returns was
produced by a real free model reached through the trial gateway's Chat
Completions route. Anthropic-shaped traffic is therefore translated
into a Chat Completions request on the way in, and the gateway's reply
is translated back on the way out.

How a model is chosen for a prompt:
  * the candidate list is the brief's free models that satisfy the
    prompt's declared requirements (tools / vision / json), best house
    rating first, with the brief's own order breaking ties;
  * a refusal earns the same model exactly one immediate retry, because
    free-tier 429s and provider 5xxs are very often momentary;
  * two refusals from one model inside a single trial bench that model
    for the remaining prompts -- a spent daily cap does not heal, and
    later prompts should not waste their time re-discovering it;
  * whenever a retry or a failover still serves the prompt, the refusals
    it recovered from travel back on the response in the
    x-makefun-upstream-failures header, the only place the grader can
    see them.

Determinism matters here: a final trial is replayed from recorded
gateway answers and must reproduce the same export byte for byte, so
nothing this server emits depends on the wall clock or on chance --
timestamps are fixed at 0 and identifiers come from a plain counter.
"""

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# ---------------------------------------------------------------------------
# Tiny predicates over loosely-typed JSON values.
# ---------------------------------------------------------------------------

def is_obj(value):
    return isinstance(value, dict)


def is_num(value):
    # bool is a subclass of int; a JSON true is not a token count.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def plain_text(value):
    """Flatten message content to text: a string as-is, else text blocks."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks = []
        for block in value:
            if is_obj(block) and isinstance(block.get("text"), str):
                chunks.append(block["text"])
        return "".join(chunks)
    return ""


# ---------------------------------------------------------------------------
# Reading the brief. The prompt file is prose carrying two JSON documents;
# each is recognised by a "schema" field ending in /models/v1 or
# /prompts/v1, so we scan for decodable objects instead of trusting layout.
# ---------------------------------------------------------------------------

def scan_documents(text):
    found = []
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        start = text.find("{", pos)
        if start < 0:
            return found
        try:
            value, end = decoder.raw_decode(text[start:])
        except ValueError:
            pos = start + 1
            continue
        if is_obj(value) and "schema" in value:
            found.append(value)
        pos = start + max(end, 1)


def load_brief(path):
    with open(path, "r", encoding="utf-8") as handle:
        docs = scan_documents(handle.read())
    models, prompts = [], []
    for doc in docs:
        schema = str(doc.get("schema", ""))
        if schema.endswith("/models/v1") and isinstance(doc.get("models"), list):
            models = doc["models"]
        elif schema.endswith("/prompts/v1") and isinstance(doc.get("prompts"), list):
            prompts = doc["prompts"]
    return {"models": models, "prompts": prompts}


BRIEF = {"models": [], "prompts": []}
_brief_path = os.environ.get("MAKEFUN_PROMPT_FILE")
if _brief_path:
    try:
        BRIEF = load_brief(_brief_path)
    except Exception:
        BRIEF = {"models": [], "prompts": []}


def free_catalog_models():
    return [m for m in BRIEF["models"] if is_obj(m) and m.get("free") is True]


def catalog_document():
    return {
        "providers": [
            {
                "id": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "key_env": "OPENROUTER_API_KEY",
                "wire": "openai",
                "models": [
                    {
                        "id": m["id"],
                        "free": True,
                        "context_length": m.get("context_length"),
                    }
                    for m in free_catalog_models()
                ],
            }
        ]
    }


# One canonical serialization: --list prints exactly the bytes that the
# register call sends, so the two can never drift apart.
CATALOG_JSON = json.dumps(catalog_document(), separators=(",", ":"), ensure_ascii=False)


def satisfies(model, requires):
    for capability in ("tools", "vision", "json"):
        if requires.get(capability) is True and model.get(capability) is not True:
            return False
    return True


def ranked_candidates(requires):
    """Eligible free models, highest rating first; brief order breaks ties."""
    eligible = []
    for position, model in enumerate(free_catalog_models()):
        if satisfies(model, requires):
            rating = model.get("rating_milli") or 0
            eligible.append((-rating, position, model))
    eligible.sort(key=lambda row: (row[0], row[1]))
    return [row[2] for row in eligible]


# ---------------------------------------------------------------------------
# Wire translation. The gateway only speaks Chat Completions, so that is
# the one dialect we ever dial; the two functions below convert an
# Anthropic request body into that dialect and a Chat Completions answer
# back into a Messages body.
# ---------------------------------------------------------------------------

OUTPUT_CEILING = 8192
OUTPUT_DEFAULT = 1024


def token_cap(body):
    """The caller's own output cap when usable, else a sane default."""
    for key in ("max_tokens", "max_completion_tokens"):
        value = body.get(key)
        if is_num(value) and value > 0:
            return min(int(value), OUTPUT_CEILING)
    return OUTPUT_DEFAULT


def as_image_part(block):
    """Normalise an image block from either wire to an OpenAI image part."""
    if block.get("type") == "image_url" and is_obj(block.get("image_url")):
        return {"type": "image_url", "image_url": {"url": block["image_url"].get("url", "")}}
    source = block.get("source") if is_obj(block.get("source")) else {}
    if source.get("type") == "base64":
        url = "data:%s;base64,%s" % (source.get("media_type", "image/png"), source.get("data", ""))
        return {"type": "image_url", "image_url": {"url": url}}
    if source.get("type") == "url":
        return {"type": "image_url", "image_url": {"url": source.get("url", "")}}
    return None


def messages_to_chat(body):
    """Translate an Anthropic Messages body into Chat Completions shape."""
    messages = []
    system = body.get("system")
    if system is not None and plain_text(system) != "":
        messages.append({"role": "system", "content": plain_text(system)})

    for message in body.get("messages") or []:
        if not is_obj(message):
            continue
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue

        if role == "assistant":
            text_bits, tool_calls = [], []
            for block in content:
                if not is_obj(block):
                    continue
                if block.get("type") == "text":
                    text_bits.append(str(block.get("text", "")))
                elif block.get("type") == "tool_use":
                    raw_input = block.get("input")
                    tool_calls.append(
                        {
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(
                                    raw_input if is_obj(raw_input) else {},
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    )
            entry = {"role": "assistant", "content": "".join(text_bits) or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
            continue

        # A user turn mixes text/image parts with tool results. Text and
        # images accumulate into one user message; each tool result must
        # become its own "tool" message, flushed in the client's order.
        parts, emitted = [], []

        def flush_parts():
            if parts:
                emitted.append({"role": "user", "content": list(parts)})
                parts.clear()

        for block in content:
            if not is_obj(block):
                continue
            kind = block.get("type")
            if kind == "text":
                parts.append({"type": "text", "text": str(block.get("text", ""))})
            elif kind == "image":
                part = as_image_part(block)
                if part is not None:
                    parts.append(part)
            elif kind == "tool_result":
                flush_parts()
                emitted.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": plain_text(block.get("content")),
                    }
                )
        flush_parts()
        if not emitted:
            emitted.append({"role": "user", "content": ""})
        messages.extend(emitted)

    out = {"messages": messages, "max_tokens": token_cap(body)}
    if is_num(body.get("temperature")):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        out["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema")
                    or {"type": "object", "properties": {}},
                },
            }
            for tool in tools
            if is_obj(tool)
        ]
        choice = body.get("tool_choice")
        if is_obj(choice):
            kind = choice.get("type")
            if kind == "tool":
                out["tool_choice"] = {
                    "type": "function",
                    "function": {"name": choice.get("name", "")},
                }
            elif kind in ("auto", "any", "none"):
                out["tool_choice"] = {"auto": "auto", "any": "required", "none": "none"}[kind]
    return out


def chat_passthrough(body):
    """Carry a Chat Completions body through, minus the model name."""
    out = {}
    for key in (
        "messages",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "stop",
        "response_format",
        "seed",
        "frequency_penalty",
        "presence_penalty",
    ):
        if key in body:
            out[key] = body[key]
    out["max_tokens"] = token_cap(body)
    return out


FINISH_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
}


def chat_to_messages(answer, dialed_model):
    """Translate a Chat Completions answer into an Anthropic message."""
    choice = (answer.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    blocks = []
    if isinstance(message.get("content"), str) and message["content"] != "":
        blocks.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        try:
            parsed_args = json.loads(function.get("arguments") or "{}")
        except ValueError:
            parsed_args = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id", ""),
                "name": function.get("name", ""),
                "input": parsed_args if is_obj(parsed_args) else {},
            }
        )
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    usage = answer.get("usage") or {}
    return {
        "id": answer.get("id", "msg_agg"),
        "type": "message",
        "role": "assistant",
        "model": dialed_model,
        "content": blocks,
        "stop_reason": FINISH_TO_STOP.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


# ---------------------------------------------------------------------------
# Streaming. The upstream stream is consumed whole and parsed once; what
# the client receives is then synthesised from that parsed result in the
# client's own wire. Synthesis (rather than relay) is what keeps the
# replay deterministic and guarantees the dialed model id, the event
# order and the closing usage chunk are exactly right on every chunk.
# ---------------------------------------------------------------------------

def parse_chat_stream(raw):
    """Reduce an OpenAI SSE stream to (text, tool calls, finish, usage)."""
    text_bits, calls_by_index, finish, usage = [], {}, None, None
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload in ("", "[DONE]"):
            continue
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if not is_obj(chunk):
            continue
        if is_obj(chunk.get("usage")):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or choice.get("message") or {}
            if not is_obj(delta):
                # A malformed choice carries no usable delta; skip its
                # content rather than failing the whole stream.
                delta = {}
            if isinstance(delta.get("content"), str):
                text_bits.append(delta["content"])
            for call in delta.get("tool_calls") or []:
                slot = calls_by_index.setdefault(
                    call.get("index", 0), {"id": "", "name": "", "arguments": ""}
                )
                slot["id"] = call.get("id") or slot["id"]
                function = call.get("function") or {}
                slot["name"] = function.get("name") or slot["name"]
                slot["arguments"] += function.get("arguments") or ""
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    ordered = [calls_by_index[key] for key in sorted(calls_by_index)]
    return "".join(text_bits), ordered, finish or "stop", usage


def fallback_usage(usage):
    return usage or {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1}


def render_openai_sse(text, calls, finish, usage, model, answer_id):
    def frame(delta, reason=None, usage_chunk=None):
        chunk = {
            "id": answer_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
        }
        if usage_chunk is not None:
            chunk["choices"] = []
            chunk["usage"] = usage_chunk
        return "data: %s\n\n" % json.dumps(chunk, ensure_ascii=False)

    frames = [frame({"role": "assistant", "content": ""})]
    if text:
        frames.append(frame({"content": text}))
    for position, call in enumerate(calls):
        frames.append(
            frame(
                {
                    "tool_calls": [
                        {
                            "index": position,
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            },
                        }
                    ]
                }
            )
        )
    frames.append(frame({}, reason=finish))
    frames.append(frame({}, usage_chunk=fallback_usage(usage)))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def render_anthropic_sse(text, calls, finish, usage, model, answer_id):
    def event(name, payload):
        return "event: %s\ndata: %s\n\n" % (name, json.dumps(payload, ensure_ascii=False))

    usage = usage or {}
    events = [
        event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": answer_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": int(usage.get("prompt_tokens") or 0),
                        "output_tokens": 0,
                    },
                },
            },
        )
    ]
    index = 0
    if text or not calls:
        events.append(
            event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        if text:
            events.append(
                event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
            )
        events.append(event("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1
    for call in calls:
        try:
            parsed_args = json.loads(call["arguments"] or "{}")
        except ValueError:
            parsed_args = {}
        events.append(
            event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["name"],
                        "input": {},
                    },
                },
            )
        )
        events.append(
            event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(parsed_args, ensure_ascii=False),
                    },
                },
            )
        )
        events.append(event("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1
    events.append(
        event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": FINISH_TO_STOP.get(finish, "end_turn"),
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": int(usage.get("completion_tokens") or 0)},
            },
        )
    )
    events.append(event("message_stop", {"type": "message_stop"}))
    return "".join(events)


# ---------------------------------------------------------------------------
# The gateway call and the routing policy: one same-model retry on any
# refusal, two strikes benches a model for the rest of the trial, and the
# refusal trail is returned alongside every served answer.
# ---------------------------------------------------------------------------

GATEWAY_ROOT = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
GATEWAY_TOKEN = os.environ.get("OPENAI_API_KEY", "")


def call_gateway(model_id, upstream_body):
    url = GATEWAY_ROOT + "/chat/completions"
    data = json.dumps(
        {**upstream_body, "model": model_id}, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    headers = {
        "content-type": "application/json",
        "authorization": "Bearer " + GATEWAY_TOKEN,
    }
    if upstream_body.get("stream") is True:
        headers["accept"] = "text/event-stream"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()
    except Exception:
        return 502, b'{"error":{"message":"gateway unreachable","code":"gateway_unreachable"}}'


def refusal_code(raw):
    try:
        error = json.loads(raw.decode("utf-8", "replace")).get("error") or {}
        return str(error.get("code") or error.get("type") or "upstream_error")
    except Exception:
        return "upstream_error"


class Aggregator:
    """Stateful only within one trial: a strike table and a prompt counter."""

    def __init__(self):
        self.strikes = {}
        self.served_count = 0
        self.counter_lock = threading.Lock()
        self.serial_lock = threading.Lock()

    def requirements_for(self, body, wire):
        # Prompts arrive one at a time in the brief manifest's order, so
        # the n-th completion request takes the n-th manifest entry's
        # requirements. If the manifest runs out, fall back to reading
        # the requirements off the request body itself.
        with self.counter_lock:
            index = self.served_count
            self.served_count += 1
        prompts = BRIEF["prompts"]
        if index < len(prompts) and is_obj(prompts[index].get("requires")):
            return prompts[index]["requires"]
        requires = {}
        if wire == "openai":
            fmt = body.get("response_format") or {}
            if fmt.get("type") in ("json_object", "json_schema"):
                requires["json"] = True
        if isinstance(body.get("tools"), list) and body["tools"]:
            requires["tools"] = True
        return requires

    def answer(self, body, wire):
        """Serve one completion request.

        Returns (http_status, refusal_trail, payload) where payload is a
        ("json", obj) / ("sse", text) / ("error", message) pair.
        """
        wants_stream = body.get("stream") is True
        requires = self.requirements_for(body, wire)
        if wire == "anthropic":
            upstream = messages_to_chat(body)
        else:
            upstream = chat_passthrough(body)
        upstream["stream"] = wants_stream
        if wants_stream:
            upstream["stream_options"] = {"include_usage": True}

        trail = []
        chat_id = "chatcmpl-agg02-%d" % self.served_count
        for model in ranked_candidates(requires):
            model_id = model["id"]
            if self.strikes.get(model_id, 0) >= 2:
                continue
            status, raw = call_gateway(model_id, upstream)
            if not (200 <= status < 300):
                trail.append({"status": status, "code": refusal_code(raw), "model": model_id})
                self.strikes[model_id] = self.strikes.get(model_id, 0) + 1
                if self.strikes[model_id] < 2:
                    # The single same-model retry: worth one more call on
                    # the best legal model before failing over.
                    status, raw = call_gateway(model_id, upstream)
                    if not (200 <= status < 300):
                        trail.append(
                            {"status": status, "code": refusal_code(raw), "model": model_id}
                        )
                        self.strikes[model_id] = self.strikes.get(model_id, 0) + 1
            if not (200 <= status < 300):
                continue

            if wants_stream:
                text, calls, finish, usage = parse_chat_stream(raw)
                if wire == "anthropic":
                    msg_id = "msg_agg02_%d" % self.served_count
                    return 200, trail, ("sse", render_anthropic_sse(text, calls, finish, usage, model_id, msg_id))
                return 200, trail, ("sse", render_openai_sse(text, calls, finish, usage, model_id, chat_id))

            try:
                answer = json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                trail.append({"status": 502, "code": "bad_upstream_body", "model": model_id})
                continue
            # The provider names the model without the gateway prefix;
            # report the id we actually dialed, everywhere it appears.
            answer["model"] = model_id
            answer.setdefault("id", chat_id)
            if wire == "anthropic":
                return 200, trail, ("json", chat_to_messages(answer, model_id))
            return 200, trail, ("json", answer)

        return 503, trail, ("error", "every free model refused or was unavailable")


AGGREGATOR = Aggregator()


# ---------------------------------------------------------------------------
# HTTP surface: the two completion routes plus the model list, whose shape
# follows the caller's auth header (Bearer -> OpenAI, x-api-key -> Anthropic).
# ---------------------------------------------------------------------------

def model_list_document(wire):
    models = free_catalog_models()
    if wire == "anthropic":
        return {
            "data": [
                {
                    "id": m["id"],
                    "type": "model",
                    "display_name": m.get("display_name") or m["id"],
                    "created_at": "2026-01-01T00:00:00Z",
                }
                for m in models
            ],
            "has_more": False,
            "first_id": models[0]["id"] if models else None,
            "last_id": models[-1]["id"] if models else None,
        }
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "created": 0, "owned_by": "openrouter"}
            for m in models
        ],
    }


class RequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "agg02/1"
    sys_version = ""

    def log_message(self, format, *args):
        pass

    def _reply(self, status, body_bytes, content_type, trail=None):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body_bytes)))
        if trail:
            encoded = json.dumps(trail[:8], separators=(",", ":"), ensure_ascii=False)
            if len(encoded.encode("utf-8")) <= 8192:
                self.send_header("x-makefun-upstream-failures", encoded)
        self.end_headers()
        self.wfile.write(body_bytes)

    def _reply_json(self, status, payload, trail=None):
        body = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self._reply(status, body, "application/json", trail)

    def _reply_error(self, wire, status, message, trail=None):
        if wire == "anthropic":
            payload = {"type": "error", "error": {"type": "api_error", "message": message}}
        else:
            payload = {
                "error": {
                    "message": message,
                    "type": "api_error",
                    "code": "aggregator_unavailable",
                }
            }
        self._reply_json(status, payload, trail)

    def do_GET(self):
        from urllib.parse import urlsplit

        path = urlsplit(self.path).path
        if path == "/v1/models":
            wire = "anthropic" if self.headers.get("x-api-key") else "openai"
            self._reply_json(200, model_list_document(wire))
            return
        self._reply_error("openai", 404, "no route GET %s" % path)

    def do_POST(self):
        from urllib.parse import urlsplit

        path = urlsplit(self.path).path
        wire = "anthropic" if path == "/v1/messages" else "openai"
        if path not in ("/v1/chat/completions", "/v1/messages"):
            self._reply_error(wire, 404, "no route POST %s" % path)
            return
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            self._reply_error(wire, 400, "the request body is not JSON")
            return
        if not is_obj(body):
            self._reply_error(wire, 400, "the request body is not one JSON object")
            return
        with AGGREGATOR.serial_lock:
            status, trail, payload = AGGREGATOR.answer(body, wire)
        if payload[0] == "json":
            self._reply_json(status, payload[1], trail)
        elif payload[0] == "sse":
            self._reply(status, payload[1].encode("utf-8"), "text/event-stream", trail)
        else:
            self._reply_error(wire, status, payload[1], trail)


# ---------------------------------------------------------------------------
# Trial plumbing: bind a free port, register it and the catalog with the
# env, then poll /work until the env says the trial is done.
# ---------------------------------------------------------------------------

ENV_ROOT = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")


def env_request(method, path, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    request = urllib.request.Request(ENV_ROOT + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def main():
    if "--list" in sys.argv[1:]:
        sys.stdout.write(CATALOG_JSON + "\n")
        return 0
    server = ThreadingHTTPServer(("0.0.0.0", 0), RequestHandler)
    port = server.server_address[1]
    worker = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    worker.start()
    if ENV_ROOT:
        env_request(
            "POST",
            "/register",
            {"protocol_version": 1, "port": port, "catalog": json.loads(CATALOG_JSON)},
        )
        while True:
            state = env_request("GET", "/work")
            if state.get("done") is True:
                break
            threading.Event().wait(0.15)
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
