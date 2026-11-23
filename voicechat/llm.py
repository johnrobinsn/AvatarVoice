"""
LLM client: streams tokens from a local vLLM server over SSE (OpenAI-compatible).

Example:
    llm = VLLMClient(model="cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit")
    async for fragment in llm.stream("Hello, how are you?"):
        print(fragment, end="", flush=True)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import AsyncIterator, List

import httpx

log = logging.getLogger("af-llm")


# Short phrase boundaries we flush on, to minimize time-to-first-audio.
# Any of these inside the token stream commits the accumulated buffer.
_PHRASE_BOUNDARY = re.compile(r"(?<=[\.\!\?,:;])\s+|(?<=\n)")


class VLLMClient:
    def __init__(self, model: str,
                 base_url: str = "http://localhost:8000/v1",
                 system_prompt: str = "You are a helpful, warm voice-chat assistant. Reply briefly and conversationally.",
                 max_history: int = 8):
        self.model = model
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.history: List[dict] = []
        self.max_history = max_history
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))

    def reset(self):
        self.history.clear()

    async def close(self):
        await self._client.aclose()

    async def stream(self, user_text: str) -> AsyncIterator[str]:
        """Yield text fragments as they arrive from vLLM. Also records the turn in history."""
        messages = [{"role": "system", "content": self.system_prompt}] + self.history + [
            {"role": "user", "content": user_text}
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            # Only a runaway guard (~10 min of speech); the system prompt keeps replies short.
            "max_tokens": 2000,
            "temperature": 0.7,
            "top_p": 0.95,
        }
        t0 = time.perf_counter()
        full = []
        first_t = None
        async with self._client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as r:
            if r.status_code != 200:
                err = await r.aread()
                raise RuntimeError(f"LLM HTTP {r.status_code}: {err[:200]}")
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                d = json.loads(chunk)
                delta = d["choices"][0]["delta"].get("content", "")
                if delta:
                    if first_t is None:
                        first_t = time.perf_counter() - t0
                    full.append(delta)
                    yield delta
        text = "".join(full).strip()
        dt = time.perf_counter() - t0
        log.info(f"llm: TTFT={first_t*1000 if first_t else -1:.0f}ms total={dt*1000:.0f}ms "
                 f"chars={len(text)}  reply={text[:120]!r}")
        # Record in history
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": text})
        # Trim
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

    @staticmethod
    def phrase_batcher() -> callable:
        """Return a helper fn that accumulates streamed fragments and yields
        complete phrases (ending in punctuation) suitable for TTS synthesis.

        Usage:
            batch = VLLMClient.phrase_batcher()
            async for frag in llm.stream(...):
                for phrase in batch(frag):
                    await tts.enqueue(phrase)
            for phrase in batch.flush():
                await tts.enqueue(phrase)
        """
        buf = [""]
        class B:
            def __call__(self, frag: str):
                buf[0] += frag
                out = []
                while True:
                    m = _PHRASE_BOUNDARY.search(buf[0])
                    if not m:
                        break
                    end = m.end()
                    phrase = buf[0][:end].strip()
                    buf[0] = buf[0][end:]
                    if phrase:
                        out.append(phrase)
                return out
            def flush(self):
                remainder = buf[0].strip()
                buf[0] = ""
                return [remainder] if remainder else []
        return B()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    async def main():
        llm = VLLMClient(model="cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit")
        for turn in ["Hi!", "Tell me a 1-sentence joke.", "Thanks, bye."]:
            print(f"\n>>> user: {turn}")
            print("<<< assistant: ", end="", flush=True)
            phrases_delivered = []
            batch = VLLMClient.phrase_batcher()
            async for frag in llm.stream(turn):
                print(frag, end="", flush=True)
                for phrase in batch(frag):
                    phrases_delivered.append(phrase)
            for phrase in batch.flush():
                phrases_delivered.append(phrase)
            print(f"\n    phrases: {phrases_delivered}")
        await llm.close()
    asyncio.run(main())
