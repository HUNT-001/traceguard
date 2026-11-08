"""Free-tier LLM client: Groq (OpenAI-compatible) and Gemini (REST). Disk cache, daily request
budget, exponential backoff on 429, temperature 0. No prices, no paid calls.

Interface: chat(system, messages, tools=None, json_mode=False) with Anthropic-style messages
(content blocks: text / tool_use / tool_result). Returns dict(content=[blocks], tokens_in,
tokens_out, latency_ms, cached). Keys come from GROQ_API_KEY / GEMINI_API_KEY.
Model names and limits live in results/providers.json (copied from provider docs)."""
import hashlib, json, os, time, datetime, urllib.request, urllib.error
from pathlib import Path


class BudgetReached(RuntimeError):
    pass


class QuotaExhausted(BudgetReached):
    pass


def _to_openai(system, messages, tools):
    out = [{"role": "system", "content": system}]
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            out.append({"role": m["role"], "content": c}); continue
        if m["role"] == "assistant":
            text = "".join(b.get("text", "") for b in c if b.get("type") == "text")
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                     for b in c if b.get("type") == "tool_use"]
            msg = {"role": "assistant", "content": text or None}
            if calls: msg["tool_calls"] = calls
            out.append(msg)
        else:
            for b in c:
                if b.get("type") == "tool_result":
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]})
                else:
                    out.append({"role": "user", "content": b.get("text", "")})
    body_tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                    "parameters": t["input_schema"]}} for t in tools or []]
    return out, body_tools


class LLM:
    def __init__(self, provider, model, daily_budget=500, cache_dir="results/llm_cache",
                 ledger="results/requests.json", transport=None, seed=0, today=None):
        assert provider in ("groq", "gemini")
        self.provider, self.model, self.budget, self.seed = provider, model, daily_budget, seed
        self.dir = Path(cache_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Path(ledger)
        self.today = today or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        self.calls, self.hits, self.retries_429 = 0, 0, 0
        self.transport = transport or self._http

    # ---- request accounting (per provider per UTC day; real limits reset per provider's own clock) ----
    def used(self):
        d = json.load(open(self.ledger)) if self.ledger.exists() else {}
        return d.get(self.today, {}).get(self.provider, 0)

    def _count(self):
        d = json.load(open(self.ledger)) if self.ledger.exists() else {}
        d.setdefault(self.today, {}).setdefault(self.provider, 0)
        d[self.today][self.provider] += 1
        json.dump(d, open(self.ledger, "w"))

    # ---- transport ----
    def _post(self, url, headers, body):
        req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json", "user-agent": "traceguard/0.1", **headers})
        for a in range(7):
            try:
                return json.load(urllib.request.urlopen(req, timeout=120))
            except urllib.error.HTTPError as e:
                eb = e.read().decode(errors='replace')
                if e.code == 429 and any(w in eb.lower() for w in ('per day', 'daily', 'tpd', 'rpd', 'perday')):
                    raise QuotaExhausted(f"{self.provider} daily quota exhausted: {eb[:300]}")
                if e.code in (429, 500, 503) and a < 6:
                    self.retries_429 += e.code == 429
                    ra = e.headers.get("retry-after")
                    time.sleep(min(90, float(ra)) if ra and ra.replace(".", "").isdigit() else min(60, 2 ** (a + 1)))
                    continue
                raise RuntimeError(f"{self.provider} HTTP {e.code}: {eb[:300]!r}")

    def _http(self, system, messages, tools, json_mode, max_tokens):
        if self.provider == "groq":
            key = os.environ.get("GROQ_API_KEY")
            if not key: raise RuntimeError("GROQ_API_KEY is not set")
            msgs, ot = _to_openai(system, messages, tools)
            body = dict(model=self.model, messages=msgs, temperature=0, seed=self.seed, max_tokens=max_tokens)
            if ot: body["tools"] = ot
            r = self._post("https://api.groq.com/openai/v1/chat/completions", {"authorization": f"Bearer {key}"}, body)
            m = r["choices"][0]["message"]; blocks = []
            if m.get("content"): blocks.append({"type": "text", "text": m["content"]})
            for c in m.get("tool_calls") or []:
                blocks.append({"type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                               "input": json.loads(c["function"]["arguments"] or "{}")})
            u = r.get("usage", {})
            return dict(content=blocks, tokens_in=u.get("prompt_tokens", 0), tokens_out=u.get("completion_tokens", 0))
        key = os.environ.get("GEMINI_API_KEY")
        if not key: raise RuntimeError("GEMINI_API_KEY is not set")
        text = "\n".join(m["content"] if isinstance(m["content"], str) else
                         "".join(b.get("text", "") for b in m["content"]) for m in messages)
        gc = dict(temperature=0, seed=self.seed, maxOutputTokens=max_tokens)  # temperature may be ignored by newest models
        if json_mode: gc["responseMimeType"] = "application/json"
        body = dict(systemInstruction={"parts": [{"text": system}]}, contents=[{"role": "user", "parts": [{"text": text}]}],
                    generationConfig=gc)
        r = self._post(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                       {"x-goog-api-key": key}, body)
        parts = (r.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        u = r.get("usageMetadata", {})
        return dict(content=[{"type": "text", "text": "".join(p.get("text", "") for p in parts)}],
                    tokens_in=u.get("promptTokenCount", 0), tokens_out=u.get("candidatesTokenCount", 0))

    def chat(self, system, messages, tools=None, json_mode=False, max_tokens=1024):
        key = json.dumps([self.provider, self.model, self.seed, system, messages, tools, json_mode, max_tokens], sort_keys=True)
        f = self.dir / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        if f.exists():
            self.hits += 1; r = json.load(open(f)); r["cached"] = True; return r
        if self.used() >= self.budget:
            raise BudgetReached(f"{self.provider}: daily request budget {self.budget} reached; resume after reset")
        t0 = time.time()
        r = self.transport(system, messages, tools, json_mode, max_tokens)
        r["latency_ms"] = (time.time() - t0) * 1000
        self._count(); self.calls += 1
        json.dump(r, open(f, "w")); r["cached"] = False
        return r
