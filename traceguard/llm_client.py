"""Minimal Anthropic Messages client: disk cache, hard spend cap, no extra deps.
Key comes from ANTHROPIC_API_KEY. Prices (USD per 1M tokens) must be supplied in
results/prices.json as {"model": {"in": x, "out": y}}. The cap is only as good as
these numbers, so the client refuses to call a model that has no price."""
import hashlib, json, os, time, urllib.request
from pathlib import Path


class SpendCapReached(RuntimeError):
    pass


class LLM:
    def __init__(self, cap_usd=15.0, cache_dir="results/llm_cache", prices_path="results/prices.json",
                 transport=None, ledger="results/spend.json"):
        self.cap, self.spent, self.calls, self.hits = cap_usd, 0.0, 0, 0
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Path(ledger)
        if self.ledger.exists():                      # the cap spans runs
            self.spent = json.load(open(self.ledger))["spent"]
        pp = Path(prices_path)
        self.prices = json.load(open(pp)) if pp.exists() else {}
        self.transport = transport or self._http

    def _http(self, body):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", json.dumps(body).encode(),
            {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        for a in range(4):
            try:
                return json.load(urllib.request.urlopen(req, timeout=120))
            except Exception:
                if a == 3:
                    raise
                time.sleep(2 ** a)

    def chat(self, model, system, messages, tools=None, max_tokens=1024, temperature=0.0):
        """Returns dict(content, tokens_in, tokens_out, latency_ms, cached, cost)."""
        body = dict(model=model, system=system, messages=messages, max_tokens=max_tokens,
                    temperature=temperature)
        if tools:
            body["tools"] = tools
        k = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        f = self.dir / f"{k}.json"
        if f.exists():
            self.hits += 1
            r = json.load(open(f))
            r["cached"], r["cost"] = True, 0.0
            return r
        if model not in self.prices:
            raise RuntimeError(f"no price for {model} in prices.json; refusing to spend blind")
        if self.spent >= self.cap:
            raise SpendCapReached(f"spend cap ${self.cap:.2f} reached (${self.spent:.2f} spent)")
        t0 = time.time()
        resp = self.transport(body)
        ms = (time.time() - t0) * 1000
        u, p = resp.get("usage", {}), self.prices[model]
        cost = (u.get("input_tokens", 0) * p["in"] + u.get("output_tokens", 0) * p["out"]) / 1e6
        self.spent += cost
        self.calls += 1
        json.dump({"spent": self.spent}, open(self.ledger, "w"))
        r = dict(content=resp["content"], tokens_in=u.get("input_tokens", 0),
                 tokens_out=u.get("output_tokens", 0), latency_ms=ms, stop=resp.get("stop_reason"))
        json.dump(r, open(f, "w"))
        r["cached"], r["cost"] = False, cost
        return r
