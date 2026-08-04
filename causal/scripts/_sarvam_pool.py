#!/usr/bin/env python3
"""Shared: rotate the sarvam-105b judge across ALL keys in .env with failover, so a dead/quota'd key
does NOT NaN the whole run. `SARVAM_API_KEY` (key 1) is out of credits (HTTP 402); keys 2/3 work.
The stock `sv.call` uses a single global `sv.KEY` — call `install(p34.sv)` once after loading the judge
module to replace it with a key-rotating version (retire a key on HTTP 401/402/403, back off on 429/5xx).
"""
import json, time, threading, urllib.request, urllib.error

ENV = "$PROJECT_ROOT/.env"


def load_keys(env=ENV):
    keys = []
    for line in open(env):
        if line.startswith("SARVAM_API_KEY"):
            v = line.split("=", 1)[1].strip()
            if v:
                keys.append(v)
    return keys


class KeyPool:
    def __init__(self, keys):
        self.lock = threading.Lock(); self.alive = list(keys); self.i = 0

    def get(self):
        with self.lock:
            if not self.alive:
                return None
            k = self.alive[self.i % len(self.alive)]; self.i += 1; return k

    def mark_dead(self, k, why):
        with self.lock:
            if k in self.alive:
                self.alive.remove(k)
                print(f"  [sarvam key ...{k[-4:]} disabled: {why}]  alive={len(self.alive)}", flush=True)


def install(sv, max_tries=10):
    """Monkeypatch sv.call with a key-rotating version. Returns the KeyPool."""
    pool = KeyPool(load_keys())
    print(f"[sarvam-pool] {len(pool.alive)} keys loaded (...{', ...'.join(k[-4:] for k in pool.alive)})", flush=True)

    def call(msg):
        body = json.dumps({"model": sv.JUDGE_MODEL, "messages": [{"role": "user", "content": msg}],
                           "max_tokens": 3000, "temperature": 0.0}).encode()
        last = "noalive"
        for attempt in range(max_tries):
            k = pool.get()
            if k is None:
                return "ERR:NOKEYS"
            try:
                req = urllib.request.Request(sv.SARVAM_URL, data=body, headers={
                    "Content-Type": "application/json", "Authorization": f"Bearer {k}"})
                r = json.load(urllib.request.urlopen(req, timeout=180))
                return (r["choices"][0]["message"].get("content", "") or "").strip()
            except urllib.error.HTTPError as e:
                last = f"HTTP{e.code}"
                if e.code in (401, 402, 403):
                    pool.mark_dead(k, last); continue
                if e.code in (429, 500, 502, 503, 529):
                    time.sleep(min(30.0, 2.0 * (attempt + 1))); continue
                return f"ERR:{last}"
            except Exception as e:
                last = type(e).__name__
                time.sleep(min(15.0, 1.5 * (attempt + 1))); continue
        return f"ERR:{last}"

    sv.call = call
    sv.KEY = "POOL"          # keep truthy for any code that checks it
    return pool
