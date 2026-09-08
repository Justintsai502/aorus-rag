"""Download the product pages and cache them under data/raw/.

The pages sit behind Akamai Bot Manager, which needs two things before it
serves the page:

1. A complete set of browser headers (UA, Accept, Accept-Language, sec-ch-ua*,
   Sec-Fetch-*). A bare request gets "403 Access Denied".
2. **HTTP/2.** The same headers over HTTP/1.1 are still refused. Measured
   across six combinations, only "full browser headers AND h2" is served;
   the two conditions are independent (a pre-HTTP/2-era browser UA over
   HTTP/1.1 is also refused, so this is not mismatch detection). httpx
   defaults to HTTP/1.1, which is why ``httpx[http2]`` is a dependency and
   ``http2=True`` below is load-bearing rather than an optimisation.

No headless browser and no JavaScript execution is required: the spec table is
server-side rendered.

Cached HTML is committed to the repo so the whole pipeline can be rebuilt
offline, which also means the evaluation numbers stay reproducible even if
GIGABYTE changes the page.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .config import BROWSER_HEADERS, RAW_DIR, SOURCES

MANIFEST_NAME = "manifest.json"


def _raw_path(key: str) -> Path:
    return RAW_DIR / f"{key}.html"


def fetch_all(force: bool = False, timeout: float = 30.0) -> dict[str, Path]:
    """Download every source page. Returns {key: path}.

    Existing files are reused unless ``force`` is set, so re-running is cheap
    and does not hammer the origin.
    """
    import httpx  # imported lazily so `parse`/`index` work with no network stack

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    manifest_path = RAW_DIR / MANIFEST_NAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    paths: dict[str, Path] = {}
    with httpx.Client(
        headers=BROWSER_HEADERS,
        timeout=timeout,
        follow_redirects=True,
        http2=True,  # required, not an optimisation: HTTP/1.1 is refused (see docstring)
    ) as client:
        for key, url in SOURCES.items():
            path = _raw_path(key)
            if path.exists() and not force:
                paths[key] = path
                continue
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
            if "Access Denied" in html[:500]:
                raise RuntimeError(
                    f"{url} returned an Akamai block page. The browser header set "
                    "in config.BROWSER_HEADERS may need refreshing."
                )
            path.write_text(html, encoding="utf-8")
            manifest[key] = {
                "url": url,
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "bytes": len(html.encode("utf-8")),
                "status": resp.status_code,
            }
            paths[key] = path

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def load_cached(key: str) -> str:
    """Read one cached page, with a helpful error if `fetch` was never run."""
    path = _raw_path(key)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `uv run aorus-rag fetch` first "
            "(or restore the committed data/raw/ cache)."
        )
    return path.read_text(encoding="utf-8")


def cache_status() -> dict[str, dict]:
    """Report which pages are cached, for the CLI."""
    manifest_path = RAW_DIR / MANIFEST_NAME
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    out = {}
    for key, url in SOURCES.items():
        path = _raw_path(key)
        out[key] = {
            "url": url,
            "cached": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "fetched_at": manifest.get(key, {}).get("fetched_at"),
        }
    return out
