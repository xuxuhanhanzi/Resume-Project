"""Freeze Wikipedia pages referenced by a small FRAMES smoke set."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
import time
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

REVISION = "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"
BLOCK_TAGS = {
    "article",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "p",
    "section",
    "table",
    "td",
    "th",
    "tr",
}


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "svg"}:
            self.skip_depth += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg"} and self.skip_depth:
            self.skip_depth -= 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        lines = (re.sub(r"\s+", " ", line).strip() for line in "".join(self.parts).splitlines())
        return "\n".join(line for line in lines if line)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _task_urls(dataset_path: Path, count: int) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    with dataset_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        for index, row in enumerate(reader):
            if index >= count:
                break
            parsed = ast.literal_eval(str(row["wiki_links"]))
            if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
                raise ValueError("FRAMES wiki_links is malformed")
            # Strip each link to mirror FramesAdapter._parse_links(), which the
            # oracle executor uses to resolve asset URIs against this corpus.
            result[f"frames-test-{index:04d}"] = {item.strip() for item in parsed if item.strip()}
    if len(result) != count:
        raise ValueError(f"requested {count} tasks but found {len(result)}")
    return result


def _encode_url(url: str) -> str:
    parts = urlsplit(url)
    path = quote(parts.path, safe="/")
    query = quote(parts.query, safe="=&")
    fragment = quote(parts.fragment, safe="")
    return urlunsplit((parts.scheme, parts.netloc, path, query, fragment))


def _download(url: str, *, attempts: int = 4) -> tuple[bytes, dict[str, str]]:
    safe_url = _encode_url(url)
    request = Request(safe_url, headers={"User-Agent": "RepoPilotResearch/0.1 (benchmark corpus)"})
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, timeout=120) as response:  # noqa: S310
                return response.read(), {
                    "content_type": response.headers.get("Content-Type", ""),
                    "etag": response.headers.get("ETag", ""),
                    "last_modified": response.headers.get("Last-Modified", ""),
                }
        except (HTTPError, URLError, TimeoutError):
            if attempt == attempts:
                break
            time.sleep(float(attempt))
        except (UnicodeEncodeError, ValueError):
            # urlopen cannot transmit this URL (non-ASCII); defer to curl immediately.
            break
    completed = subprocess.run(  # noqa: S603
        [
            "curl",
            "--location",
            "--retry",
            "8",
            "--retry-all-errors",
            "--max-time",
            "180",
            "--silent",
            "--show-error",
            "--user-agent",
            "RepoPilotResearch/0.1 (benchmark corpus)",
            safe_url,
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout, {"content_type": "text/html"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10, choices=range(1, 201))
    arguments = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    dataset_path = project / "evaluation" / "benchmarks" / "data" / "frames" / REVISION / "test.tsv"
    corpus_root = (
        project
        / "evaluation"
        / "benchmarks"
        / "corpora"
        / "frames"
        / REVISION
        / f"smoke{arguments.count}"
    )
    documents_root = corpus_root / "documents"
    documents_root.mkdir(parents=True, exist_ok=True)
    task_urls = _task_urls(dataset_path, arguments.count)
    urls = sorted(set().union(*task_urls.values()))
    documents: list[dict[str, object]] = []
    for index, url in enumerate(urls):
        name = f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.txt"
        path = documents_root / name
        metadata_path = documents_root / f"{name}.json"
        if path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            html, headers = _download(url)
            parser_instance = _VisibleTextParser()
            parser_instance.feed(html.decode("utf-8", errors="replace"))
            text = parser_instance.text() + "\n"
            path.write_text(text, encoding="utf-8")
            metadata = {
                "url": url,
                "title": unquote(urlsplit(url).path.rsplit("/", 1)[-1]).replace("_", " "),
                "html_sha256": _sha256(html),
                "text_sha256": _sha256(path.read_bytes()),
                "fetched_at": datetime.now(UTC).isoformat(),
                **headers,
            }
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        documents.append({**metadata, "path": path.relative_to(corpus_root).as_posix()})
        print(f"[{index + 1}/{len(urls)}] {url}")
    manifest = {
        "schema_version": 1,
        "dataset_revision": REVISION,
        "task_ids": sorted(task_urls),
        "task_urls": {task_id: sorted(task_urls[task_id]) for task_id in sorted(task_urls)},
        "documents": documents,
    }
    manifest_path = corpus_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), "documents": len(documents)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
