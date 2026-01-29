#!/usr/bin/env python3
"""
Download selected ImageNet-family datasets with optional auth and resume.
- Set IMAGENET_USER and IMAGENET_PASS for endpoints that require login.
- Use --dataset to choose items or --list to see available options.
"""
import argparse
import hashlib
import os
import pathlib
import re
import sys
from typing import Dict, Optional, TypedDict
from urllib.parse import urljoin, urlparse, parse_qs
import requests

# Known datasets. Add new entries if you have URLs and checksums.
class DatasetInfo(TypedDict):
    url: str
    md5: Optional[str]
    size_gb: Optional[float]
    needs_auth: bool
    desc: str


DATASETS: Dict[str, DatasetInfo] = {
    # "ilsvrc2012_train": {
    #     "url": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar",
    #     "md5": "1d675b47d978889d74fa0da5fadfb00e",
    #     "size_gb": 138.0,
    #     "needs_auth": True,
    #     "desc": "ILSVRC2012 classification train split (1k classes).",
    # },
    # "ilsvrc2012_val": {
    #     "url": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
    #     "md5": "29b22e2961454d5413ddabcf34fc5622",
    #     "size_gb": 6.3,
    #     "needs_auth": True,
    #     "desc": "ILSVRC2012 classification validation split (1k classes).",
    # },
    # "tiny_imagenet_200": {
    #     "url": "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    #     "md5": None,  # checksum not provided; set if you have it.
    #     "size_gb": 0.24,
    #     "needs_auth": False,
    #     "desc": "Tiny ImageNet (200 classes, 64x64), useful for quick experiments.",
    # },
}

DEFAULT_DATASETS = ["ilsvrc2012_train", "ilsvrc2012_val"]

CHUNK = 1024 * 1024  # 1 MB


def md5sum(path: pathlib.Path, chunk: int = CHUNK) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def human(x: float) -> str:
    return f"{x:.1f} GB"


def _basename_from_url(raw_url: str) -> Optional[str]:
    parsed = urlparse(raw_url)
    path_name = pathlib.Path(parsed.path).name
    if path_name:
        return path_name
    qs = parse_qs(parsed.query)
    for val in qs.values():
        for candidate in val:
            fname = pathlib.Path(candidate).name
            if fname:
                return fname
    return None


def scrape_index(index_url: str, user: Optional[str], pwd: Optional[str], cookie: Optional[str], dump_html: Optional[pathlib.Path], md5_window: int) -> Dict[str, DatasetInfo]:
    """Best-effort scrape of the ImageNet download page to pull URLs and MD5.

    Strategy:
    - Grab full HTML.
    - Find all href positions (quoted or unquoted).
    - Find all MD5 hashes and assign the nearest hash within a small window to each href.
    - Keep links that point to tar/zip/tgz/npz (including query strings).
    """

    session = requests.Session()
    if cookie:
        session.headers.update({"Cookie": cookie})
    # Some endpoints still honor basic auth on the same host.
    if user and pwd:
        session.auth = (user, pwd)
    session.headers.setdefault("User-Agent", "Mozilla/5.0 (scraper for research; contact if issues)")

    r = session.get(index_url)
    r.raise_for_status()
    html = r.text

    if dump_html:
        dump_html.parent.mkdir(parents=True, exist_ok=True)
        dump_html.write_text(html, encoding="utf-8")

    if r.url != index_url and any(tok in r.url.lower() for tok in ("login", "access", "auth")):
        print(f"Warn: fetched was redirected to {r.url}. Page likely requires session cookie (e.g., PHPSESSID=...).")

    href_pattern = re.compile(r"href\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^\s>]+))", re.IGNORECASE)
    md5_pattern = re.compile(r"([0-9a-fA-F]{32})")

    anchors = []  # (href, position)
    for m in href_pattern.finditer(html):
        href = m.group(1) or m.group(2) or m.group(3)
        if not href:
            continue
        anchors.append((href, m.start()))

    md5_hits = [(m.group(1).lower(), m.start()) for m in md5_pattern.finditer(html)]

    def nearest_md5(pos: int, window: int = md5_window) -> Optional[str]:
        best = None
        best_d = window + 1
        for md5, md5_pos in md5_hits:
            d = abs(md5_pos - pos)
            if d < best_d and d <= window:
                best = md5
                best_d = d
        return best

    scraped: Dict[str, DatasetInfo] = {}
    for href, pos in anchors:
        if not any(ext in href for ext in (".tar", ".zip", ".tgz", ".npz")):
            continue
        full_url = urljoin(index_url, href)
        name = _basename_from_url(full_url)
        if not name:
            continue
        key = pathlib.Path(name).stem
        if key in scraped:
            continue
        scraped[key] = {
            "url": full_url,
            "md5": nearest_md5(pos),
            "size_gb": None,
            "needs_auth": True,
            "desc": f"Auto-discovered from {index_url}",
        }

    if not scraped:
        print("Warn: scraped 0 downloadable links; page may require authenticated cookie. Set IMAGENET_COOKIE='PHPSESSID=...'. Use --dump-html to inspect.")
    return scraped


def download_file(url: str, dest: pathlib.Path, md5: Optional[str], needs_auth: bool, user: Optional[str], pwd: Optional[str]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    pos = dest.stat().st_size if dest.exists() else 0
    if pos:
        headers["Range"] = f"bytes={pos}-"

    with requests.Session() as s:
        if needs_auth:
            if not user or not pwd:
                raise SystemExit("This dataset needs credentials. Set IMAGENET_USER and IMAGENET_PASS or pass --user/--pass.")
            s.auth = (user, pwd)

        head = s.head(url, allow_redirects=True)
        head.raise_for_status()
        total = int(head.headers.get("Content-Length", "0"))

        if total and pos >= total:
            print(f"{dest.name} already at {pos/1e6:.1f} MB (>= server {total/1e6:.1f} MB); verifying instead of downloading...")
        else:
            print(f"Downloading {dest.name} ({total/1e9:.1f} GB) ... resume from {pos/1e6:.1f} MB")

        def fetch(start_pos: int, use_range: bool = True) -> None:
            nonlocal headers
            if not use_range and "Range" in headers:
                headers = {k: v for k, v in headers.items() if k.lower() != "range"}
            with s.get(url, headers=headers, stream=True) as r:
                if r.status_code == 416:
                    raise requests.HTTPError("416", response=r)
                r.raise_for_status()
                mode = "ab" if start_pos else "wb"
                downloaded = start_pos
                last_report = downloaded
                report_step = max(int(total * 0.01), 8 * CHUNK) if total else 8 * CHUNK
                with open(dest, mode) as f:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded - last_report >= report_step:
                            if total:
                                pct = downloaded * 100 / total
                                msg = f"\r{dest.name}: {downloaded/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)"
                            else:
                                msg = f"\r{dest.name}: {downloaded/1e6:.1f} MB"
                            print(msg, end="", flush=True)
                            last_report = downloaded
                if total:
                    print(f"\r{dest.name}: {downloaded/1e6:.1f}/{total/1e6:.1f} MB (100%)")
                else:
                    print(f"\r{dest.name}: {downloaded/1e6:.1f} MB")

        if total and pos >= total:
            pass  # go straight to md5 verify below
        else:
            try:
                fetch(pos, use_range=bool(pos))
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 416:
                    print("Server returned 416 (range not satisfiable); restarting download from scratch...")
                    dest.unlink(missing_ok=True)
                    fetch(0, use_range=False)
                else:
                    raise

    if md5:
        print(f"Verifying md5 for {dest.name} ...")
        got = md5sum(dest)
        if got != md5:
            raise SystemExit(f"MD5 mismatch for {dest.name}: got {got}, expected {md5}")
        print("MD5 OK")


def list_datasets(avail: Dict[str, DatasetInfo]) -> None:
    print("Available datasets:")
    for k, v in avail.items():
        size = v.get("size_gb")
        size_txt = human(size) if isinstance(size, (int, float)) else "n/a"
        print(f"- {k}: {v.get('desc')} (size~{size_txt}, auth={'yes' if v.get('needs_auth') else 'no'}, md5={v.get('md5') or 'n/a'})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download ImageNet-related datasets with resume and md5.")
    p.add_argument("--dataset", "-d", nargs="+", default=DEFAULT_DATASETS, help="Datasets to download; names come from built-ins or scraped index")
    p.add_argument(
        "--out",
        "--save-dir",
        dest="out",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("IMAGENET_OUT", "imagenet_data")),
        help="Output directory (can also set IMAGENET_OUT)",
    )
    p.add_argument("--list", action="store_true", help="List available datasets and exit")
    p.add_argument("--index-url", default="https://image-net.org/download-images.php", help="Index page to scrape for dataset links")
    p.add_argument("--fetch-index", action="store_true", help="Scrape index page to extend available datasets")
    p.add_argument("--md5-window", type=int, default=120, help="Max character distance to associate MD5 with a link when scraping (smaller reduces false matches)")
    p.add_argument("--user", dest="user", default=os.environ.get("IMAGENET_USER"), help="Username/email for protected endpoints")
    p.add_argument("--pass", dest="pwd", default=os.environ.get("IMAGENET_PASS"), help="Password for protected endpoints")
    p.add_argument("--cookie", default=os.environ.get("IMAGENET_COOKIE"), help="Cookie header value, e.g., PHPSESSID=...; used for scraping/download when login is cookie-based")
    p.add_argument("--dump-html", type=pathlib.Path, help="Debug: save fetched index HTML for inspection")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out = args.out.expanduser()

    available: Dict[str, DatasetInfo] = dict(DATASETS)

    if args.fetch_index or args.list:
        try:
            scraped = scrape_index(args.index_url, args.user, args.pwd, args.cookie, args.dump_html, args.md5_window)
            available.update(scraped)
        except Exception as e:  # keep listing even if scrape fails
            print(f"Warn: failed to scrape index ({e}). Using built-ins only.")

    if args.list:
        list_datasets(available)
        return

    for name in args.dataset:
        if name not in available:
            raise SystemExit(f"Dataset '{name}' not found. Use --list with --fetch-index to see options.")

    for name in args.dataset:
        info = available[name]
        url = info["url"]
        dest = args.out / pathlib.Path(url).name
        download_file(
            url=url,
            dest=dest,
            md5=info.get("md5"),
            needs_auth=bool(info.get("needs_auth")),
            user=args.user,
            pwd=args.pwd,
        )
    print("All done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
