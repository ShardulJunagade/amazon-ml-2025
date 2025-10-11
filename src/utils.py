import os
from pathlib import Path
from functools import partial
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_thread_local = threading.local()


def _get_session() -> requests.Session:
    """Get or create a thread-local requests session with retries and pooling."""
    if getattr(_thread_local, "session", None) is None:
        session = requests.Session()
        retries = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _thread_local.session = session
    return _thread_local.session


def download_image(image_link, sample_id, savefolder):
    """
    Download a single image and save it as <sample_id>.jpg (or correct extension if found).
    """
    if not isinstance(image_link, str) or not image_link:
        return

    parsed = urlparse(image_link)
    if parsed.scheme not in ("http", "https"):
        print(f"Warning: Unsupported URL scheme - {sample_id} {image_link}")
        return

    # Try to infer file extension from URL (default: .jpg)
    ext = os.path.splitext(unquote(parsed.path))[-1].lower()
    if not ext or len(ext) > 5:
        ext = ".jpg"

    # Final filename = sample_id + extension
    filename = f"{sample_id}{ext}"
    image_save_path = os.path.join(savefolder, filename)

    # Skip if already downloaded and non-empty
    try:
        if os.path.exists(image_save_path) and os.path.getsize(image_save_path) > 0:
            return
    except OSError:
        pass

    session = _get_session()
    try:
        with session.get(image_link, stream=True, timeout=(5, 20)) as resp:
            resp.raise_for_status()
            os.makedirs(savefolder, exist_ok=True)
            with open(image_save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as ex:
        print(f"Warning: Not able to download - {sample_id} {image_link}\n{ex}")
        try:
            if os.path.exists(image_save_path) and os.path.getsize(image_save_path) == 0:
                os.remove(image_save_path)
        except OSError:
            pass


def download_images(image_links, sample_ids, download_folder):
    """
    Download many images concurrently using a thread pool.
    Each image is saved using its corresponding sample_id.
    
    Parameters:
    - image_links: Iterable of image URLs
    - sample_ids: Iterable of sample IDs (same length as image_links)
    - download_folder: Destination folder path
    """
    os.makedirs(download_folder, exist_ok=True)

    if len(image_links) != len(sample_ids):
        raise ValueError("image_links and sample_ids must have the same length")

    urls_and_ids = [
        (u, sid)
        for u, sid in zip(image_links, sample_ids)
        if isinstance(u, str) and u
    ]
    if not urls_and_ids:
        return

    cpu = os.cpu_count() or 8
    max_workers = min(64, max(8, cpu * 5))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_image, url, sid, download_folder)
            for url, sid in urls_and_ids
        ]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass
    return