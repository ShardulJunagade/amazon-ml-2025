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


def download_image(image_link, savefolder):
    """
    Download a single image to the given folder.
    Keeps the original filename from the URL path.
    Skips download if file already exists and is non-empty.
    """
    if not isinstance(image_link, str) or not image_link:
        return

    parsed = urlparse(image_link)
    if parsed.scheme not in ("http", "https"):
        print(f"Warning: Unsupported URL scheme - {image_link}")
        return

    # Derive a safe filename from the URL path
    filename = unquote(os.path.basename(parsed.path)) or "image"
    image_save_path = os.path.join(savefolder, filename)

    # Fast exit if already downloaded
    try:
        if os.path.exists(image_save_path) and os.path.getsize(image_save_path) > 0:
            return
    except OSError:
        # If stat fails for any reason, attempt download anyway
        pass

    session = _get_session()
    try:
        with session.get(image_link, stream=True, timeout=(5, 20)) as resp:
            resp.raise_for_status()
            # Ensure target dir exists (in case savefolder is nested)
            os.makedirs(savefolder, exist_ok=True)
            with open(image_save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as ex:
        print(f"Warning: Not able to download - {image_link}\n{ex}")
        # Best-effort clean up partial files
        try:
            if os.path.exists(image_save_path) and os.path.getsize(image_save_path) == 0:
                os.remove(image_save_path)
        except OSError:
            pass
    return

def download_images(image_links, download_folder):
    """
    Download many images concurrently using a thread pool (I/O-bound optimized).

    Parameters:
    - image_links: Iterable of image URLs (list, Series, etc.)
    - download_folder: Destination folder path
    """
    os.makedirs(download_folder, exist_ok=True)

    # Normalize and filter inputs quickly (avoid None/NaN/non-strings)
    try:
        # If it's a pandas Series, turn into list
        image_links = list(image_links)
    except TypeError:
        # If it's already an iterator, this will fail only on non-iterables
        image_links = []

    urls = [u for u in image_links if isinstance(u, str) and u]
    if not urls:
        return

    # Choose a sensible default for I/O concurrency
    cpu = os.cpu_count() /2
    max_workers = min(64, max(8, cpu * 5))

    download_image_partial = partial(download_image, savefolder=download_folder)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_image_partial, url) for url in urls]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass
    return