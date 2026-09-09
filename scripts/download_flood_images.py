import hashlib
import mimetypes
from pathlib import Path
from urllib.parse import quote_plus, urlparse

import requests
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


DATA_DIR = Path("data/processed/flood_classified")
MIN_BYTES = 5 * 1024
PER_QUERY = 100
TIMEOUT = 15
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
QUERIES = [
    ("mild", "mild flooding road"),
    ("mild", "light flood water street"),
    ("severe", "severe flood submerged buildings"),
    ("severe", "extreme flood disaster aerial"),
]


def get_output_dir(class_name):
    train_dir = DATA_DIR / "train" / class_name
    if train_dir.exists():
        train_dir.mkdir(parents=True, exist_ok=True)
        return train_dir

    class_dir = DATA_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    return class_dir


def existing_web_count(class_name):
    output_dir = get_output_dir(class_name)
    return len(list(output_dir.glob(f"web_{class_name}_*.jpg")))


def next_output_path(class_name):
    output_dir = get_output_dir(class_name)
    index = existing_web_count(class_name) + 1
    while True:
        candidate = output_dir / f"web_{class_name}_{index:03d}.jpg"
        if not candidate.exists():
            return candidate
        index += 1


def is_valid_image_response(response):
    if len(response.content) < MIN_BYTES:
        return False

    content_type = response.headers.get("content-type", "").split(";")[0].lower()
    if content_type in {"image/jpeg", "image/jpg", "image/png"}:
        return True

    path = urlparse(response.url).path.lower()
    ext = Path(path).suffix
    guessed_type = mimetypes.guess_type(path)[0]
    return ext in IMAGE_EXTS or guessed_type in {"image/jpeg", "image/png"}


def is_decodable_image(content):
    try:
        from io import BytesIO

        with Image.open(BytesIO(content)) as image:
            image.verify()
        return True
    except Exception:
        return False


def save_image(class_name, image_url, seen_hashes):
    try:
        response = requests.get(
            image_url,
            timeout=TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
    except Exception:
        return False

    if not is_valid_image_response(response):
        return False
    if not is_decodable_image(response.content):
        return False

    digest = hashlib.sha256(response.content).hexdigest()
    if digest in seen_hashes:
        return False
    seen_hashes.add(digest)

    output_path = next_output_path(class_name)
    output_path.write_bytes(response.content)
    return True


def download_with_icrawler(class_name, query, seen_hashes):
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        return 0

    temp_dir = DATA_DIR / "_web_downloads" / class_name / query.replace(" ", "_")
    temp_dir.mkdir(parents=True, exist_ok=True)
    crawler = BingImageCrawler(storage={"root_dir": str(temp_dir)})

    try:
        crawler.crawl(keyword=query, max_num=PER_QUERY)
    except Exception as exc:
        print(f"{query}: icrawler failed: {exc}")
        return 0

    saved = 0
    for path in sorted(temp_dir.iterdir()):
        if saved >= PER_QUERY:
            break
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            content = path.read_bytes()
        except OSError:
            continue
        if len(content) < MIN_BYTES:
            continue
        if not is_decodable_image(content):
            continue
        digest = hashlib.sha256(content).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        output_path = next_output_path(class_name)
        output_path.write_bytes(content)
        saved += 1

    return saved


def extract_bing_image_urls(query, offset):
    search_url = (
        "https://www.bing.com/images/search"
        f"?q={quote_plus(query)}&first={offset}&form=HDRSC2"
    )
    response = requests.get(
        search_url,
        timeout=TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(response.text, "html.parser")
    urls = []
    for tag in soup.select("a.iusc"):
        metadata = tag.get("m")
        if not metadata:
            continue
        marker = '"murl":"'
        if marker not in metadata:
            continue
        image_url = metadata.split(marker, 1)[1].split('"', 1)[0]
        urls.append(image_url.encode("utf-8").decode("unicode_escape"))
    return urls


def download_with_bing_html(class_name, query, seen_hashes):
    saved = 0
    offset = 0
    while saved < PER_QUERY and offset < 500:
        try:
            urls = extract_bing_image_urls(query, offset)
        except Exception as exc:
            print(f"{query}: Bing HTML search failed: {exc}")
            break

        if not urls:
            break

        for image_url in urls:
            if saved >= PER_QUERY:
                break
            if save_image(class_name, image_url, seen_hashes):
                saved += 1

        offset += 35

    return saved


def main():
    seen_hashes = set()
    for class_name, query in QUERIES:
        print(f"Downloading up to {PER_QUERY} images for {class_name}: {query}")
        saved = download_with_icrawler(class_name, query, seen_hashes)
        if saved < PER_QUERY:
            print(f"{query}: icrawler saved {saved}, trying Bing HTML fallback")
            saved += download_with_bing_html(class_name, query, seen_hashes)
        print(f"{query}: saved {saved} images")


if __name__ == "__main__":
    main()
