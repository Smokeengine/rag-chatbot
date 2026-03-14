import requests
from bs4 import BeautifulSoup
import os, re, time
from urllib.parse import urljoin
from collections import deque
import PyPDF2
from io import BytesIO

# ==== CONFIGURATION ====
START_URL = "https://tilt.colostate.edu/"
SAVE_DIR = "data"
MAX_PAGES = 3000        # increase if you want deeper coverage
DELAY = 1.5             # polite delay between requests (seconds)
TIMEOUT = 15            # max seconds to wait per request
# ========================

os.makedirs(SAVE_DIR, exist_ok=True)


# ---------- Helper: Clean & Save ----------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def save_text(name, text):
    path = os.path.join(SAVE_DIR, f"{name[:80]}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------- Helper: PDF Extraction ----------
def extract_pdf_text(pdf_bytes):
    try:
        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return clean_text(text)
    except Exception as e:
        print(f"⚠️ PDF extraction error: {e}")
        return ""


# ---------- Helper: HTTP Safe Get ----------
def safe_get(url, retries=1):
    for attempt in range(retries + 1):
        try:
            res = requests.get(url, timeout=TIMEOUT, headers={'User-Agent': 'Mozilla/5.0'})
            return res
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout on {url}, attempt {attempt+1}/{retries+1}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Request error on {url}: {e}")
        time.sleep(2)
    return None


# ---------- Save & Load Progress ----------
def save_progress(visited, queue):
    with open("visited.txt", "w", encoding="utf-8") as f:
        for url in visited:
            f.write(url + "\n")
    with open("queue.txt", "w", encoding="utf-8") as f:
        for url in queue:
            f.write(url + "\n")


def load_progress():
    visited, queue = set(), deque()
    if os.path.exists("visited.txt"):
        with open("visited.txt", "r", encoding="utf-8") as f:
            visited = set(line.strip() for line in f if line.strip())
    if os.path.exists("queue.txt"):
        with open("queue.txt", "r", encoding="utf-8") as f:
            queue = deque(line.strip() for line in f if line.strip())
    return visited, queue


# ---------- Page Processing ----------
def process_url(url):
    res = safe_get(url)
    if not res:
        return []

    ctype = res.headers.get("Content-Type", "")

    # --- Handle PDFs ---
    if "application/pdf" in ctype or url.lower().endswith(".pdf"):
        print(f"📄 PDF found: {url}")
        text = extract_pdf_text(res.content)
        if text:
            name = re.sub(r'[^a-zA-Z0-9]+', '_', url.replace(START_URL, '').strip('/'))
            save_text(name or "pdf_file", text)
            print(f"✅ Saved PDF text: {url}")
        return []

    # --- Handle HTML Pages ---
    if "text/html" in ctype:
        soup = BeautifulSoup(res.text, "html.parser")
        text = clean_text(soup.get_text(separator="\n", strip=True))
        name = re.sub(r'[^a-zA-Z0-9]+', '_', url.replace(START_URL, '').strip('/'))
        save_text(name or "page", text)
        print(f"✅ Saved page: {url}")

        # Collect new internal links
        new_links = []
        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"])
            if link.startswith(START_URL):
                new_links.append(link)
        return new_links
    return []


# ---------- MAIN LOOP ----------
visited, queue = load_progress()
if not queue:
    queue.append(START_URL)
print(f"🔁 Loaded progress: {len(visited)} visited, {len(queue)} in queue.")

while queue and len(visited) < MAX_PAGES:
    current = queue.popleft()
    if current in visited:
        continue
    visited.add(current)

    print(f"🌐 Visiting ({len(visited)}): {current}")
    new_links = process_url(current)

    # Add discovered links
    for link in new_links:
        if link not in visited:
            queue.append(link)

    save_progress(visited, queue)
    time.sleep(DELAY)

print(f"🎯 Finished! Scraped {len(visited)} pages (HTML + PDFs).")
