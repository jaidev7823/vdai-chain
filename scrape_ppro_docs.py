# scrape_ppro_docs.py
import trafilatura, requests, os
from bs4 import BeautifulSoup
from urllib.parse import urljoin

base = "https://ppro-scripting.docsforadobe.dev/"
resp = requests.get(base)
soup = BeautifulSoup(resp.text, "html.parser")

links = {urljoin(base, a["href"]) for a in soup.select("a[href]") if a["href"].endswith("/")}

os.makedirs("docs_raw", exist_ok=True)
for link in links:
    try:
        downloaded = trafilatura.fetch_url(link)
        text = trafilatura.extract(downloaded)
        if text:
            fname = link.rstrip("/").split("/")[-1] or "index"
            with open(f"docs_raw/{fname}.txt", "w", encoding="utf-8") as f:
                f.write(text)
            print("saved", fname)
    except Exception as e:
        print("fail", link, e)
