import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://developer.adobe.com/premiere-pro/uxp/ppro_reference/"
OUT_DIR = "docs_txt"
os.makedirs(OUT_DIR, exist_ok=True)

def parse_table(table):
    headers = [th.get_text(strip=True) for th in table.select("thead th")]
    if not headers:
        first_row = table.select_one("tr")
        if first_row:
            headers = [td.get_text(strip=True) for td in first_row.select("td, th")]

    rows = []
    for tr in table.select("tbody tr") if table.select("tbody tr") else table.select("tr")[1:]:
        cells = [td.get_text(strip=True) for td in tr.select("td")]
        if cells:
            rows.append(cells)

    lines = []
    if headers:
        lines.append("    " + " | ".join(headers))
        for r in rows:
            lines.append("    " + " | ".join(r))
    else:
        for r in rows:
            lines.append("    " + " | ".join(r))
    return "\n".join(lines)

def write_line(buf, text, indent=0):
    buf.append("  " * indent + text)

def parse_div(div):
    buf = []
    h1 = div.find("h1")
    if h1:
        write_line(buf, f"Title: {h1.get_text(strip=True)}")

    desc = None
    for sibling in div.find_all("p", recursive=False):
        desc = sibling.get_text(strip=True)
        if desc:
            break
    if desc:
        write_line(buf, f"Description: {desc}")
    buf.append("")

    current_section = None
    current_sub = None

    for el in div.find_all(recursive=False):
        if el.name == "h1":
            continue

        elif el.name == "h2":
            current_section = el.get_text(strip=True)
            write_line(buf, f"Section: {current_section}")
            current_sub = None

        elif el.name in ["h3", "h4"]:
            current_sub = el.get_text(strip=True)
            write_line(buf, f"  Subsection: {current_sub}")

        elif el.name == "p":
            text = el.get_text(strip=True)
            if current_sub:
                write_line(buf, f"    Content: {text}")
            elif current_section:
                write_line(buf, f"  Content: {text}")
            else:
                write_line(buf, f"Content: {text}")

        elif el.name == "table":
            table_text = parse_table(el)
            if current_sub:
                write_line(buf, "    Table:")
                buf.append(table_text)
            elif current_section:
                write_line(buf, "  Table:")
                buf.append(table_text)
            else:
                write_line(buf, "Table:")
                buf.append(table_text)

        elif el.name in ["ul", "ol"]:
            items = [li.get_text(strip=True) for li in el.select("li")]
            for li in items:
                if current_sub:
                    write_line(buf, f"    - {li}")
                elif current_section:
                    write_line(buf, f"  - {li}")
                else:
                    write_line(buf, f"- {li}")

        elif el.name in ["pre", "code"]:
            code_text = el.get_text()
            if current_sub:
                write_line(buf, "    Code:")
                for line in code_text.splitlines():
                    write_line(buf, f"      {line}")
            elif current_section:
                write_line(buf, "  Code:")
                for line in code_text.splitlines():
                    write_line(buf, f"    {line}")
            else:
                write_line(buf, "Code:")
                for line in code_text.splitlines():
                    write_line(buf, f"  {line}")

    return "\n".join(buf)

def scrape_page(url):
    try:
        r = requests.get(url)
        s = BeautifulSoup(r.text, "html.parser")
        div = s.select_one("div.css-1e0trr3")
        if not div:
            print("skip", url)
            return
        parsed = parse_div(div)
        fname = url.rstrip("/").split("/")[-1] or "index"
        with open(f"{OUT_DIR}/{fname}.txt", "w", encoding="utf-8") as f:
            f.write(parsed)
        print("saved", fname)
    except Exception as e:
        print("fail", url, e)

def main():
    base_resp = requests.get(BASE_URL)
    soup = BeautifulSoup(base_resp.text, "html.parser")
    links = {urljoin(BASE_URL, a["href"]) for a in soup.select("a[href]") if a["href"].endswith("/")}
    for link in links:
        scrape_page(link)

if __name__ == "__main__":
    main()
