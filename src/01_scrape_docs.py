import os
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://developer.adobe.com/premiere-pro/uxp/ppro_reference/"
OUT_DIR = "docs_json"
os.makedirs(OUT_DIR, exist_ok=True)

def parse_table(table):
    """Parse table and preserve structure"""
    headers = [th.get_text(strip=True) for th in table.select("thead th")]
    if not headers:
        first_row = table.select_one("tr")
        if first_row:
            headers = [td.get_text(strip=True) for td in first_row.select("td, th")]
    
    rows = []
    for tr in table.select("tbody tr") if table.select("tbody tr") else table.select("tr")[1:]:
        cells = [td.get_text(strip=True) for td in tr.select("td")]
        if cells and len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
        elif cells:
            rows.append(cells)
    
    return {"headers": headers, "rows": rows} if headers else {"rows": rows}

def normalize_commands(section_dict):
    """Convert a section like Instance Methods to commands array"""
    commands = []
    
    for key, value in section_dict.items():
        if key == "content":
            # Skip generic content
            continue
            
        command_obj = {
            "command": {
                "name": key,
                "description": None,
                "parameters": None,
                "returns": None,
                "details": []
            }
        }
        
        # Value is a list of content items
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    if "content" in item:
                        # Text description
                        if command_obj["command"]["description"] is None:
                            command_obj["command"]["description"] = item["content"]
                        else:
                            command_obj["command"]["details"].append(item)
                    
                    elif "Table" in item:
                        # Determine table type from headers
                        table = item["Table"]
                        if isinstance(table, dict) and "headers" in table:
                            headers = [h.lower() for h in table["headers"]]
                            
                            # Parameter table
                            if any(h in headers for h in ["parameter", "param", "name", "type"]):
                                command_obj["command"]["parameters"] = table["rows"]
                            # Return value table
                            elif any(h in headers for h in ["return", "returns", "type"]):
                                command_obj["command"]["returns"] = table["rows"]
                            else:
                                command_obj["command"]["details"].append(item)
                        else:
                            command_obj["command"]["details"].append(item)
                    
                    else:
                        # Other structured content (lists, code, etc.)
                        command_obj["command"]["details"].append(item)
        
        elif isinstance(value, str):
            command_obj["command"]["description"] = value
        
        # Clean up empty fields
        if not command_obj["command"]["details"]:
            del command_obj["command"]["details"]
        
        commands.append(command_obj)
    
    return {"commands": commands}

def parse_div(div):
    data = {"title": None, "description": None, "sections": []}
    current_section = None
    current_section_key = None
    current_subkey = None

    # Find h1 first
    h1 = div.find("h1")
    if h1:
        data["title"] = h1.get_text(strip=True)
        # Next siblings until first h2 or p
        for sibling in h1.find_next_siblings():
            if sibling.name == "h2":
                break
            if sibling.name == "p":
                data["description"] = sibling.get_text(strip=True)
                break

    for el in div.find_all(recursive=False):
        if el.name == "h1":
            continue

        elif el.name == "h2":
            # Save previous section
            if current_section_key and current_section:
                # Check if this looks like a methods/properties section
                if any(keyword in current_section_key.lower() for keyword in ["method", "function", "property", "properties"]):
                    current_section = normalize_commands(current_section)
                data["sections"].append({current_section_key: current_section})
            
            current_section_key = el.get_text(strip=True)
            current_section = {}
            current_subkey = None

        elif el.name in ["h3", "h4"]:
            current_subkey = el.get_text(strip=True)
            if current_section is None:
                current_section = {}
            if current_subkey not in current_section:
                current_section[current_subkey] = []

        elif el.name == "p":
            text = el.get_text(strip=True)
            if current_section is None:
                current_section = {}
            if current_subkey:
                current_section[current_subkey].append({"content": text})
            else:
                current_section.setdefault("content", []).append({"content": text})

        elif el.name == "table":
            table_data = parse_table(el)
            if current_section is None:
                current_section = {}
            if current_subkey:
                current_section[current_subkey].append({"Table": table_data})
            else:
                current_section.setdefault("content", []).append({"Table": table_data})

        elif el.name in ["ul", "ol"]:
            lst = [li.get_text(strip=True) for li in el.select("li")]
            if current_section is None:
                current_section = {}
            if current_subkey:
                current_section[current_subkey].append({"list": lst})
            else:
                current_section.setdefault("content", []).append({"list": lst})

        elif el.name in ["pre", "code"]:
            code_text = el.get_text()
            if current_section is None:
                current_section = {}
            if current_subkey:
                current_section[current_subkey].append({"code": code_text})
            else:
                current_section.setdefault("content", []).append({"code": code_text})

    # Append last section
    if current_section_key and current_section:
        if any(keyword in current_section_key.lower() for keyword in ["method", "function", "property", "properties"]):
            current_section = normalize_commands(current_section)
        data["sections"].append({current_section_key: current_section})
    elif current_section:
        data["sections"].append(current_section)

    return data

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
        with open(f"{OUT_DIR}/{fname}.json", "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
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