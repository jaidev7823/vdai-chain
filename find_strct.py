import json
import requests
from collections import defaultdict

# Download JSON
url = "https://ppro-scripting.docsforadobe.dev/search/search_index.json"
response = requests.get(url)
response.raise_for_status()
raw = response.json()

docs = raw.get("docs", [])

# Dictionary: first location path -> set of titles
section_titles = defaultdict(set)

for entry in docs:
    loc = entry.get("location", "").strip()
    title = entry.get("title", "").strip()

    # Skip empty locations or global config/readme
    if not loc or loc.startswith("_global") or "config" in title.lower() or "readme" in title.lower():
        continue

    # Use only first path segment
    first_segment = loc.split("/")[0]

    # Add title to set (automatically avoids duplicates)
    section_titles[first_segment].add(title)

# Convert sets to sorted lists for easier readability
section_titles_sorted = {k: sorted(list(v)) for k, v in section_titles.items()}

# Print result
for section, titles in section_titles_sorted.items():
    print(f"Section: {section}")
    for t in titles:
        print(f"  - {t}")
    print("\n")

# Optional: save to JSON
with open("section_titles.json", "w", encoding="utf-8") as f:
    json.dump(section_titles_sorted, f, indent=2, ensure_ascii=False)
