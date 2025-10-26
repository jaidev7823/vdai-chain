import json
import requests
from collections import defaultdict

# Download JSON
url = "https://extendscript.docsforadobe.dev/"
response = requests.get(url)
response.raise_for_status()
raw = response.json()
docs = raw.get("docs", [])

# Define standard column titles we care about
standard_titles = {"description", "type", "example", "parameters", "returns", "attributes", "methods"}

# Temporary storage per section/object
temp_store = defaultdict(lambda: defaultdict(dict))
# Key: (section, object_name)

for entry in docs:
    loc = entry.get("location", "").strip()
    title = entry.get("title", "").strip()
    text = entry.get("text", "").strip()

    if not loc or loc.startswith("_global") or "config" in title.lower() or "readme" in title.lower():
        continue

    # First segment of location as section
    section = loc.split("/")[0]

    # Determine if this title is a standard column
    if title.lower() in standard_titles:
        # Assign under latest object in this section if exists
        objects_in_section = [k for k in temp_store if k[0] == section]
        if not objects_in_section:
            continue  # Skip metadata without object
        object_name = objects_in_section[-1][1]
        key = (section, object_name)
        temp_store[key][title.lower()] = text
    else:
        # Treat this as a new object or formula
        object_name = title
        key = (section, object_name)
        # Initialize empty standard columns if not exists
        if key not in temp_store:
            temp_store[key].update({col: "" for col in standard_titles})
        # Store extra info in 'details' and keep raw html in 'raw_text'
        temp_store[key]["details"] = text
        # temp_store[key]["raw_text"] = text

# Group by section -> object_name
grouped = defaultdict(dict)
for (section, object_name), fields in temp_store.items():
    grouped[section][object_name] = fields

# Save grouped JSON
with open("ppro_grouped_with_details.json", "w", encoding="utf-8") as f:
    json.dump(grouped, f, indent=2, ensure_ascii=False)

print(f"Grouped JSON with details saved. Sections: {len(grouped)}")
