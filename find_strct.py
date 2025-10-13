import json
import requests
from collections import defaultdict

# Download JSON
url = "https://ppro-scripting.docsforadobe.dev/search/search_index.json"
response = requests.get(url)
response.raise_for_status()
raw = response.json()
docs = raw.get("docs", [])

# Define standard column titles we care about
standard_titles = {"description", "type", "example", "parameters", "returns", "attributes", "methods"}

# Result storage: list of structured objects
structured_data = []

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

    # Decide object_name
    # If title is a standard column, we assign under current object
    if title.lower() in standard_titles:
        # Try to get current object_name
        object_name = None
        # Look backwards in temp_store for latest object in this section
        # If no object exists yet, skip
        objects_in_section = [k for k in temp_store if k[0] == section]
        if objects_in_section:
            object_name = objects_in_section[-1][1]
        else:
            # No object yet, skip this metadata
            continue
    else:
        # Treat this title as object_name
        object_name = title

    key = (section, object_name)
    # Assign text to the right field if it's a standard title
    if title.lower() in standard_titles:
        temp_store[key][title.lower()] = text
    else:
        # New object: initialize row with empty standard columns
        temp_store[key].update({col: "" for col in standard_titles})

# Convert temp_store into list of dicts
for (section, object_name), fields in temp_store.items():
    row = {
        "section": section,
        "object_name": object_name,
    }
    row.update(fields)
    structured_data.append(row)

# Save as JSON
with open("ppro_structured.json", "w", encoding="utf-8") as f:
    json.dump(structured_data, f, indent=2, ensure_ascii=False)

print(f"Structured JSON saved. Total objects: {len(structured_data)}")
