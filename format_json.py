import json

file = "data/scrape_dc.json"

with open(file, "r") as f:
    data = json.load(f)
    new_data = []
    # Concatenate all of the string fields into one string
    for item in data.values():
        item_text = ""
        for entry in item.values():
            if type(entry) == str:
                item_text += entry + "\n"
        new_data.append({
            "text":item_text.strip()})
    
    # Write the new data to a new file
    with open("data/scrape_dc_formatted.json", "w") as f:
        json.dump(new_data, f)
