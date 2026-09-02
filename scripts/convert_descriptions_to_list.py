"""
Convert species_descriptions.json values from flat strings to lists of bullet points.
Run once to migrate existing generated data.
"""
import json

path = 'config/LLM_description/species_descriptions.json'
new_path = 'config/LLM_description/species_descriptions_list.json'

with open(path, 'r') as fin:
    data = json.load(fin)

converted = {}
for species, text in data.items():
    items = []
    for line in text.split('\n'):
        line = line.strip().lstrip('\u2022').strip()
        if line and not line.endswith(':'):
            items.append(line)
    converted[species] = items

with open(new_path, 'w') as fout:
    json.dump(converted, fout, indent=2)

print(f"Converted {len(converted)} species. Example:")
first = next(iter(converted))
for item in converted[first]:
    print(f"  • {item}")
