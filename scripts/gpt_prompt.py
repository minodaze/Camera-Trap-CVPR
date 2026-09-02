import os
from openai import OpenAI
import json

# Initialize the OpenAI client
# It will automatically look for the OPENAI_API_KEY environment variable.
# Alternatively, you can pass it directly: OpenAI(api_key="your_api_key_here")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def generate_species_description(species_name):
    # The system prompt based exactly on the WildIng paper's template
    system_prompt = """You are an AI assistant specialized in biology and providing accurate and detailed descriptions of animal species. We are creating detailed and specific prompts to describe various species. The goal is to generate multiple sentences that capture different aspects of each species' appearance and behavior. Please follow the structure and style shown in the examples below. Each species should have a set of descriptions that highlight key characteristics.
Example Structure:
Badger:
• a badger is a mammal with a stout body and short sturdy legs.
• a badger's fur is coarse and typically grayish-black.
• badgers often feature a white stripe running from the nose to the back of the head dividing into two stripes along the sides of the body to the base of the tail.
• badgers have broad flat heads with small eyes and ears.
• badger noses are elongated and tapered ending in a black muzzle.
• badgers possess strong well-developed claws adapted for digging burrows.
• overall badgers have a rugged and muscular appearance suited for their burrowing lifestyle."""

    try:
        # Call the OpenAI API using the Chat Completions endpoint
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                # We simply provide the species name as the user prompt, 
                # prompting the model to complete the pattern.
                {"role": "user", "content": f"{species_name}:"}
            ],
            temperature=0.7, # A slight bit of creativity while remaining factual
            max_tokens=250   # Short descriptions don't need too many tokens
        )
        
        # Parse bullet points into a list, skipping the header line (e.g. "Springbok:")
        raw = response.choices[0].message.content
        items = []
        for line in raw.split('\n'):
            line = line.strip().lstrip('\u2022').lstrip('•').strip()
            if line and not line.endswith(':'):
                items.append(line)
        return items
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# --- Example Usage ---
if __name__ == "__main__":
    train_path = '/fs/scratch/PAS2099/camera-trap-benchmark/dataset/nz/nz_EFH_HCAME09/30/train.json'
    test_path = '/fs/scratch/PAS2099/camera-trap-benchmark/dataset/nz/nz_EFH_HCAME09/30/test.json'
    with open(train_path, 'r') as fin:
        data = json.load(fin)
    with open(test_path, 'r') as fin:
        data_test = json.load(fin)
        data.update(data_test)
    class_names = []
    label_type = 'common'
    for key, value in data.items():
        for v in value:
            if v[label_type] not in class_names:
                class_names.append(v[label_type])

    output_path = 'config/LLM_description/nz_EFH_HCAME09_species_descriptions.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing results to allow resuming interrupted runs
    if os.path.exists(output_path):
        with open(output_path, 'r') as fin:
            descriptions = json.load(fin)
    else:
        descriptions = {}

    for class_name in class_names:
        if class_name in descriptions:
            print(f"Skipping {class_name} (already generated)")
            continue
        print(f"Generating description for {class_name}...")
        desc = generate_species_description(class_name)
        descriptions[class_name] = desc
        for item in desc:
            print(f"  • {item}")
        print("\n" + "="*50 + "\n")
        # Save after each species so progress is not lost on failure
        with open(output_path, 'w') as fout:
            json.dump(descriptions, fout, indent=2)

    print(f"Saved descriptions to {output_path}")
