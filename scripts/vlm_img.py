import argparse
import json
import os

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"

print("Loading model... (this may take a minute)")
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to("cuda:1")


def generate_image_description(image_path):
    image = Image.open(image_path).convert("RGB")
    prompt = (
        "SYSTEM: You are an AI assistant specialized in biology and providing accurate "
        "and detailed descriptions of animal species.\n"
        "USER: <image>\n"
        "You are given the description of an animal species. "
        "Provide a very detailed description of the appearance of the species and describe each body part of the animal in detail. "
        "Only include details that can be directly visible in a photograph of the animal. "
        "Only include information related to the appearance of the animal and nothing else. "
        "Make sure to only include information that is present in the species description and is certainly true for the given species. "
        "Do not include any information related to the sound or smell of the animal. "
        "Do not include any numerical information related to measurements in the text in units: m cm in inches ft feet km/h kg lb lbs. "
        "Remove any special characters such as unicode tags from the text. Return the answer as a single paragraph.\n"
        "ASSISTANT:"
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda:1", torch.float16)
    output = model.generate(**inputs, max_new_tokens=200, use_cache=True, do_sample=True, temperature=0.2)
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    return generated_text.split("ASSISTANT:")[1].strip()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json-paths", nargs="+",
                   default=[
                       '/fs/scratch/PAS2099/camera-trap-benchmark/dataset/nz/nz_EFH_HCAME09/30/train.json',
                       '/fs/scratch/PAS2099/camera-trap-benchmark/dataset/nz/nz_EFH_HCAME09/30/test.json',
                   ],
                   help="One or more CkpDataset JSON files (train AND test) to process")
    p.add_argument("--output", default='config/LLM_description/nz_EFH_HCAME09_species_vlm_descriptions.json')
    args = p.parse_args()

    # Load existing output to support resuming
    if os.path.exists(args.output):
        with open(args.output) as f:
            vlm_data = json.load(f)
        already_done = {e['img_path'] for e in vlm_data}
        print(f"Resuming — {len(already_done)} images already processed.")
    else:
        vlm_data = []
        already_done = set()

    for json_path in args.json_paths:
        print(f"\nProcessing {json_path}...")
        with open(json_path) as fin:
            data_dict = json.load(fin)

        # Collect all (img_path, species) pairs to process
        pending = []
        for ckpt, data in data_dict.items():
            if not ckpt.startswith("ckp_"):
                continue
            for img_data in data:
                img_path = img_data['image_path']
                if img_path not in already_done:
                    pending.append((img_path, img_data.get('common', '')))

        print(f"  {len(pending)} images to process (skipping {len(already_done)} already done)")

        for i, (img_path, species_name) in enumerate(pending, 1):
            print(f"  [{i}/{len(pending)}] {os.path.basename(img_path)}", flush=True)
            try:
                description = generate_image_description(img_path)
                vlm_data.append({'img_path': img_path, 'species': species_name, 'description': description})
                already_done.add(img_path)
            except FileNotFoundError:
                print(f"    Image not found, skipping: '{img_path}'")
            except Exception as e:
                print(f"    Error processing '{img_path}': {e}")

            # Save every 50 images so progress is not lost on crash
            if i % 50 == 0:
                os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
                with open(args.output, 'w') as fout:
                    json.dump(vlm_data, fout, indent=2)
                print(f"  Checkpoint saved ({len(vlm_data)} total so far)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w') as fout:
        json.dump(vlm_data, fout, indent=2)
    print(f"\nSaved {len(vlm_data)} descriptions -> {args.output}")
