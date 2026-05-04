from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image
import requests
import json
from torch import nn
import torch.nn.functional as F

# Replace with the actual path to your 'siglip2-large-patch16-256' folder
model_path = "/users/PAS2099/mino/ICICLE/pretrained_weights/siglip2-large-patch16-256"

# Load the processor and the model
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

OPENAI_IMAGENET_TEMPLATE = [
    'a photo of {CLZ_NAME}.',
    'a bad photo of a {CLZ_NAME}.',
    'a photo of many {CLZ_NAME}.',
    'a sculpture of a {CLZ_NAME}.',
    'a photo of the hard to see {CLZ_NAME}.',
    'a low resolution photo of the {CLZ_NAME}.',
    'a rendering of a {CLZ_NAME}.',
    'graffiti of a {CLZ_NAME}.',
    'a bad photo of the {CLZ_NAME}.',
    'a cropped photo of the {CLZ_NAME}.',
    'a tattoo of a {CLZ_NAME}.',
    'the embroidered {CLZ_NAME}.',
    'a photo of a hard to see {CLZ_NAME}.',
    'a bright photo of a {CLZ_NAME}.',
    'a photo of a clean {CLZ_NAME}.',
    'a photo of a dirty {CLZ_NAME}.',
    'a dark photo of the {CLZ_NAME}.',
    'a drawing of a {CLZ_NAME}.',
    'a photo of my {CLZ_NAME}.',
    'the plastic {CLZ_NAME}.',
    'a photo of the cool {CLZ_NAME}.',
    'a close-up photo of a {CLZ_NAME}.',
    'a black and white photo of the {CLZ_NAME}.',
    'a painting of the {CLZ_NAME}.',
    'a painting of a {CLZ_NAME}.',
    'a pixelated photo of the {CLZ_NAME}.',
    'a sculpture of the {CLZ_NAME}.',
    'a bright photo of the {CLZ_NAME}.',
    'a cropped photo of a {CLZ_NAME}.',
    'a plastic {CLZ_NAME}.',
    'a photo of the dirty {CLZ_NAME}.',
    'a jpeg corrupted photo of a {CLZ_NAME}.',
    'a blurry photo of the {CLZ_NAME}.',
    'a photo of the {CLZ_NAME}.',
    'a good photo of the {CLZ_NAME}.',
    'a rendering of the {CLZ_NAME}.',
    'a {CLZ_NAME} in a video game.',
    'a photo of one {CLZ_NAME}.',
    'a doodle of a {CLZ_NAME}.',
    'a close-up photo of the {CLZ_NAME}.',
    'a photo of a {CLZ_NAME}.',
    'the origami {CLZ_NAME}.',
    'the {CLZ_NAME} in a video game.',
    'a sketch of a {CLZ_NAME}.',
    'a doodle of the {CLZ_NAME}.',
    'a origami {CLZ_NAME}.',
    'a low resolution photo of a {CLZ_NAME}.',
    'the toy {CLZ_NAME}.',
    'a rendition of the {CLZ_NAME}.',
    'a photo of the clean {CLZ_NAME}.',
    'a photo of a large {CLZ_NAME}.',
    'a rendition of a {CLZ_NAME}.',
    'a photo of a nice {CLZ_NAME}.',
    'a photo of a weird {CLZ_NAME}.',
    'a blurry photo of a {CLZ_NAME}.',
    'a cartoon {CLZ_NAME}.',
    'art of a {CLZ_NAME}.',
    'a sketch of the {CLZ_NAME}.',
    'a embroidered {CLZ_NAME}.',
    'a pixelated photo of a {CLZ_NAME}.',
    'itap of the {CLZ_NAME}.',
    'a jpeg corrupted photo of the {CLZ_NAME}.',
    'a good photo of a {CLZ_NAME}.',
    'a plushie {CLZ_NAME}.',
    'a photo of the nice {CLZ_NAME}.',
    'a photo of the small {CLZ_NAME}.',
    'a photo of the weird {CLZ_NAME}.',
    'the cartoon {CLZ_NAME}.',
    'art of the {CLZ_NAME}.',
    'a drawing of the {CLZ_NAME}.',
    'a photo of the large {CLZ_NAME}.',
    'a black and white photo of a {CLZ_NAME}.',
    'the plushie {CLZ_NAME}.',
    'a dark photo of a {CLZ_NAME}.',
    'itap of a {CLZ_NAME}.',
    'graffiti of the {CLZ_NAME}.',
    'a toy {CLZ_NAME}.',
    'itap of my {CLZ_NAME}.',
    'a photo of a cool {CLZ_NAME}.',
    'a photo of a small {CLZ_NAME}.',
    'a tattoo of the {CLZ_NAME}.',
]

BIOCLIP_TEMPLATE = [
    'a photo of {CLZ_NAME}.',
]
CAMERA_TRAP_TEMPLATE = [
    'a camera trap photo of {CLZ_NAME}.',
]

def hf_get_text_features(model: nn.Module, inputs: dict) -> torch.Tensor:
    out = model.get_text_features(**inputs)
    # SiglipModel may return BaseModelOutputWithPooling instead of a plain tensor
    if torch.is_tensor(out):
        return out
    if hasattr(out, 'pooler_output') and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, 'last_hidden_state'):
        return out.last_hidden_state[:, 0]  # CLS token
    raise RuntimeError(f"Cannot extract tensor from model output: {type(out)}")


def hf_get_image_features(model: nn.Module, inputs: dict) -> torch.Tensor:
    out = model.get_image_features(pixel_values=inputs)
    if torch.is_tensor(out):
        return out
    if hasattr(out, 'pooler_output') and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, 'last_hidden_state'):
        return out.last_hidden_state[:, 0]
    raise RuntimeError(f"Cannot extract image tensor from model output: {type(out)}")

def get_class_embedding_hf(model: nn.Module, processor, embed_dim: int, class_name_idx, text_template: str = "openai"):
    """Compute normalized per-class text embeddings using a HF model/processor.

    This mirrors `get_class_embedding()` for OpenCLIP, but uses `processor` + HF forward.
    """

    device = next(model.parameters()).device
    with torch.no_grad():
        class_embedding = torch.empty(len(class_name_idx), embed_dim)
        for class_name, class_idx in class_name_idx.items():
            texts = get_texts(class_name, text_template)
            inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            text_feats = hf_get_text_features(model, inputs)

            if not torch.is_tensor(text_feats):
                raise RuntimeError(f"HF text features must be a Tensor, got: {type(text_feats)}")
            if text_feats.ndim == 1:
                text_feats = text_feats.unsqueeze(0)
            if text_feats.shape[-1] != embed_dim:
                raise RuntimeError(
                    f"HF text feature dim ({int(text_feats.shape[-1])}) does not match embed_dim ({int(embed_dim)}). "
                    "This usually indicates the wrong config field was used (projection vs hidden size) or the model "
                    "returned unprojected pooled features."
                )

            text_feats = F.normalize(text_feats, dim=-1).mean(dim=0)
            text_feats = F.normalize(text_feats, dim=-1)
            class_embedding[class_idx] = text_feats.detach().cpu()
    return class_embedding

def get_texts(c, text_template='openai'):
    texts = [template.format(CLZ_NAME=c) for template in CAMERA_TRAP_TEMPLATE]
    return texts

train_json_path = "/fs/scratch/PAS2099/camera-trap-benchmark/dataset/APN/APN_BOSP/30/train.json"
test_json_path = "/fs/scratch/PAS2099/camera-trap-benchmark/dataset/APN/APN_BOSP/30/test.json"

with open(train_json_path, 'r') as fin:
        data = json.load(fin)
with open(test_json_path, 'r') as fin:
    data_test = json.load(fin)
    data.update(data_test)

class_names = []
for key, value in data.items():
    for v in value:
        if v["common"] not in class_names:
            class_names.append(v["common"])

image_path = '/fs/scratch/PAS2099/camera-trap-benchmark/dataset/APN/APN_BOSP/images/APN_S4_BosP_R7_IMAG0003.JPG'

# Preprocess inputs
# inputs = processor(text=texts, images=image, return_tensors="pt", padding="max_length").to(device)

with torch.no_grad():
    class_name_idx = {name: idx for idx, name in enumerate(class_names)}
    embed_dim = model.config.text_config.hidden_size
    class_embedding = get_class_embedding_hf(model, processor, embed_dim, class_name_idx)
    class_embedding = class_embedding.to(device)
    (f"Class embedding shape: {class_embedding.shape}")
    print(f"Class embedding sample (first 5 values): {class_embedding[0][:10]}")
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt")
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    pixel_values = image_inputs['pixel_values']
    print(f"Pixel values shape: {pixel_values.shape}")
    print(f"Pixel values sample (first 5 values): {pixel_values[0, 0, :5, :5]}")  # Print a small patch of the first channel
    image_features = hf_get_image_features(model, pixel_values)
    image_features = F.normalize(image_features, dim=-1)
    image_features = image_features.to(device)
    print(f"Image features shape: {image_features.shape}")
    print(f"Image features sample (first 10 values): {image_features[0][:10]}")
    logits_per_image = image_features @ class_embedding.t()
    print(f"Logits shape: {logits_per_image.shape}")
    print(f"Logits sample (first 5 values): {logits_per_image}")
    # softmax_probs = F.softmax(logits_per_image, dim=-1)
    # print(f"Softmax probabilities shape: {softmax_probs.shape}")
    # print(f"Softmax probabilities sample (first 5 values): {softmax_probs[0][:5]}")

#     outputs = model(**inputs)
    
# # Get probabilities
# logits_per_image = outputs.logits_per_image
# probs = torch.sigmoid(logits_per_image) # SigLIP uses sigmoid instead of softmax
# print(f"Label probabilities: {probs}")