import os
import json
import torch
import numpy as np
import pyiqa
# import wandb

from tqdm.auto import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchmetrics.multimodal import CLIPImageQualityAssessment
from pytorch_image_generation_metrics import (
    get_inception_score,
    get_fid,
    get_inception_score_and_fid
)


def pil_to_torch(img, device, normalize=True):
    img = torch.tensor(np.array(img), device=device).permute(2, 0, 1)
    if normalize:
        img = img / 255.0
    return img


def clip_score_and_iqa(image_folder, prefix_pos, prefix_neg, device="cuda:0"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    prompt_metric = ("quality", "colorfullness", "sharpness")
    clipiqa_model = CLIPImageQualityAssessment(
        model_name_or_path="openai/clip-vit-base-patch16",
        prompts=prompt_metric,
        data_range=1.0
    ).to(device)

    image_files = [
        f for f in os.listdir(image_folder)
        if f.startswith(prefix_pos) and not f.startswith(prefix_neg) and f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if len(image_files) == 0:
        return None, None, None, None
    images = []
    for f in image_files:
        img = Image.open(os.path.join(image_folder, f))
        w, h = img.size
        img = img.resize((w // 2, h // 2), Image.LANCZOS)
        images.append(img)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parts = os.path.normpath(image_folder).split(os.sep)
    scene_id = parts[parts.index('output_5') + 2]
    if scene_id is None:
        raise ValueError(f"Cannot derive scene id from path: {image_folder}")
    text_path = os.path.join(project_root, "input", "scene_texts", f"{scene_id}.txt")
    if not os.path.isfile(text_path):
        raise FileNotFoundError(f"Scene text file not found: {text_path}")
    with open(text_path, "r") as f:
        prompt = f.readline().strip()

    scores = torch.zeros((len(prompt_metric), len(images)), device=device)
    clip_scores = torch.zeros(len(images), device=device)

    batch_size = 64
    pbar = tqdm(range(0, len(images), batch_size), desc="Calc CLIP Score and CLIP IQA (batched)")
    with torch.no_grad():
        for start in pbar:
            end = min(start + batch_size, len(images))
            batch_images = images[start:end]

            # 用 PIL 交给 processor，随后把返回的张量迁移到 GPU
            inputs = processor(
                text=[prompt],      # 单个 prompt
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,  # 关键：启用截断
                max_length=processor.tokenizer.model_max_length  # 通常为 77
            )
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

            outputs = model(**inputs)
            clip_scores[start:end] = outputs.logits_per_image.detach().squeeze(-1)
            # logit_scale = model.logit_scale.exp()
            # clip_scores_norm = (outputs.logits_per_image / logit_scale).detach().squeeze(-1)
            # clip_scores_unit = (clip_scores_norm + 1) / 2

            # CLIP-IQA 批量：转为 GPU 张量并评估
            img_torch_batch = [pil_to_torch(img, device, normalize=False) for img in batch_images]
            img_batch_for_iqa = torch.stack(img_torch_batch, dim=0).to(device)
            clipiqa_out = clipiqa_model(img_batch_for_iqa)
            for prompt_idx in range(len(prompt_metric)):
                scores[prompt_idx][start:end] = clipiqa_out[prompt_metric[prompt_idx]].detach().to(device)

    return clip_scores.mean().cpu().numpy(), scores[0].mean().cpu().numpy(), scores[1].mean().cpu().numpy(), scores[2].mean().cpu().numpy()


def brisque_and_niqe_score(image_folder, prefix_pos, prefix_neg):       # pyiqa
    # print(f"image_folder: {image_folder}, prefix: {prefix}")
    # images = [Image.open(os.path.join(image_folder, f)) for f in os.listdir(image_folder) if "png" in f and f.startswith(prefix)]
    images_tensor = []
    for f in os.listdir(image_folder):
        if f.endswith("png") and f.startswith(prefix_pos) and not f.startswith(prefix_neg):
            image = Image.open(os.path.join(image_folder, f))
            w, h = image.size
            image = image.resize((w // 2, h // 2), Image.LANCZOS)
            image_t = pil_to_torch(image, "cpu", normalize=True)  
            images_tensor.append(image_t)

    if len(images_tensor) == 0:
        return None, None

    if len(images_tensor) >=64 :
        batch_size = 64
    else:
        batch_size = len(images_tensor)
    brisque_metric = pyiqa.create_metric('brisque')
    niqe_metric = pyiqa.create_metric('niqe')

    brisque_scores_all = []
    niqe_scores_all = []

    with torch.no_grad():
        for start in range(0, len(images_tensor), batch_size):
            batch_list = images_tensor[start:start + batch_size]
            batch = torch.stack(batch_list, dim=0)

            brisque_scores = brisque_metric(batch).tolist()
            if not isinstance(brisque_scores, list):
                brisque_scores = [brisque_scores]

            niqe_scores = niqe_metric(batch).tolist()
            if not isinstance(niqe_scores, list):
                niqe_scores = [niqe_scores]

            brisque_scores_all.extend(brisque_scores)
            niqe_scores_all.extend(niqe_scores)

    return np.mean(brisque_scores_all), np.mean(niqe_scores_all)

    # wandb.log({
    #     'brisque': np.mean(brisque_scores),
    #     'niqe': np.mean(niqe_scores)
    # })

def inception_score(image_folder, prefix_pos, prefix_neg):
    images_tensor = []
    for f in os.listdir(image_folder):
        if f.endswith("png") and f.startswith(prefix_pos) and not f.startswith(prefix_neg):
            image = Image.open(os.path.join(image_folder, f))
            w, h = image.size
            image = image.resize((w // 2, h // 2), Image.LANCZOS)
            image_t = pil_to_torch(image, "cpu", normalize=True)
            images_tensor.append(image_t)

    if len(images_tensor) == 0:
        return None

    inception_scores = []
    batch_size = 64
    with torch.no_grad():
        for start in range(0, len(images_tensor), batch_size):
            batch_list = images_tensor[start:start + batch_size]
            batch = torch.stack(batch_list, dim=0)
            score, std = get_inception_score(batch)
            inception_scores.append(score)
        
    return np.mean(inception_scores)
