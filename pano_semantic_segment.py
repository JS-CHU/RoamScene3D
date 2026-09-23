"""SAM proposals + frozen CLIP + conservative depth-verified graph grounding."""
import json
import os
import cv2
import numpy as np
import torch
from PIL import Image
from pano_segment.local_utils.config_utils import parse_config_utils
from pano_segment.models import build_sam_clip_text_ins_segmentor
from pano_segment.models.clip import tokenize
from utils.scene_geometry import panorama_points
from utils.semantic_grounding import validate_graph, grounding_text, mask_geometry, ground_graph


class PanoSemanticSegmentor:
    def __init__(self, image_path, scene_graph):
        self.image_path = image_path
        self.scene_graph = validate_graph(scene_graph)
        self.bindings = {}
        self.verified_relations = []
        self.anchor_ids = []

    @torch.no_grad()
    def segment(self, save_dir, depth, image=None):
        os.makedirs(save_dir, exist_ok=True)
        if image is None:
            image = np.asarray(Image.open(self.image_path).convert('RGB').resize((depth.shape[1], depth.shape[0])))
        nodes = list(self.scene_graph['objects'])
        label_map = np.zeros(depth.shape, dtype=np.int32)
        if not self.scene_graph['major']:
            return label_map, 0
        cfg = parse_config_utils.Config(config_path='./pano_segment/config/insseg.yaml')
        model = build_sam_clip_text_ins_segmentor(cfg=cfg)
        model.clip_model.eval().requires_grad_(False)
        # Shift the seam so a physical object crossing it can be proposed intact.
        candidates = []
        for shift in (0, image.shape[1]//2):
            proposals = model._generate_sam_mask(np.roll(image, shift, axis=1))
            for mask, stability in zip(proposals['segmentations'], proposals['stability_scores']):
                candidates.append((np.roll(mask, -shift, axis=1).astype(bool), stability))
        # Suppress duplicate physical masks, never merge all masks of a category.
        masks = []
        mask_bounds = []
        for mask, _ in sorted(candidates, key=lambda item: -item[1]):
            if not mask.any():
                continue
            yy, xx = np.where(mask)
            area = len(xx)
            bounds = (yy.min(), yy.max()+1, xx.min(), xx.max()+1, area)
            duplicate = False
            for old, (oy0, oy1, ox0, ox1, old_area) in zip(masks, mask_bounds):
                y0, y1 = max(bounds[0], oy0), min(bounds[1], oy1)
                x0, x1 = max(bounds[2], ox0), min(bounds[3], ox1)
                # Exact cheap upper bound before touching full-resolution masks.
                upper = min(max(y1-y0, 0)*max(x1-x0, 0), area, old_area)
                if upper/max(area, old_area) <= .8:
                    continue
                intersection = np.count_nonzero(mask[y0:y1, x0:x1] & old[y0:y1, x0:x1])
                if intersection/(area+old_area-intersection) > .8:
                    duplicate = True
                    break
            if duplicate:
                continue
            masks.append(mask)
            mask_bounds.append(bounds)
        del candidates
        if not masks:
            return label_map, 0
        texts = [grounding_text(self.scene_graph, node) for node in nodes]
        text_features = model.clip_model.encode_text(tokenize(texts, truncate=True).to(model.device)).float()
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        image_features = []
        for mask in masks:
            # Circular crop keeps seam-crossing instances contiguous for CLIP.
            cols = np.where(mask.any(0))[0]
            gaps = np.diff(np.r_[cols, cols[0]+mask.shape[1]])
            start = cols[(int(gaps.argmax())+1) % len(cols)]
            rolled_mask = np.roll(mask, -int(start), axis=1)
            rolled_image = np.roll(image, -int(start), axis=1)
            yy, xx = np.where(rolled_mask)
            masked = rolled_image * rolled_mask[..., None]
            crop = masked[yy.min():yy.max()+1, xx.min():xx.max()+1]
            tensor = model.clip_preprocess(Image.fromarray(crop)).unsqueeze(0).to(model.device)
            image_features.append(model.clip_model.encode_image(tensor).float())
        image_features = torch.nn.functional.normalize(torch.cat(image_features), dim=-1)
        unary = (text_features @ image_features.T).cpu().numpy()
        points = panorama_points(depth)
        geometries = [mask_geometry(points, m) for m in masks]
        self.bindings, self.verified_relations = ground_graph(self.scene_graph, unary, geometries)
        self.anchor_ids = [n for n in self.scene_graph['major'] if n in self.bindings]
        self.anchor_geometries = [geometries[self.bindings[n]] for n in self.anchor_ids]
        # Labels are only an optional visualization; actual boxes use immutable masks.
        for label, node in enumerate(self.anchor_ids, 1):
            label_map[masks[self.bindings[node]]] = label
        np.savez_compressed(os.path.join(save_dir, 'bound_masks.npz'),
                            **{n: masks[j] for n, j in self.bindings.items()})
        with open(os.path.join(save_dir, 'grounding.json'), 'w') as f:
            json.dump({'bindings': self.bindings, 'anchors': self.anchor_ids,
                       'unresolved': [n for n in nodes if n not in self.bindings],
                       'verified_relations': self.verified_relations}, f, indent=2)
        return label_map, len(self.anchor_ids)
