"""Metric3D v2 joint depth/normal inference with the official preprocessing."""
import os
import torch
import torch.nn.functional as F


class Metric3DJointPredictor:
    def __init__(self):
        repo = os.environ.get('METRIC3D_REPO')
        backbone = os.environ.get('METRIC3D_MODEL', 'metric3d_vit_large')
        if not backbone.startswith('metric3d_vit_'):
            raise ValueError('Metric3D v2 ViT weights are required for joint normals and depth')
        self.model = torch.hub.load(repo or 'yvanyin/metric3d', backbone,
                                    source='local' if repo else 'github', pretrain=True,
                                    **({} if repo else {'trust_repo': True}))
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def predict(self, images, focal_lengths):
        _, _, h, w = images.shape
        self.model.to(images.device)
        scale = min(616/h, 1064/w)
        rh, rw = int(h*scale), int(w*scale)
        top, left = (616-rh)//2, (1064-rw)//2
        mean = images.new_tensor([123.675, 116.28, 103.53])[None, :, None, None]
        std = images.new_tensor([58.395, 57.12, 57.375])[None, :, None, None]
        rgb = F.interpolate(images*255, size=(rh, rw), mode='bilinear', align_corners=False)
        rgb = F.pad((rgb-mean)/std, (left, 1064-rw-left, top, 616-rh-top))
        depth, _, output = self.model.inference({'input': rgb})
        if depth.ndim == 3:
            depth = depth[:, None]
        if 'prediction_normal' not in output:
            raise RuntimeError('Metric3D checkpoint did not produce surface normals')
        normal = output['prediction_normal'][:, :3]
        depth = F.interpolate(depth[..., top:top+rh, left:left+rw], size=(h, w), mode='bilinear', align_corners=False)
        depth *= focal_lengths.reshape(-1, 1, 1, 1).to(depth)*scale/1000
        normal = F.interpolate(normal[..., top:top+rh, left:left+rw], size=(h, w), mode='bilinear', align_corners=False)
        return depth.clamp_min(0), F.normalize(normal, dim=1)
