import torch
import torch.nn as nn
from peft import LoraConfig
from packaging import version
from diffusers import UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available

class PositionalEncoding(nn.Module):
    def __init__(self, input_dims, num_freqs, include_input=True):
        super().__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.log_sampling = True
        self.include_input = include_input
        self.periodic_fns = [torch.sin, torch.cos]
        
        if self.log_sampling:
            self.freq_bands = 2.**torch.linspace(0., self.num_freqs - 1, steps=self.num_freqs)
        else:
            self.freq_bands = torch.linspace(1., 2.**(self.num_freqs - 1), steps=self.num_freqs)
            
        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dims
        self.out_dim += self.input_dims * self.num_freqs * len(self.periodic_fns)

    def forward(self, x):
        out = []
        freq_bands = self.freq_bands.to(x.device)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                # print(p_fn(x * freq))
                out.append(p_fn(x * freq))

        if self.include_input:
            out.append(x)

        return torch.cat(out, dim=-1)

class UNetWithCam(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        self.unet.requires_grad_(False)
        # print(self.unet)
        self.config = self.unet.config
        
        self.pos_encoder = PositionalEncoding(input_dims=3, num_freqs=10)
        # self.cam_encoder = nn.Sequential(
        #     nn.Linear(self.pos_encoder.out_dim+3, args.cam_latent_dim),
        #     nn.ReLU(),
        #     nn.Linear(args.cam_latent_dim, self.unet.config.cross_attention_dim)
        # )
        self.cam_encoder = nn.Sequential(
            nn.Linear(self.pos_encoder.out_dim+3, args.cam_latent_dim),
            nn.LayerNorm(args.cam_latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.cam_latent_dim, args.cam_latent_dim),
            nn.LayerNorm(args.cam_latent_dim),
            nn.ReLU(),
            nn.Linear(args.cam_latent_dim, self.unet.config.cross_attention_dim)
        )
        for layer in self.cam_encoder:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        self.use_learnable_pose_token = False
        if args.use_learnable_pose_token:
            self.use_learnable_pose_token = True
            self.pose_token = nn.Parameter(torch.randn(1, 1, self.unet.config.cross_attention_dim))
            # self.pose_scale = nn.Parameter(torch.ones(1))

        # Add adapter and make sure the trainable params are in float32.
        unet_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0", "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0"], #"attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0"
        )
        if args.enable_lora: 
            self.unet.add_adapter(unet_lora_config)
        else:
            for name, param in self.unet.named_parameters():
                if "attn1" in name or "attn2" in name:
                    param.requires_grad_(True)
        if args.mixed_precision == "fp16":
            for param in self.unet.parameters():
                # only upcast trainable parameters (LoRA) into fp32
                if param.requires_grad:
                    param.data = param.to(torch.float32)
        
        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers
                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
            

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        pose=None,
        cam_encoding_strategy="expand",  # "concat", "replace", "expand"
        **kwargs
    ):
        if pose is not None:
            # encoder_hidden_states = encoder_hidden_states[0].unsqueeze(0)

            position = pose[:, :3]
            pose_encoded = self.pos_encoder(position)
            direction = pose[:, 3:]
            pose_emb = self.cam_encoder(torch.cat([pose_encoded, direction], dim=-1)) # (batch, cross_dim)
            
            batch_size = encoder_hidden_states.size(0)
            seq_len = encoder_hidden_states.size(1)
            
            if cam_encoding_strategy == "replace":
                # 方案1：完全替换文本编码为相机编码
                pose_emb = pose_emb.unsqueeze(1) # (batch, 1, cross_dim)
                
                if self.use_learnable_pose_token:
                    pose_token = self.pose_token.expand(batch_size, -1, -1)
                    pose_emb = pose_emb + pose_token
                else:
                    pose_emb = pose_emb.expand(batch_size, -1, -1)
                # 将单个pose_emb扩展到整个序列长度
                encoder_hidden_states = pose_emb.expand(-1, seq_len, -1)
                
            elif cam_encoding_strategy == "expand":
                # 方案2：扩展相机编码到整个序列
                pose_emb = pose_emb.unsqueeze(1).repeat(2, 1, 1) # (batch, 1, cross_dim)
                
                if self.use_learnable_pose_token:
                    pose_token = self.pose_token.expand(batch_size, -1, -1)
                    # print(pose_emb.shape)
                    # print(pose_token.shape)
                    pose_emb = pose_emb + pose_token
                    # print(pose_emb.shape)
                else:
                    pose_emb = pose_emb.expand(batch_size, -1, -1)
                
                # 广播相机信息到所有位置
                pose_expanded = pose_emb.expand(-1, seq_len, -1)
                encoder_hidden_states = encoder_hidden_states + pose_expanded
                
            else:
                # 方案3：默认拼接方法（原始方法）
                pose_emb = pose_emb.unsqueeze(1) # (batch, 1, cross_dim)
                if self.use_learnable_pose_token:
                    # 拼接 learnable token（broadcast batch 维度）
                    pose_token = self.pose_token.expand(encoder_hidden_states.size(0), -1, -1)
                    # 最终的 pose token = learned token + projected pose
                    pose_emb = pose_emb + pose_token
                else:
                    pose_emb = pose_emb.expand(encoder_hidden_states.size(0), -1, -1)

                # 把 pose token 拼接到 encoder_hidden_states 前面
                # 注意：cross-attention 会看到这个额外 token 并可使用它作为条件信息
                encoder_hidden_states = torch.cat([pose_emb, encoder_hidden_states], dim=1)
        

        return self.unet.forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs
        )

class UNetWOCam(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        self.unet.requires_grad_(False)
        self.config = self.unet.config

        # Add adapter and make sure the trainable params are in float32.
        unet_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0"]#, "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0"],
        )
        if args.enable_lora: 
            self.unet.add_adapter(unet_lora_config)
        else:
            for name, param in self.unet.named_parameters():
                if "attn1" in name:
                    param.requires_grad_(True)
        if args.mixed_precision == "fp16":
            for param in self.unet.parameters():
                # only upcast trainable parameters (LoRA) into fp32
                if param.requires_grad:
                    param.data = param.to(torch.float32)
        
        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers
                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
            

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        pose=None,
        **kwargs
    ):
        
        return self.unet.forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs
        )