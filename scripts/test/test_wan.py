# pip install ftfy
import torch
import numpy as np
from diffusers import AutoModel, WanPipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from transformers import UMT5EncoderModel

text_encoder = UMT5EncoderModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="text_encoder", torch_dtype=torch.bfloat16)
vae = AutoModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
transformer = AutoModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16)

# group-offloading
onload_device = torch.device("cuda")
offload_device = torch.device("cpu")
apply_group_offloading(text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="block_level", num_blocks_per_group=4)
transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", use_stream=True)

pipeline = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers", vae=vae, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

prompt = """
The camera rushes from far to near in a low-angle shot, 
revealing a white ferret on a log. It plays, leaps into the water, and emerges, as the camera zooms in 
for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves blanket the ground. 
Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side lighting casts dynamic 
shadows and warm highlights. Medium composition, front view, low angle, with depth of field.
"""
negative_prompt = """
Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, 
low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, 
misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
"""

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=81,
    guidance_scale=5.0,
).frames[0]
export_to_video(output, "output.mp4", fps=16)
