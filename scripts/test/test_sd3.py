import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

# transformer = SD3Transformer2DModel.from_pretrained(
#     "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="transformer", torch_dtype=torch.float16
# )
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

image = pipe(
    "A cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]
image
