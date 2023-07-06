from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

from tqdm.auto import tqdm

"""
civitaiModel = r'D:\StableDiffusionWebUI\stable-diffusion-webui\models\Stable-diffusion\lyriel_v16.safetensors'
convertedModel = r'D:\Ecommerce_FakeModel\Models_Converted\Lyriel_Diffusers' #diffusion_pytorch_model

# Preload Models
print('loading pretrained models...')
vae = AutoencoderKL.from_pretrained(convertedModel,subfolder = 'vae')
tokenize = CLIPTokenizer.from_pretrained(convertedModel,subfolder='tokenizer')
text_encoder = CLIPTextModel.from_pretrained(convertedModel,subfolder= 'text_encoder')
unet = UNet2DConditionModel.from_pretrained(convertedModel,subfolder='unet')
scheduler = PNDMScheduler.from_pretrained(convertedModel,subfolder='scheduler') # Use a different Scheduler Later
print('...loaded')

torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)
"""
convertedModel = r'D:\Ecommerce_FakeModel\Models_Converted\Lyriel_Diffusers'
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(convertedModel,torch_dtype=torch.float16)

pipe = pipe.to("cuda")

# how to manage CLIP 77 Token Limit *************************************
prompt = '25 * "a photo of an astronaut riding a horse on mars"' # a ridiculously large  prompt
negative_prompt = 'THINGS WE DONT WANT'
max_length = pipe.tokenizer.model_max_length

input_ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to("cuda")

negative_ids = pipe.tokenizer(negative_prompt, truncation=False, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids
negative_ids = negative_ids.to("cuda")

concat_embeds = []
neg_embeds = []
for i in range(0, input_ids.shape[-1], max_length):
    concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
    neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])

prompt_embeds = torch.cat(concat_embeds, dim=1)
negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds).images[0]

#*********************************************************


#image.save("astronaut_rides_horse.png")

#image = pipe(prompt).images[0]