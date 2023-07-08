from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def build_Img2Img_SD_pipeline(checkpoint_path: str, device: str = 'cuda', **kwargs):
    """
    :param checkpoint_path: Checkpoint path to Diffusers Type Folder Containing the VAE, UNET, Scheduler, text_encoder
    tokenizer
    :param device: string name of the device used for DL and inference. Default is 'cuda'
    :return: Pipeline for Stable Diffusion given a model
    """

    assert checkpoint_path, 'Checkpoint_path is empty'

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(checkpoint_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    # Enables DDIM scheduler for Stable Diffusion Models
    if kwargs.get('scheduler') == 'DDIM':
        # This Scheduler may need to be tuned
        pipe.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                                       beta_schedule="scaled_linear", steps_offset=1, clip_sample=False)

    return pipe


def resize_png_image(image_path, new_width, new_height):
    """
    resizes an input image to fit the desired new paramters
    """
    # Open the image
    image = Image.open(image_path)

    # Resize the image
    resized_image = image.resize((new_width, new_height), resample=Image.BILINEAR)

    return resized_image


def make_img_2_img_prediction(pipeline, prompt: str, negative_prompt: str, **kwargs):
    """
    :param pipeline: The Diffusion pipeline used to generate Images
    :param prompt: The text prompt
    :param negative_prompt: The negative text prompt. i,e What you don't want
    :param kwargs: relevant arguments for the pipeline
    :return: Generated Images
    """

    assert kwargs.get('reference_img_path'), print('No reference Image path')
    assert kwargs.get('height'), print('No height specified')
    assert kwargs.get('width'), print('No width specified')

    reference_img = resize_png_image(kwargs['reference_img_path'], kwargs['width'], kwargs['height'])

    if kwargs.get('device'):
        device = kwargs['device']
    else:
        device = 'cpu'

    # Manual Embedding of prompt. This is to counter the 77 Token limit imposed by CLIP
    max_length = pipeline.tokenizer.model_max_length

    input_ids = pipeline.tokenizer(prompt, truncation=False, padding="max_length",
                                   return_tensors="pt").input_ids
    input_ids = input_ids.to(device)

    negative_ids = pipeline.tokenizer(negative_prompt, truncation=False, padding="max_length",
                                      max_length=input_ids.shape[-1], return_tensors="pt").input_ids
    negative_ids = negative_ids.to(device)

    concat_embeds = []
    neg_embeds = []
    for i in range(0, input_ids.shape[-1], max_length):
        concat_embeds.append(pipeline.text_encoder(input_ids[:, i: i + max_length])[0])
        neg_embeds.append(pipeline.text_encoder(negative_ids[:, i: i + max_length])[0])

    prompt_embeds = torch.cat(concat_embeds, dim=1)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

    image = pipeline(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, image=reference_img,
                     guidance_scale=kwargs['CFG'], num_inference_steps=kwargs['num_inference_steps'],
                     strength=kwargs['img2img_strength']).images[0]

    return image
