from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def resize_png_image(image_path, new_width, new_height):
    """
    resizes an input image to fit the desired new paramters
    """
    # Open the image
    image = Image.open(image_path)

    # Resize the image
    resized_image = image.resize((new_width, new_height), resample=Image.BILINEAR)

    return resized_image


def make_Inpainting_prediction(pipeline, prompt: str, negative_prompt: str, **kwargs):
    """
    :param pipeline: The Diffusion pipeline used to generate Images
    :param prompt: The text prompt
    :param negative_prompt: The negative text prompt. i,e What you don't want
    :param kwargs: relevant arguments for the pipeline
    :return: Generated Images
    """

    assert kwargs.get('init_img'), print('No initial Image path')
    assert kwargs.get('mask_img'), print('No mask Image path')
    assert kwargs.get('height'), print('No height specified')
    assert kwargs.get('width'), print('No width specified')

    # make both init and mask image equal size
    init_img = resize_png_image(kwargs['init_img'], kwargs['width'], kwargs['height'])
    mask_img = resize_png_image(kwargs['mask_img'], kwargs['width'], kwargs['height'])

    # Create assert statement on image size (B, H, W, 1).
    # For mask


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

    image = pipeline(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, image=init_img,mask_image=mask_img,
                     guidance_scale=kwargs['CFG'], num_inference_steps=kwargs['num_inference_steps'],
                     strength=kwargs['img2img_strength']).images[0]

    return image
