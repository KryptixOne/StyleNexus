from PIL import Image

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from Utils.Create_Embeddings import build_text_embeddings

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
    prompt_embeds, negative_prompt_embeds = build_text_embeddings(pipeline, prompt, negative_prompt, device)

    image = pipeline(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, image=reference_img,
                     guidance_scale=kwargs['CFG'], num_inference_steps=kwargs['num_inference_steps'],
                     strength=kwargs['img2img_strength']).images[0]

    return image
