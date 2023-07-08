from PIL import Image
import warnings
from utils.create_embeddings import build_text_embeddings

warnings.simplefilter(action="ignore", category=FutureWarning)


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

    assert kwargs.get("init_img"), print("No initial Image path")
    assert kwargs.get("mask_img"), print("No mask Image path")
    assert kwargs.get("height"), print("No height specified")
    assert kwargs.get("width"), print("No width specified")

    # make both init and mask image equal size
    init_img = resize_png_image(kwargs["init_img"], kwargs["width"], kwargs["height"])
    mask_img = resize_png_image(kwargs["mask_img"], kwargs["width"], kwargs["height"])

    # Create assert statement on image size (B, H, W, 1).
    # For mask

    if kwargs.get("device"):
        device = kwargs["device"]
    else:
        device = "cpu"

    # Manual Embedding of prompt. This is to counter the 77 Token limit imposed by CLIP
    prompt_embeds, negative_prompt_embeds = build_text_embeddings(
        pipeline, prompt, negative_prompt, device
    )

    image = pipeline(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        image=init_img,
        mask_image=mask_img,
        guidance_scale=kwargs["CFG"],
        num_inference_steps=kwargs["num_inference_steps"],
        strength=kwargs["img2img_strength"],
    ).images[0]

    return image
