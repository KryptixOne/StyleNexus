import cv2
from PIL import Image
import warnings
from utils.create_embeddings import build_text_embeddings
import numpy as np
from werkzeug.datastructures import FileStorage
from matplotlib import pyplot as plt
warnings.simplefilter(action="ignore", category=FutureWarning)


def resize_mask(image_path, new_width, new_height):
    
    # resizes an input image to fit the desired new paramters
    
    # Open the image
    image_path = cv2.bitwise_not(image_path)
    image = Image.fromarray(image_path)

    # Resize the image

    resized_image = image.resize((new_width, new_height), resample=Image.BILINEAR)

    return resized_image


def resize_image(image_path, new_width, new_height):
    """
    Resizes an input image to fit the desired new parameters.

    Parameters:
        image_path: str or numpy.ndarray or PIL.Image.Image
            The input image file path, numpy array, or a PIL image.
        new_width: int
            The new width for the resized image.
        new_height: int
            The new height for the resized image.

    Returns:
        PIL.Image.Image
            The resized image.
    """

    # If the input is a file path
    if isinstance(image_path, str):
        image = Image.open(image_path)

    # If the input is a numpy array
    elif isinstance(image_path, np.ndarray):
        image = Image.fromarray(image_path)

    # If the input is already a PIL image
    elif isinstance(image_path, Image.Image):
        image = image_path

    elif isinstance(image_path, FileStorage):
        image = Image.open(image_path.stream)

    else:
        raise ValueError("Invalid input type. The 'image_path' should be a str, numpy.ndarray, or PIL.Image.Image.")

    # Resize the image
    resized_image = image.resize((new_width, new_height), resample=Image.BILINEAR)

    return resized_image


"""
def resize_mask(mask, new_width, new_height):
    '
    Resizes an input mask to fit the desired new parameters and processes the mask.

    Parameters:
        mask: str or numpy.ndarray or PIL.Image.Image
            The input mask file path, numpy array, or a PIL image.
        new_width: int
            The new width for the resized mask.
        new_height: int
            The new height for the resized mask.

    Returns:
        PIL.Image.Image
            The processed and resized binary mask (pixel values: 0 or 1).
    '
    # Function to process the mask to binary
    def process_mask(mask):
        mask = mask.convert("L")  # Convert to grayscale
        mask = np.array(mask)  # Convert to NumPy array
        mask = (mask > 0).astype(np.uint8)  # Threshold to binary (pixel values: 0 or 1)
        return mask

    # Process the mask
    if isinstance(mask, str):
        mask = Image.open(mask)
    elif isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    elif isinstance(mask, Image.Image):
        pass
    else:
        raise ValueError("Invalid input type for 'mask'. It should be a str, numpy.ndarray, or PIL.Image.Image.")

    # Always process the mask to ensure it's binary
    mask = process_mask(mask)

    # Resize the mask
    resized_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # Invert the binary mask using cv2.bitwise_not
    resized_mask = cv2.bitwise_not(resized_mask)

    # Convert back to PIL Image
    resized_mask = Image.fromarray(resized_mask)

    return resized_mask

"""
def superimpose_images(diffused_image, orig_image, mask):
    # Convert PIL images to NumPy arrays
    diffused_image = np.array(diffused_image.convert("RGB"))
    orig_image = np.array(orig_image.convert("RGB"))
    mask = np.array(mask.convert("L"))

    # Binarize the mask to have only 0 and 255 intensity values
    _, mask_binarized = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # Invert the mask
    mask_inverted = cv2.bitwise_not(mask_binarized)

    # Rescale the original image to match the size of the diffused image
    orig_image_rescaled = cv2.resize(orig_image, (diffused_image.shape[1], diffused_image.shape[0]))

    # Rescale the mask to match the original image's dimensions
    mask_rescaled = cv2.resize(mask_inverted, (orig_image_rescaled.shape[1], orig_image_rescaled.shape[0]))

    # Extract the masked section from the original image
    masked_section = cv2.bitwise_and(orig_image_rescaled, orig_image_rescaled, mask=mask_rescaled)

    # Extract the non-masked section from the diffused image
    non_masked_section = cv2.bitwise_and(diffused_image, diffused_image, mask=mask_binarized)

    # Combine the masked section and non-masked section to superimpose the extracted part on the diffused image
    superimposed_image = cv2.add(masked_section, non_masked_section)

    # Convert the result back to PIL image
    superimposed_image = Image.fromarray(superimposed_image)

    return superimposed_image


# Example usage:
# from PIL import Image
# diffused_image = Image.open('diffused_image.jpg')
# orig_image = Image.open('orig_image.jpg')
# mask = Image.open('mask.jpg').convert('L')
# result = superimpose_images(diffused_image, orig_image, mask)
# result.show()


def make_inpainting_prediction(pipeline, prompt: str, negative_prompt: str, **kwargs):
    """
    :param pipeline: The Diffusion pipeline used to generate Images
    :param prompt: The text prompt
    :param negative_prompt: The negative text prompt. i,e What you don't want
    :param kwargs: relevant arguments for the pipeline
    :return: Generated Images
    """

    assert kwargs.get("init_img"), print("No initial Image or image path")
    assert kwargs.get("mask_img") is not None, print("No mask Image or image path ")
    assert kwargs.get("height"), print("No height specified")
    assert kwargs.get("width"), print("No width specified")
    height = kwargs.get("height")
    width = kwargs.get("width")

    # make both init and mask image equal size
    init_img = resize_image(kwargs["init_img"], kwargs["width"], kwargs["height"])
    mask_img = resize_mask(kwargs["mask_img"], kwargs["width"], kwargs["height"])

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
        height=height,
        width=width,
        guidance_scale=kwargs["CFG"],
        num_inference_steps=kwargs["num_inference_steps"],
        strength=kwargs["img2img_strength"],
    ).images[0]

    if kwargs.get('make_lossless_superimposition'):
        image = superimpose_images(image, init_img, mask_img)

    return image
