from modules.text_to_img_SD import make_txt_2_img_prediction
from modules.img_2_img_SD import make_img_2_img_prediction
from modules.in_painting_SD import make_inpainting_prediction
from utils.pipeline_loader import build_SD_pipeline_based_on_input, load_lora_weights_safetensor_to_diffusers
from utils.create_masks_from_prompts import create_border_mask, build_SAM, build_boundingbox_image_stack_from_masks, \
    create_all_mask_sam, clip_scoring, close_mask_holes
from PIL import Image
import cv2
import numpy as np

def inpainting_api(prompt: str,
                   negative_prompt: str,
                   segmentation_prompt: str,
                   device: str,
                   seeds,
                   height: int, width: int,
                   cfg_val: int,
                   num_inference_steps: int,
                   reference_image_path,
                   checkpoint_directory_SD,
                   checkpoint_path_SAM,
                   lora_path,
                   border_mask_width:int,
                   img2img_strength_first_pass_val,
                   img2img_strength_second_pass_val,
                   scheduler='DDIM',
                   lora_alpha=1,
                   make_lossless_superimposition=True):
    """
    API call for running ML model for in-painting.

    :param prompt: generation prompt
    :param negative_prompt: negative generative prompt
    :param segmentation_prompt: item to segment and leave unaltered prompt
    :param device: Cuda Device
    :param seeds: any beginning seeds
    :param height: desired output image height
    :param width: desired output image width
    :param cfg_val: Classifier Free Guidance Value. How strongly SD adhers to prompt
    :param num_inference_steps: Number of steps desired for inference
    :param reference_image_path: Original Image to segment and In-paint on
    :param checkpoint_directory_SD: Checkpoint to load SD model from
    :param checkpoint_path_SAM: Checkpoint for Segment Anything Model
    :param lora_path: Path to Lora Model
    :param border_mask_width: Desired Border Mask Width
    :param img2img_strength_first_pass_val: Denoising Strength on first Pass
    :param img2img_strength_second_pass_val: Denoising Strength on second Pass (border pass)
    :param scheduler: Desired sampling Scheduler
    :param lora_alpha: Lora Adherence Alpha. How strongly to implement Lora weights to model
    :param make_lossless_superimposition: To superimpose image or not. Default: True
    :return: Returns a list of generated Images of form [first_pass_image, second_pass_image]
    """

    pipeline = build_SD_pipeline_based_on_input(checkpoint_directory_SD, device, pipeline_type="Inpaint",
                                                scheduler=scheduler)

    if lora_path:
        pipeline = load_lora_weights_safetensor_to_diffusers(pipeline, lora_path, alpha=lora_alpha)

    # Build mask here
    #img = cv2.imread(reference_image_path) # for some reason SAM has issues with non-CV2 images
    pil_image = Image.open(reference_image_path)
    opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2RGB)
    orig_image_pil = Image.fromarray(img)
    sam = build_SAM(checkpoint_path=checkpoint_path_SAM, device='cuda')
    masks = create_all_mask_sam(img, sam=sam)
    img_bboxes = build_boundingbox_image_stack_from_masks(masks, orig_image_pil)
    probabilities, values, indices = clip_scoring(img_bboxes, segmentation_prompt)

    outmask_holes_filled, outmask_contour_filled = close_mask_holes(masks[indices[0]]['segmentation'])
    border_mask = create_border_mask(outmask_contour_filled, border_mask_width)

    output_images = []

    image_out = make_inpainting_prediction(
        pipeline,
        prompt,
        negative_prompt,
        init_img=reference_image_path,
        mask_img=outmask_contour_filled,
        device=device,
        seeds=seeds,
        height=height,
        width=width,
        CFG=cfg_val,
        num_inference_steps=num_inference_steps,
        img2img_strength=img2img_strength_first_pass_val,
        make_lossless_superimposition=make_lossless_superimposition
    )

    output_images.append(image_out)

    image_out = make_inpainting_prediction(
        pipeline,
        prompt,
        negative_prompt,
        init_img=image_out,
        mask_img=border_mask,
        device=device,
        seeds=seeds,
        height=height,
        width=width,
        CFG=cfg_val,
        num_inference_steps=num_inference_steps,
        img2img_strength=img2img_strength_second_pass_val,
        make_lossless_superimposition=make_lossless_superimposition
    )

    output_images.append(image_out)

    return output_images


def img_to_image_api(prompt: str,
                     negative_prompt: str,
                     device: str,
                     seeds,
                     height: int, width: int,
                     cfg_val: int,
                     num_inference_steps: int,
                     img2img_strength,
                     reference_image_path,
                     checkpoint_directory_SD,
                     scheduler='DDIM'):
    pipeline = build_SD_pipeline_based_on_input(checkpoint_directory_SD, device, pipeline_type="Img2Img",
                                                scheduler=scheduler)
    image_out = make_img_2_img_prediction(
        pipeline,
        prompt,
        negative_prompt,
        device=device,
        seeds=seeds,
        height=height,
        width=width,
        CFG=cfg_val,
        num_inference_steps=num_inference_steps,
        img2img_strength=img2img_strength,
        reference_img_path=reference_image_path,
    )
    return image_out


def text_to_image_api(prompt: str,
                      negative_prompt: str,
                      device: str,
                      seeds,
                      height: int, width: int,
                      cfg_val: int,
                      num_inference_steps: int,
                      img2img_strength,
                      reference_image_path,
                      checkpoint_directory_SD,
                      scheduler='DDIM'):

    pipeline = build_SD_pipeline_based_on_input(checkpoint_directory_SD, device, pipeline_type="Text2Img",
                                                scheduler=scheduler)
    image_out = make_txt_2_img_prediction(
        pipeline,
        prompt,
        negative_prompt,
        device=device,
        seeds=seeds,
        height=height,
        width=width,
        CFG=cfg_val,
        num_inference_steps=num_inference_steps,
        img2img_strength=img2img_strength,
        reference_img_path=reference_image_path,
    )
    return image_out
