from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    DDIMScheduler,
)
import torch
from segment_anything import sam_model_registry

def build_SD_pipeline_based_on_input(
    checkpoint_path: str, device: str = "cuda", pipeline_type: str = None, **kwargs
):
    """

    :param checkpoint_path: Checkpoint path to Diffusers Type Folder Containing the VAE, UNET, Scheduler, text_encoder
    tokenizer
    :param device: string name of the device used for DL and inference. Default is 'cuda'
    :param pipeline_type: one of the following types: "Inpaint", "Img2Img", "Text2Img", or "DreamBooth"
    :return: Pipeline for Stable Diffusion given a model
    """
    valid_pipeline_types = ["Inpaint", "Img2Img", "Text2Img" ]
    assert (
        pipeline_type in valid_pipeline_types
    ), "pipeline specified cannot be NoneType. Choose from the following: {}".format(
        valid_pipeline_types
    )

    assert checkpoint_path, "Checkpoint_path cannot be NoneType"

    if pipeline_type == "Inpaint":

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            checkpoint_path, torch_dtype=torch.float16
        )


    elif pipeline_type == "Img2Img":
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            checkpoint_path, torch_dtype=torch.float16
        )

    elif pipeline_type == "Text2Img":
        pipe = StableDiffusionPipeline.from_pretrained(
            checkpoint_path, torch_dtype=torch.float16
        )


    else:
        raise KeyError(
            'Pipeline specified does not exist. Check that input is one of the following:["Inpaint","Img2Img","Text2Img","DreamBooth"] '
        )

    pipe = pipe.to(device)

    # Enables DDIM scheduler for Stable Diffusion Models
    if kwargs.get("scheduler") == "DDIM":
        # This Scheduler may need to be tuned
        pipe.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            steps_offset=1,
            clip_sample=False,
        )

    return pipe

def build_SAM(checkpoint_path:str,device:str ='cuda'):
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device)
    return sam
