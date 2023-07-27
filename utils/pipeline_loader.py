from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    DDIMScheduler,
)
import torch
from segment_anything import sam_model_registry
from safetensors.torch import load_file


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

    # turn off NSFW filter
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))


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

def load_lora_weights_safetensor_to_diffusers(pipeline, checkpoint_path, alpha = 0.75):
    """
    Usage:
            lora_model = lora_models + "/" + opt.lora + ".safetensors"
            self.pipe = load_lora_weights(self.pipe, lora_model)

    :param pipeline: input SD pipeline
    :param checkpoint_path: Lora Checkpoint Path
    :return: Pipeline with lora added to attention layers
    """
    # load base model
    pipeline.to("cuda")
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device="cuda")
    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return pipeline

