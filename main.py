from matplotlib import pyplot as plt
from modules.text_to_img_SD import make_txt_2_img_prediction
from modules.img_2_img_SD import make_img_2_img_prediction
from modules.in_painting_SD import make_inpainting_prediction
from utils.pipeline_loader import build_SD_pipeline_based_on_input
from utils.create_masks_from_prompts import build_SAM, build_boundingbox_image_stack_from_masks, create_all_mask_sam, \
    clip_scoring, close_mask_holes
from PIL import Image

import cv2


def main():
    # inputs
    seeds = None
    scheduler = "DDIM"
    checkpoint_directory_SD = r"D:\Ecommerce_FakeModel\Models_Converted\Lyriel_Diffusers"
    checkpoint_path_SAM = r'D:\Ecommerce_FakeModel\SAM_Checkpoint\sam_vit_h_4b8939.pth'

    device = "cuda"
    prompt = "A sexy Asian model wearing a graphic T-Shirt, (blue hair), Plain White Background,masterpiece, best quality, ultra-detailed, solo"
    negative_prompt = (
        "lowres, bad anatomy, bad hands, text, error, "
        "missing fingers, extra digit, fewer digits, cropped,"
        " worst quality, low quality, normal quality, jpeg artifacts,"
        " signature, watermark, username, blurry, artist name, young, loli"
    )
    segmentation_prompt = 'a photo of a graphic T-Shirt'

    num_inference_steps = 20  # The number of denoising steps. Higher number usually leads to higher quality
    CFG = 7
    height = 864
    width = 592

    img2img_strength = 0.8
    reference_image_path = r'D:\ArtDesigns\Forselling\GirlWearingLion.PNG'
    """
    img2img_strength (float, optional, defaults to 0.8) — Conceptually, says how much to transform the reference img.
    Must be between 0 and 1. image will be used as a starting point, adding more noise to it the larger the strength.
    The number of denoising steps depends on the amount of noise initially added. When strength is 1, added noise will
    be maximum and the denoising process will run for the full number of iterations specified in num_inference_steps.
    A value of 1, therefore, essentially ignores image.
    """

    # For text to img
    run_text_to_image = False
    if run_text_to_image:
        pipeline = build_SD_pipeline_based_on_input(
            checkpoint_directory_SD, device, pipeline_type="Text2Img", scheduler=scheduler
        )
        image_out = make_txt_2_img_prediction(
            pipeline,
            prompt,
            negative_prompt,
            device=device,
            seeds=seeds,
            height=height,
            width=width,
            CFG=CFG,
            num_inference_steps=num_inference_steps,
            img2img_strength=img2img_strength,
            reference_img_path=reference_image_path,
        )

    # For Image to Image
    run_image_to_image = False
    if run_image_to_image:
        pipeline = build_SD_pipeline_based_on_input(
            checkpoint_directory_SD, device, pipeline_type="Img2Img", scheduler=scheduler
        )

        image_out = make_img_2_img_prediction(
            pipeline,
            prompt,
            negative_prompt,
            device=device,
            seeds=seeds,
            height=height,
            width=width,
            CFG=CFG,
            num_inference_steps=num_inference_steps,
            img2img_strength=img2img_strength,
            reference_img_path=reference_image_path,
        )

    # inpaint img2img
    run_inpainting = True
    if run_inpainting:
        pipeline = build_SD_pipeline_based_on_input(
            checkpoint_directory_SD, device, pipeline_type="Inpaint", scheduler=scheduler
        )

        # Build mask here
        img = cv2.imread(reference_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_image_pil = Image.fromarray(img)

        sam = build_SAM(checkpoint_path=checkpoint_path_SAM, device='cuda')
        masks = create_all_mask_sam(img, sam=sam)

        img_bboxes = build_boundingbox_image_stack_from_masks(masks, orig_image_pil)
        probabilities, values, indices = clip_scoring(img_bboxes, segmentation_prompt)

        outmask_holes_filled, outmask_contour_filled = close_mask_holes(masks[indices[0]]['segmentation'])

        # note that for inpainting. Black pixels are preserved and White pixels are repainted

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
            CFG=CFG,
            num_inference_steps=num_inference_steps,
            img2img_strength=img2img_strength
        )
        plt.imshow(image_out)
        plt.show()
        print()

if __name__ == "__main__":
    main()
