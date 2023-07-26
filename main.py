from matplotlib import pyplot as plt
from modules.text_to_img_SD import make_txt_2_img_prediction
from modules.img_2_img_SD import make_img_2_img_prediction
from modules.in_painting_SD import make_inpainting_prediction
from utils.pipeline_loader import build_SD_pipeline_based_on_input
from utils.create_masks_from_prompts import create_border_mask, build_SAM, build_boundingbox_image_stack_from_masks, \
    create_all_mask_sam, clip_scoring, close_mask_holes
from PIL import Image
import cv2
import random


def main():
    # inputs
    seeds = None
    scheduler = "DDIM"
    checkpoint_directory_SD = r"D:\Ecommerce_FakeModel\Models_Converted\Lyriel_inpaint"
    # checkpoint_directory_SD = r'D:\Ecommerce_FakeModel\Models_Converted\Lyriel_Diffusers'
    checkpoint_path_SAM = r'D:\Ecommerce_FakeModel\SAM_Checkpoint\sam_vit_h_4b8939.pth'

    device = "cuda"
    prompt = " A sexy female model with sunglasses wearing a fitted graphic T-Shirt, Plain White Background,8k uhd, high quality"
    negative_prompt = (
        "(deformed iris, deformed pupils, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, ugly"
    )
    segmentation_prompt = 'a photo of a graphic T-Shirt'

    num_inference_steps_list = [10, 20, 30, 40,
                                50]  # The number of denoising steps. Higher number usually leads to higher quality
    CFG_list = [4, 5, 6, 7] #6 is awesome
    height = 864
    width = 592
    border_mask_width = 32
    img2img_strength = 0.8
    img2img_strength_first_pass = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # 0.9 on first. Heavy alteration should be given
    img2img_strength_second_pass = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8] #0.4 -0.5 best visual # lower to reduce effects of superimposition but also to limit border distortion
    HyperParameterTune_num = 2000
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
            CFG=CFG_list[0],
            num_inference_steps=num_inference_steps_list[0],
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
            CFG=CFG_list[0],
            num_inference_steps=num_inference_steps_list[0],
            img2img_strength=img2img_strength,
            reference_img_path=reference_image_path,
        )

    direct = r'D:/Ecommerce_FakeModel/OutputPics_Issues/Tuning/'
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
        print('loading SAM...')
        sam = build_SAM(checkpoint_path=checkpoint_path_SAM, device='cuda')
        print('loaded')
        print('creating masks...')
        masks = create_all_mask_sam(img, sam=sam)
        print('found ' + str(len(masks)) + ' masks')
        img_bboxes = build_boundingbox_image_stack_from_masks(masks, orig_image_pil)
        probabilities, values, indices = clip_scoring(img_bboxes, segmentation_prompt)

        outmask_holes_filled, outmask_contour_filled = close_mask_holes(masks[indices[0]]['segmentation'])
        border_mask = create_border_mask(outmask_contour_filled, border_mask_width)
        # note that for inpainting. Black pixels are preserved and White pixels are repainted

        for i in range(HyperParameterTune_num):
            CFG_choice = random.randrange(len(CFG_list))
            num_inference_choice = random.randrange(len(num_inference_steps_list))
            img2img_first_choice = random.randrange(len(img2img_strength_first_pass))
            img2img_second_choice = random.randrange(len(img2img_strength_second_pass))

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
                CFG=CFG_list[CFG_choice],
                num_inference_steps=num_inference_steps_list[num_inference_choice],
                img2img_strength=img2img_strength_first_pass[img2img_first_choice],
                make_lossless_superimposition=True
            )
            #plt.imshow(image_out)
            #plt.show()
            output = str(i)+'_num_inference_' + str(
                num_inference_steps_list[num_inference_choice]) + '_img2img2strfirst_' + str(
                img2img_strength_first_pass[img2img_first_choice]) + '_CFG_' + str(CFG_list[CFG_choice])

            image_out.save(direct+output+'.png')

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
                CFG=CFG_list[CFG_choice],
                num_inference_steps=num_inference_steps_list[num_inference_choice],
                img2img_strength=img2img_strength_second_pass[img2img_second_choice],
                make_lossless_superimposition=True
            )
            #plt.imshow(image_out)
            #plt.show()

            output = str(i)+'_num_inference_' + str(
                num_inference_steps_list[num_inference_choice]) + '_img2img2strSecond_' + str(
                img2img_strength_second_pass[img2img_second_choice]) + '_CFG_' + str(CFG_list[CFG_choice])

            image_out.save(direct+output+'.png')


if __name__ == "__main__":
    main()
