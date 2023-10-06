from matplotlib import pyplot as plt
from modules.text_to_img_SD import make_txt_2_img_prediction
from modules.img_2_img_SD import make_img_2_img_prediction
from modules.in_painting_SD import make_inpainting_prediction
from utils.pipeline_loader import build_SD_pipeline_based_on_input, load_lora_weights_safetensor_to_diffusers
from utils.create_masks_from_prompts import create_border_mask, build_SAM, build_boundingbox_image_stack_from_masks, \
    create_all_mask_sam, clip_scoring, close_mask_holes
from PIL import Image
import cv2
import random
import torch


def main():
    # inputs ***************************
    seeds = None
    scheduler = "DDIM"
    reference_image_path = r"D:\Ecommerce_FakeModel\Reference_imgs\WaterGunRef.jpg"
    checkpoint_directory_SD = r"D:\Ecommerce_FakeModel\Models_Converted\Photon_inpaint"
    checkpoint_path_SAM = r'D:\Ecommerce_FakeModel\SAM_Checkpoint\sam_vit_h_4b8939.pth'
    direct = r'D:/Ecommerce_FakeModel/OutputPics_Issues/Tuning/'
    lora_path = '' #"D:\Ecommerce_FakeModel\Models_Converted\Lora\polyhedron_new_skin_v1.1.safetensors"
    lora_alpha = 1
    prompt = " (A model with sunglasses and a water gun), waterfall and river background"
    negative_prompt = ('cartoon, painting, illustration, (worst quality, low quality, normal quality:2), NSFW'
       )
    segmentation_prompt = 'a photo of water-gun, water gun '
    num_inference_steps_list = [50]  # The number of denoising steps. Higher number usually leads to higher quality
    cfg_list = [6] #6 is awesome
    height = 784
    width = 512
    border_mask_width = 8
    img2img_strength_first_pass = [0.8] # 0.9 on first. Heavy alteration should be given
    img2img_strength_second_pass = [0.5] #0.4 -0.5 best visual # lower to reduce effects of superimposition but also to limit border distortion
    HyperParameterTune_num = 1
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # ******************************

    # inpaint img2img

    run_inpainting = True
    if run_inpainting:
        pipeline = build_SD_pipeline_based_on_input(
            checkpoint_directory_SD, device, pipeline_type="Inpaint", scheduler=scheduler
        )

        if lora_path:
            pipeline = load_lora_weights_safetensor_to_diffusers(pipeline, lora_path, alpha = lora_alpha)

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
            cfg_choice = random.randrange(len(cfg_list))
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
                CFG=cfg_list[cfg_choice],
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
                CFG=cfg_list[cfg_choice],
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
