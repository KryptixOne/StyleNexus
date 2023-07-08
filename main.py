from matplotlib import pyplot as plt
from modules.text_to_img_SD import make_txt_2_img_prediction
from modules.img_2_img_SD import make_img_2_img_prediction
from utils.pipeline_loader import build_SD_pipeline_based_on_input


def main():
    # inputs
    seeds = None
    scheduler = "DDIM"
    checkpoint_directory = r"D:\Ecommerce_FakeModel\Models_Converted\Lyriel_Diffusers"
    device = "cuda"
    prompt = "A cute puppy"
    negative_prompt = (
        "lowres, bad anatomy, bad hands, text, error, "
        "missing fingers, extra digit, fewer digits, cropped,"
        " worst quality, low quality, normal quality, jpeg artifacts,"
        " signature, watermark, username, blurry, artist name, young, loli"
    )

    num_inference_steps = 50  # The number of denoising steps. Higher number usually leads to higher quality
    CFG = 7
    height = 512
    width = 512

    img2img_strength = 0.7
    reference_image_path = None
    """
    img2img_strength (float, optional, defaults to 0.8) — Conceptually, says how much to transform the reference img.
    Must be between 0 and 1. image will be used as a starting point, adding more noise to it the larger the strength.
    The number of denoising steps depends on the amount of noise initially added. When strength is 1, added noise will
    be maximum and the denoising process will run for the full number of iterations specified in num_inference_steps.
    A value of 1, therefore, essentially ignores image.
    """

    # For text to img
    pipeline = build_SD_pipeline_based_on_input(
        checkpoint_directory, device, pipeline_type="Text2Img", scheduler=scheduler
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

    plt.imshow(image_out)
    plt.show()

    # For Image to Image
    pipeline = build_SD_pipeline_based_on_input(
        checkpoint_directory, device, pipeline_type="Img2Img", scheduler=scheduler
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

    plt.imshow(image_out)
    plt.show()


if __name__ == "__main__":
    main()