
from matplotlib import pyplot as plt
from Modules.Text_To_Img_SD import build_SD_pipeline, make_img_prediction
def main():
    # inputs
    seeds = None
    scheduler = 'DDIM'
    checkpoint_directory = r'D:\Ecommerce_FakeModel\Mogit dels_Converted\Lyriel_Diffusers'
    device = 'cuda'
    prompt = 'A cute puppy'
    negative_prompt = 'lowres, bad anatomy, bad hands, text, error, ' \
                      'missing fingers, extra digit, fewer digits, cropped,' \
                      ' worst quality, low quality, normal quality, jpeg artifacts,' \
                      ' signature, watermark, username, blurry, artist name, young, loli'
    CFG = 7
    height = 512
    width = 512

    pipeline = build_SD_pipeline(checkpoint_directory, device, scheduler=scheduler)
    image_out = make_img_prediction(pipeline, prompt, negative_prompt, device=device, seeds=seeds, height=height,
                                    width=width, CFG=CFG)
    plt.imshow(image_out)
    plt.show()

if __name__ == '__main__':
    main()
