"""
Project Should do the following:

1. Take the following as inputs from the User: [ ]
    a. sequence of Reference Images [ ]
    b. Desired Prompt [X]
    c. Desired Negative Prompt. Or This can be predetermined by us. [X]

2. Identify area of interest based on reference images. [ ]
    --> Create masks with segmentation net. [ ]
    --> Define Inpainting area [ ]
    --> Define Border of Inpainting area for Second pass to smooth border artifacts [ ]
    --> If using DreamBooth-like methodology, Not required

3. Incorporate Visual Similarity Metric.  [ ]
    --> Metric will rate how "similar" the generated object is to reference.  [ ]
    --> can use segmentation net to identify AoI [ ]
    --> Need to determine this metric still. As variation, pose, angle, lighting, etc. should not negatively affect
    the Metric but visual distortions to the reference should [ ]

4. Incorporate Image Filtering Based on Similarity Metric [ ]
    --> Remove Images that don't achieve a certain threshold. [ ]
    --> For Successful Images, Log Hyperparameters, seed, and reference Images (useful for future training) [ ]
    --> Return X number of Generated Images [ ]

5. Build a WebUI that Allows for independent user usage [ ]
"""
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
