from flask import Flask, render_template, request, jsonify
from PIL import Image
from modules.api_calls import inpainting_api
import base64
from io import BytesIO
from matplotlib import pyplot as plt
app = Flask(__name__)

# Configuring static file route
app.static_folder = 'static'


def generate_image(desired_prompt, segment_prompt, input_image):
    checkpoint_directory_SD = r'./checkpoints/Photon_inpaint'
    checkpoint_path_SAM = r'./checkpoints/SAM_Checkpoint/sam_vit_h_4b8939.pth'
    lora_path = None
    border_mask_width = 8  # make it based off of image size
    negative_prompt = ('(worst quality, low quality, normal quality:2), long neck bad quality, bad hands, ugly, NSFW'
                       )
    #desired_prompt = 'sexy, attractive, beautiful, '+desired_prompt
    # get init dimensions
    image = Image.open(input_image)

    max_dimension = 1024
    image_width, image_height = image.size

    # Calculate the new dimensions while maintaining the aspect ratio
    aspect_ratio = image_width / image_height
    new_width = min(image_width, max_dimension)
    new_height = min(image_height, max_dimension)
    if new_width / aspect_ratio > new_height:
        new_width = int(new_height * aspect_ratio)
    else:
        new_height = int(new_width / aspect_ratio)

    # Ensure new_width and new_height are divisible by 8
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8

    image = image.resize((new_width, new_height), Image.LANCZOS)

    # Extract the dimensions after resizing
    image_width, image_height = image.size
    print(image_height, image_width)

    # Now that we have extracted the dimensions, delete the 'image' variable to free up memory
    del image
    generated_img = inpainting_api(prompt=desired_prompt,
                                   negative_prompt=negative_prompt,
                                   segmentation_prompt=segment_prompt,
                                   device='cuda',
                                   seeds=None,
                                   height=image_height, width=image_width,
                                   cfg_val=6,
                                   num_inference_steps=50,
                                   reference_image_path=input_image,
                                   checkpoint_directory_SD=checkpoint_directory_SD,
                                   checkpoint_path_SAM=checkpoint_path_SAM,
                                   lora_path=lora_path,
                                   border_mask_width=border_mask_width,
                                   img2img_strength_first_pass_val=0.8,
                                   img2img_strength_second_pass_val=0.4)

    # You can process the image as needed and return the URL of the generated image.
    generated_image_url = generated_img[1]  #"placeholder_image.jpg"
    return generated_image_url


@app.route("/generate-image", methods=["POST"])
def generate_image_api():
    desired_prompt = request.form.get("desiredPrompt")
    segment_prompt = request.form.get("segmentPrompt")
    input_image = request.files.get("inputImage")

    if desired_prompt is None or segment_prompt is None or input_image is None:
        return jsonify({"error": "Missing required parameters"}), 400

    generated_image = generate_image(desired_prompt, segment_prompt, input_image)

    buffered = BytesIO()
    generated_image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Return the JSON response with the base64-encoded image
    return jsonify({"image_data": encoded_image})


@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/examples')
def examples():
    return render_template('examples.html')


@app.route('/demo')
def demo():
    return render_template('demo.html')


if __name__ == '__main__':
    # to run:  python ./webApp_Flask/webApp_flask.py in cmd prompt
    app.run()
