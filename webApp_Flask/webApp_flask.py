from flask import Flask, render_template, request, jsonify

from modules.api_calls import inpainting_api

app = Flask(__name__)

# Configuring static file route
app.static_folder = 'static'


def generate_image(desired_prompt, segment_prompt, input_image):
    checkpoint_directory_SD = None
    checkpoint_path_SAM = None
    lora_path = None
    border_mask_width = 8  # make it based off of image size
    negative_prompt = ('(worst quality, low quality, normal quality:2), bad quality, bad hands, ugly, NSFW'
                       )
    height = None
    width = None

    generated_img = inpainting_api(prompt=desired_prompt,
                                   negative_prompt=negative_prompt,
                                   segmentation_prompt=segment_prompt,
                                   device='cuda',
                                   seeds=None,
                                   height=height, width=width,
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
    generated_image_url = generated_img #"placeholder_image.jpg"
    return generated_image_url


@app.route("/generate-image", methods=["POST"])
def generate_image_api():
    desired_prompt = request.form.get("desiredPrompt")
    segment_prompt = request.form.get("segmentPrompt")
    input_image = request.files.get("inputImage")

    if desired_prompt is None or segment_prompt is None or input_image is None:
        return jsonify({"error": "Missing required parameters"}), 400

    image_url = generate_image(desired_prompt, segment_prompt, input_image)
    return jsonify({"image_url": image_url})


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
