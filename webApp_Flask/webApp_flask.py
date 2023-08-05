from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)

# Configuring static file route
app.static_folder = 'static'
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
def examples():
    return render_template('demo.html')

def generate_image(desired_prompt, segment_prompt, input_image):
    # Convert the input image to RGB format
    input_image = Image.open(input_image).convert("RGB")

    # Replace this with your actual image generation logic
    # The 'input_image' here is a PIL Image object
    # You can process the image as needed and return the URL of the generated image.
    generated_image_url = "placeholder_image.jpg"
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




if __name__ == '__main__':
    # to run:  python ./webApp_Flask/webApp_flask.py in cmd prompt
    app.run()
