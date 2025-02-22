# StyleNexus

**Diffusion-Based Model for E-Commerce Applications**

<p align="center">
  <img src="https://github.com/KryptixOne/ECommerce_Model/blob/main/webApp_Flask/static/Video_Gif.gif" alt="StyleNexus Demo">
</p>

## Overview
StyleNexus is a diffusion-based model designed for e-commerce applications. It enables high-quality image generation and transformation for various fashion and retail use cases.

## 🚀 Features
- **Diffusion Model Integration**: Leverages state-of-the-art diffusion models for image manipulation.
- **Customizable Inference**: Supports hyperparameter tuning to optimize image generation.
- **Web Interface**: Includes a Flask-based web application for easy interaction.
- **LoRA Support**: Allows for fine-tuning and integration of Low-Rank Adaptation (LoRA) models.
- **Docker Support (Work in Progress)**: Aims to provide seamless deployment using CUDA-enabled Docker images.

## 📌 Setup (Work in Progress)
Setup is currently under development, with a goal to integrate Docker using a CUDA base image. The requirements include PyTorch, but the base image may already provide necessary dependencies.

### **Checkpoint Structure**
Ensure the following directory structure for model checkpoints:
```plaintext
checkpoints/
├── Lora
├── Photon_inpaint
└── SAM_Checkpoint
```

---

## 🔍 Inference Usage
To run inference locally, follow these steps:

1. Open `main.py`.
2. Set the following parameters inside `def main()`:
```python
seeds = None
scheduler = "DDIM"
reference_image_path = "Set to input image path"
checkpoint_directory_SD = "Set to checkpoint directory for Diffusion model"
checkpoint_path_SAM = "Set to checkpoint directory for SAM model"
direct = "Set to directory for output image"
lora_path = None  # Set to LoRA file path if available, else keep None
lora_alpha = 0.5  # Value between 0-1 (0: No LoRA, 1: Full LoRA)
device = "cuda"
prompt = "A stylish model wearing sunglasses and a T-shirt, simple plain background"
negative_prompt = "cartoon, painting, illustration, (worst quality, low quality, normal quality:2), NSFW"
segmentation_prompt = "a photo of a T-shirt"
num_inference_steps_list = [50]  # Higher value improves quality
cfg_list = [6]  # 6 is optimal
height = 784  # Output image height
width = 512  # Output image width
border_mask_width = 8  # Border fix width
img2img_strength_first_pass = [0.9]  # High alteration in first pass
img2img_strength_second_pass = [0.5]  # Optimized balance for visual quality
HyperParameterTune_num = 1  # Set >1 for hyperparameter tuning
```
3. Run `main.py` after setting the values.

---

## 🎛 Hyperparameter Tuning
To optimize the inference process:

1. Set `HyperParameterTune_num` to a value `>1`.
2. Tune the following hyperparameters by providing a list of desired values:
```python
num_inference_steps_list = [list of values]
cfg_list = [list of values]
img2img_strength_first_pass = [list of values]
img2img_strength_second_pass = [list of values]
```

---

## 🌐 Flask Web App Usage
Run the Flask-based web application with the following steps:

1. Execute the following command in the terminal:
   ```sh
   python ./webApp_Flask/webApp_flask.py
   ```
2. Click on the link that appears in the terminal to access the web interface.

---

## 📌 Roadmap
- [ ] Complete Docker-based setup with CUDA.
- [ ] Improve UI/UX for Flask web application.
- [ ] Optimize model performance with better hyperparameter tuning.
- [ ] Support additional diffusion models.

## 📜 License
This project is licensed under the MIT License.

## 🙌 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## 📬 Contact
For any queries, reach out via GitHub issues or discussions.

---

_Enhance your e-commerce product visuals with StyleNexus! 🚀_

