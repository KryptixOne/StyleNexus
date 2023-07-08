"""
Acquire Select Group of reference Images and turn them into tensors
"""
import tkinter as tk
from tkinter import filedialog
import torch
import torchvision.transforms as transforms
from PIL import Image

def acquire_image_paths():
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    file_paths = filedialog.askopenfilenames(
        title="Select Reference Images",
        filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.gif"), ("All files", "*.*"))
    )

    return list(file_paths)

def load_images_as_tensors(file_paths:list):
    """
    loads, scales to 1024x1024 and transforms input images to torch tensors
    :param file_paths: Image file Paths
    :return: Torch Tensors of images [image_num, channels, height, width]
    """
    images = []
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor()
    ])

    for file_path in file_paths:
        image = Image.open(file_path).convert("RGBA")
        image = image.resize((1024, 1024), resample=Image.BILINEAR)
        image_tensor = transform(image)
        images.append(image_tensor)

    return torch.stack(images)

if __name__ == '__main__':
    selected_images = acquire_image_paths()
    image_tensors = load_images_as_tensors(selected_images)
    print(image_tensors.shape)