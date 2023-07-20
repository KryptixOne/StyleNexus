from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import clip
from PIL import Image

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
def create_all_mask_sam(img):
    checkpoin_path = r'D:\Ecommerce_FakeModel\SAM_Checkpoint\sam_vit_h_4b8939.pth'
    sam = sam_model_registry["vit_h"](checkpoint=checkpoin_path)
    sam.to('cuda')
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(img)
    return masks


def clip_scoring(image_bbox:list, prompt:str):
    """
    Determines CLIP zero-shot scoring of input images and prompt
    :param image:
    :param prompt:
    :return:
    """

    device = "cuda"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_pre = [preprocess(x).unsqueeze(0).to(device) for x in image_bbox]
    stacked_images = torch.stack(image_pre).squeeze(1)
    text = clip.tokenize([prompt]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(stacked_images)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)

        #logits_per_image, logits_per_text = model(stacked_images, text)
        #probs = logits_per_text.softmax(dim=-1).cpu().numpy()

    print("Label probs:", similarity)
    return similarity
def convert_back_to_pil(img_tensor):
    import numpy as np
    from PIL import Image

    # Step 1: Convert to a NumPy array
    T_np = img_tensor.squeeze().cpu().numpy() if isinstance(img_tensor, torch.Tensor) else img_tensor.squeeze()

    # Step 2: Reshape to [3, 256, 256]
    T_np = T_np.transpose(1, 2, 0)

    # Step 3: Ensure values are in the [0, 255] range
    T_np = np.clip(T_np, 0, 1) * 255.0
    T_np = T_np.astype(np.uint8)

    # Step 4: Convert to a PIL image
    pil_image = Image.fromarray(T_np)

    return pil_image

def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    return segmented_image

def build_boundingbox_image_stack_from_masks(masks:list, original_img):
    img_bboxes= []
    for mask in masks:
        bbox = [mask["bbox"][0], mask["bbox"][1], mask["bbox"][0] + mask["bbox"][2], mask["bbox"][1] + mask["bbox"][3]]
        img_bboxes.append(segment_image(original_img,mask['segmentation']).crop(bbox))


    return img_bboxes


if __name__ == "__main__":
    img_path = r'D:\Ecommerce_FakeModel\DreamboothTut\Doggy\alvan-nee-9M0tSjb-cpA-unsplash.jpeg'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_image_pil = Image.fromarray(img)

    prompt = 'a photo of a dog'
    masks = create_all_mask_sam(img)

    print()
    img_bboxes = build_boundingbox_image_stack_from_masks(masks,orig_image_pil)
    probabilities = clip_scoring(img_bboxes, prompt)

    print()

