# ECommerce_Model


Project Should do the following:

## 1. Take the following as inputs from the User: [X]

    a. sequence of Reference Images or a Single Image [X]
        --> Function for Acquisition and Transformation developed [X]
        --> Integrate into SD pipeline [X] 

    b. Desired Prompt [X]
        --> Integrated into SD Pipeline [X]

    c. Desired Negative Prompt. Or This can be predetermined by us. [X]
        --> Integrated into SD Pipeline [X]

## 2. Build Base In-Painting Img2Img and DreamBooth Pipeline For SD [X]
    --> Img2Img Integration [X]
    --> DreamBooth Integration [ ] *Not Required
        -> Temporary Hold on Dreambooth Integration. Training is too resource intensive and it takes too long to train
    --> In-Painting Integration (Don't worry about mask creation yet) [X]

## 3. Build workflow that ultilizes each pipeline and produces output images given an input [X] 
    Note that the goal here blends with goal number 4
    --> Inpainting Method Workflow Creation [X]
        a. Acquire Reference Image of object [X] 
        b. Create Mask for Img2Img Inpainting [X]
            --> Implement Segment Anything Model (Note that text prompt input not public [X]
            --> Implement CLIP for prompt-to-mask similarity scoring to enable text prompt [X]
            --> Fill in holes in mask detection [X]
                a. Small hole fill in using morphology [X]
                b. Contour-level fill [X]
            --> Identify first 5 most relevant masks [X]
        c. Generate Img [X]
            --> Generated images affecting non-masked areas.
            --> need to determine whats up
        d. Build secondary Mask for Generated Image (on borders) [X] *Not Required
        e. Generate Img again using previous generated img and new mask. [X] *Not Required
        f. Save image. [X]

    --> Nice-To-Haves:
        a. LoRA integration (from .safetensors if using publicly available LoRAs) [X]
        b. Color Corrections [ ] 
        c. First Pass Auto-Beautify Filter (May include auto correction) [ ]
        d. Multiple-Beauty Filters on first pass (allow selection prior to second pass) [ ]
        e. When image is just the Shirt without Human structural Information. Superimpose on random model/manikin [ ] 
    -->

    
<img src="https://github.com/KryptixOne/ECommerce_Model/blob/DreamBooth_inpaint_Etc_Integration/OutputPics_Issues/GirlWearingLion.PNG" alt="Original Photo" width="30%"> <img src="https://github.com/KryptixOne/ECommerce_Model/blob/DreamBooth_inpaint_Etc_Integration/OutputPics_Issues/outputnew.png" alt="Inpainted Photo" width="30%">

See left: Original Image, Right: Inpainted Image. Notice the alterations occurs during inpainting

### Update on Issue:
Problem showing improvements after updating model weights.

Merging Inpainting Model V1.5 with Model not used for in-painting updates successfully according to the following:
A +(B-C) 
Where A is Inpaint Model, B is our Model, and C is the non-inpaint version of A

Note we also choose a merge multiplier of 1.

See results below

<img src="https://github.com/KryptixOne/ECommerce_Model/blob/DreamBooth_inpaint_Etc_Integration/OutputPics_Issues/ErroneousMale%20Model.png" alt="Original_superimposed Photo" width="30%"> <img src="https://github.com/KryptixOne/ECommerce_Model/blob/DreamBooth_inpaint_Etc_Integration/OutputPics_Issues/With_New_inpaintModel.png" alt="Inpainted_model Photo" width="30%">




## 4. Identify area of interest based on reference images. [X]

    --> Create masks with segmentation net. [X]
        a. "Segment Anything" Paper: https://arxiv.org/pdf/2304.02643.pdf
    --> Define Inpainting area [X]
    --> Define Border of Inpainting area for Second pass to smooth border artifacts [X]
        * Not required if Segmentation Mask is of high Quality
    --> If using DreamBooth-like methodology, Not required

## 5. Hyperparameter Tuning [ ]

## 6. Incorporate Visual Similarity Metric.  [ ]

    --> Metric will rate how "similar" the generated object is to reference.  [ ]
    --> can use segmentation net to identify AoI [ ]
    --> Need to determine this metric still. As variation, pose, angle, lighting, etc. should not negatively affect
    the Metric but visual distortions to the reference should [ ]

## 7. Incorporate Image Filtering Based on Similarity Metric [ ]

    --> Remove Images that don't achieve a certain threshold. [ ]
    --> For Successful Images, Log Hyperparameters, seed, and reference Images (useful for future training) [ ]
    --> Return X number of Generated Images [ ]

## 8. Build a WebUI that Allows for independent user usage [ ]

    --> Try different frameworks [X]
        a. Streamlit [X]
        b. Gradio [X]
        c. Flask [X]
    --> Determine which framework best suits your needs [X]
    --> Build sections using desired framework [ ]
        a. Homepage [X]
        b. About Us [ ]
        c. Examples [ ]
        d. Demo [ ]
        e. Contact [ ]
