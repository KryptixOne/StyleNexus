# ECommerce_Model


Project Should do the following:

## 1. Take the following as inputs from the User: [ ]

    a. sequence of Reference Images [ ]
        --> Function for Acquisition and Transformation developed [X]
        --> Integrate into SD/Dreambooth pipeline [ ] 

    b. Desired Prompt [X]
        --> Integrated into SD Pipeline [X]

    c. Desired Negative Prompt. Or This can be predetermined by us. [X]
        --> Integrated into SD Pipeline [X]

## 2. Build Base In-Painting Img2Img and DreamBooth Pipeline For SD [ ]
    --> Img2Img Integration [X]
    --> DreamBooth Integration [ ] 
        -> Temporary Hold on Dreambooth Integration. Training is too resource intensive and it takes too long to train
    --> In-Painting Integration (Don't worry about mask creation yet) [X]

## 3. Build workflow that ultilizes each pipeline and produces output images given an input [ ] 
    Note that the goal here blends with goal number 4
    --> Inpainting Method Workflow Creation [ ]
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
        d. Build secondary Mask for Generated Image (on borders) [ ] *Not Required
        e. Generate Img again using previous generated img and new mask. [ ] *Not Required
        f. Save image. [ ]

<img src="https://github.com/KryptixOne/ECommerce_Model/blob/DreamBooth_inpaint_Etc_Integration/OutputPics_Issues/GirlWearingLion.PNG" alt="Original Photo" width="50%"> <img src="https://github.com/KryptixOne/ECommerce_Model/blob/DreamBooth_inpaint_Etc_Integration/OutputPics_Issues/outputnew.png" alt="Inpainted Photo" width="50%">

See left: Original Image, Right: Inpainted Image. Notice the alterations occurs during inpainting

## 4. Identify area of interest based on reference images. [ ]

    --> Create masks with segmentation net. [X]
        a. "Segment Anything" Paper: https://arxiv.org/pdf/2304.02643.pdf
    --> Define Inpainting area [X]
    --> Define Border of Inpainting area for Second pass to smooth border artifacts [ ]
        * Not required if Segmentation Mask is of high Quality
    --> If using DreamBooth-like methodology, Not required

## 5. Incorporate Visual Similarity Metric.  [ ]

    --> Metric will rate how "similar" the generated object is to reference.  [ ]
    --> can use segmentation net to identify AoI [ ]
    --> Need to determine this metric still. As variation, pose, angle, lighting, etc. should not negatively affect
    the Metric but visual distortions to the reference should [ ]

## 6. Incorporate Image Filtering Based on Similarity Metric [ ]

    --> Remove Images that don't achieve a certain threshold. [ ]
    --> For Successful Images, Log Hyperparameters, seed, and reference Images (useful for future training) [ ]
    --> Return X number of Generated Images [ ]

## 7. Build a WebUI that Allows for independent user usage [ ]
