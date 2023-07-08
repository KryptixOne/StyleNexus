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
    --> In-Painting Integration (Don't worry about mask creation yet) [X]

## 3. Build workflow that ultilizes each pipeline and produces output images given an input [ ] 
    Note that the goal here blends with goal number 4

## 4. Identify area of interest based on reference images. [ ]

    --> Create masks with segmentation net. [ ]
    --> Define Inpainting area [ ]
    --> Define Border of Inpainting area for Second pass to smooth border artifacts [ ]
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
