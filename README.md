These tools were created to as part of a project for a graduate class on computer vision. The overall goal of the project was to train a convolutional neural network classifier to recognize/categorize medications. This part of the project identified a subset of images to be used for training These images and more information about them can be found at: https://data.lhncbc.nlm.nih.gov/public/Pills/index.html 

Getting started:
For a demo of the image processing and run:
    python demo.py

This work focuses on processing three types of images for use in a CNN model/workflow. Please refer to the image metadata to learn more about these types. A brief description of the images follows:

    - MC_C3PI_REFERENCE_SEG_V1.6 : These are 16bit RGBA images on a grey background. 41330 total images available. 
    
    - MC_SPL_SPLIMAGE_V3.0 : These are 8bit RGB images with both the front and back of the image measurement shown on a grey background with scales along the lefthand and lower edges. 
    
    - C3PI_TEST : Also 8bit RGB images. These are also referred to as 'consumer images', presumably, in the sense that as opposed to the above two classes, these images are not taken under controlled conditions, i.e. they have more natural variation. Presented here is an example of using a support vector machine classifier in combinaiton with sklearn.feature.hog() to extract images, although training an FCN blob detector is perhaps a better approach.

Features:
The available features depend on the type of image.

    - Images of type MC_C3PI_REFERENCE_SEG_V1.6 can be 'cleaned' in the sense that the banner is removed and the image is resized (while respecting aspect ratio) to the given output requirement. Optionally, these can also be augmented. See the section on augmentation. 
    
    - Images of type MC_SPL_SPLIMAGE_V3.0 images can be processed into separate images for the front and back of the pill. These could likely be augmented as well although that functionality is not currently implemented. 
    
    - Extraction of consumer images using a support vector machine classifier in conjunction with a historgram of oriented gradients. This a computationally intensive approach, which can be prone to false positives. As noted above, training an FCN blob detector or alternatively, using a CUDA accelerated HOG() might be more a scalable alternative.  

Augmentation:
This is currently only available for images of type MC_C3PI_REFERENCE_SEG_V1.6. The isolated image can be:

    - randomly rotated from 1-359 degrees
    
    - randomly resized from 0.5x - 1x of the model input requirement
    
    - randomly inserted (i.e. the pill is not centered in the image) onto a random background 
    
    - optionally return a boolean mask for semantic segmentation

Image Metadata:
Image metadata is available in several formats:

    1. XML files at https://data.lhncbc.nlm.nih.gov/public/Pills/ALLXML/index.html
    
    2. A bar delimited text file at https://data.lhncbc.nlm.nih.gov/public/Pills/directory_consumer_grade_images.txt (Please Note: Although the filename seems to indicate that the metadata is specific to consumer images, it actually contains information on all the classes.)
    
    3. An optional tabular format available containing more extensive metadata than option (2) and very closely - although not exactly - mirroring the metadata contained  in the XML. 

    Note: The information in the metadata differs somewhat. The XML (and therefore the tabular formatted option) contain much more information than the bar delimited file. The metadata available in the XML or tabular formats that is NOT available in the bar delimited data include (this is not meant to be an exhaustive list):
    
        - physical characteristics (size, shape, color, presence of a 'score'), 
        - the type of imprint and the text of the imprint (i.e. the text printed or stamped onto the pill), 
        - generic / proprietary medication name,
        - in some cases camera metadata, and information about the background upon which the pill appears.



Alternatively, see the Gist for each type of image processing at:

    - Gist1
    - Gist2
    - Gist3

Please note some of the processing functionalities requires downloading metadata from the NIH (see read_nih_data() in data_gen.py) or background images for the augmentation demo. 

To create images for training, I recommend using a concatenation of NDC11 and Part. A single NDC11 may have multiple distinct pills, indeed, even a concatenation of NDC11 and Part still leads to a few images that while very similar visually, have a distinct imprint. For example, NDC11 = 00093316753, Part = 1 has images [CZT0ZZN08CQWE6UA6EUFM43SPG6UCLI.JPG] (https://data.lhncbc.nlm.nih.gov/public/Pills/PillProjectDisc84/images/CZT0ZZN08CQWE6UA6EUFM43SPG6UCLI.JPG), and [CSWXMSOIU836B3RJM3M-IZO_BFJBQ2G.JPG] (https://data.lhncbc.nlm.nih.gov/public/Pills/PillProjectDisc77/images/CSWXMSOIU836B3RJM3M-IZO_BFJBQ2G.JPG). It is left to the end user to decide whether to handle these as one class or not. 
