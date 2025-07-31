# VoidScape: AI-Powered Chinese Gongbi Painting Style Transfer Tool

VoidScape is an AI-driven style transfer tool that transforms images into the traditional Chinese Gongbi painting style. Built with PyTorch and Streamlit, the system uses a pre-trained VGG19 model to preserve fine content structures while applying delicate brushwork and harmonious color palettes. Users can upload images, select from curated Gongbi styles, and adjust brightness/saturation for personalized output. This project promotes Chinese artistic aesthetics through accessible digital interaction.

## Link to project video recording: https://mega.nz/file/2nICUSgK#KrqUBZBay9V4FryA5qIafYGrFNMqjTcfDa0fplz08Eg

# Documentation:
[content_images_example](content_images_example): Inside the folder is a collection of sample original images used to test the effects of image generation for my AI applications.

[demo_media](demo_media)：The purpose of this folder is to show the AI-generated effects of my project, which contains six original input images (stored in the [input](demo_media/input) folder) and their corresponding AI-generated effect images (stored in the [output](demo_media/output) folder). Each of the six effect images uses six different artisanal-style drawings as reference images.

[styled_image](styled_image): Six predefined Chinese Gongbi painting style images serve as reference images for AI-generated outputs.

[app_VoidScape.py](app_VoidScape.py): The core function of this file is to provide a Streamlit interactive interface and utilize the imported style transfer model to convert user-uploaded images into the Gongbi painting style.

[style_transfer_model.py](style_transfer_model.py): The core function of this file is to load the pre-trained VGG19 model, transform the input content image into a predefined Gongbi painting style, and incorporate a visual optimization process, ultimately achieving VGG19-based image style transfer.

# Setup instructions:

This document provides step-by-step instructions on how to set up the Conda environment, download necessary files, and run the project.

### 1. Clone the Repository
Clone the project repository to your computer.
```
git clone https://github.com/Mingzhao-Du/VoidScape-AI-Powered-Chinese-Gongbi-Painting-Style-Transfer-Tool
```
### 2. Activate the existing environment
```
conda activate aim
```
### 3. Install required dependencies
```
pip install torch
pip install torchvision
pip install streamlit
pip install Pillow
```
### 4. Run programme (Image generation)
Execute [app_VoidScape.py](app_VoidScape.py) in the terminal to launch the Streamlit interface. This allows users to upload images and generate AI-stylized Gongbi paintings.
```
streamlit run "D:\GitHub Desktop\KeJian\AI-4-Media-Project-Mingzhao-Du\app_VoidScape.py"
```
Once the application starts, open the following URL in your browser:
```
Local URL: http://localhost:8501
```

### 5. Upload image and adjust parameters
You can upload image by dragging them directly or by previewing the folder. You can choose to adjust the brightness and saturation of the image.

![156e0b8e52528aae09abb41421142e3](https://git.arts.ac.uk/24004238/AI-4-Media-Project-Mingzhao-Du/assets/1145/235028cc-20c4-41c6-b52e-60fddb0dfc47)

### 6. Select style image
You can preview each style image and select one.

![58a841ae1dd4783e434ceadb3117540](https://git.arts.ac.uk/24004238/AI-4-Media-Project-Mingzhao-Du/assets/1145/578c6f57-181f-40d2-890d-5ce370c89366)

### 7. Wait for it to be generated and downloaded
When you click "Start conversion", the process starts. Wait a few minutes for the conversion to complete and allow the user to download.

![ccbe90930503d12b6abcfedaa836ea6](https://git.arts.ac.uk/24004238/AI-4-Media-Project-Mingzhao-Du/assets/1145/270eaad4-7375-49a9-98c3-2d3ab1cb694e)
