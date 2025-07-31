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

<img width="1402" height="1196" alt="f7a6a52239e3084c02e4ab0011d08f3a" src="https://github.com/user-attachments/assets/68304a86-3df6-4456-92ef-d5b26271a6fe" />

### 6. Select style image
You can preview each style image and select one.

<img width="1658" height="1095" alt="44f4a6755258c0f2d6dda6055acdd03e" src="https://github.com/user-attachments/assets/c0a1752d-3ad5-407b-b91b-558c39db5bc7" />

### 7. Wait for it to be generated and downloaded
When you click "Start conversion", the process starts. Wait a few minutes for the conversion to complete and allow the user to download.

<img width="1267" height="1156" alt="36d3ea8df5ed8c7da12c6fa9f5d856a9" src="https://github.com/user-attachments/assets/476f4981-d815-4995-82ed-a1e9fe5b70c7" />
