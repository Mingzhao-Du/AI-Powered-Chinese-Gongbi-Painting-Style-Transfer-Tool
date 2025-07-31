import streamlit as st
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
import torch
import torch.optim as optim
from torchvision.models import vgg19, VGG19_Weights


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Predefined style image paths
STYLE_IMAGES = {
    "Gongbi 1": "D:/GitHub Desktop/KeJian/AI-4-Media-Project-Mingzhao-Du/styled_image/Gongbi Painting Style 1.jpg",
    "Gongbi 2": "D:/GitHub Desktop/KeJian/AI-4-Media-Project-Mingzhao-Du/styled_image/Gongbi Painting Style 2.jpg",
    "Gongbi 3": "D:/GitHub Desktop/KeJian/AI-4-Media-Project-Mingzhao-Du/styled_image/Gongbi Painting Style 3.jpg",
    "Gongbi 4": "D:/GitHub Desktop/KeJian/AI-4-Media-Project-Mingzhao-Du/styled_image/Gongbi Painting Style 4.jpg",
    "Gongbi 5": "D:/GitHub Desktop/KeJian/AI-4-Media-Project-Mingzhao-Du/styled_image/Gongbi Painting Style 5.jpg",
    "Gongbi 6": "D:/GitHub Desktop/KeJian/AI-4-Media-Project-Mingzhao-Du/styled_image/Gongbi Painting Style 6.jpg"
}

# Training parameters
ITERATIONS = 100
STYLE_WEIGHT = 1e5
CONTENT_WEIGHT = 1e1
TV_WEIGHT = 1e-3
MAX_SIZE = 720

# Style Transfer Class
class StyleTransfer:
    def __init__(self):
        # Load the full VGG19 model and manually select layers
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.layers = vgg.to(device)

        # Disable gradients
        for param in self.layers.parameters():
            param.requires_grad = False
        
        # Preload style features
        self.style_features_cache = {}
        self.load_style_features()

    def load_style_features(self):
        transform = transforms.Compose([
            transforms.Resize(MAX_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        for style_name, filename in STYLE_IMAGES.items():
            style_image = Image.open(filename).convert("RGB")
            style_tensor = transform(style_image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = self.get_features(style_tensor)
                gram_matrices = {key: self.gram_matrix(value) for key, value in features.items()}
            
            self.style_features_cache[style_name] = gram_matrices

    def get_features(self, image):
        """Extract features from specific layers of VGG19"""
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'content',
            '28': 'conv5_1'
        }
        features = {}
        x = image
        for name, layer in self.layers._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    @staticmethod
    def gram_matrix(tensor):
        """Compute the Gram matrix"""
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram

    def transfer(self, content_input, style_name, progress_callback=None):
        """Perform style transfer, supporting both Tensor and byte stream inputs"""
        if isinstance(content_input, torch.Tensor):  
            # If the input is a Tensor, use it directly
            content_tensor = content_input
        else:
            # If the input is a byte stream, convert it to a Tensor
            content_tensor, _ = self.load_image(content_input)

        generated_image = content_tensor.clone().requires_grad_(True)
        style_grams = self.style_features_cache[style_name]
        
        optimizer = optim.LBFGS([generated_image], lr=0.5, max_iter=20)
        content_features = self.get_features(content_tensor)

        # Placeholder for preview display
        preview_placeholder = st.empty()

        run = [0]

        def closure():
            optimizer.zero_grad()
            gen_features = self.get_features(generated_image)
            
            # Content loss
            content_loss = torch.mean((gen_features['content'] - content_features['content']) ** 2)
            
            # Style loss
            style_loss = sum(torch.mean((self.gram_matrix(gen_features[layer]) - style_grams[layer]) ** 2)
                            for layer in style_grams)

            # Total variation loss (smoothness)
            tv_loss = torch.sum(torch.abs(generated_image[:, :, :, :-1] - generated_image[:, :, :, 1:])) + \
                      torch.sum(torch.abs(generated_image[:, :, :-1, :] - generated_image[:, :, 1:, :]))

            total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss + TV_WEIGHT * tv_loss
            total_loss.backward()

            # Real-time preview
            if run[0] % 10 == 0:
                preview_image = self.denormalize(generated_image)
                preview_placeholder.image(preview_image, caption=f"Iteration {run[0]}", use_container_width=True)

            # Update progress
            if progress_callback and run[0] % 10 == 0:
                progress_callback((run[0] + 1) / ITERATIONS, total_loss.item())

            run[0] += 1
            return total_loss

        while run[0] < ITERATIONS:
            optimizer.step(closure)

        preview_placeholder.empty()
        return self.denormalize(generated_image)

    @staticmethod
    def load_image(image_bytes):
        # Load and normalize input image
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        original_size = image.size

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0).to(device), original_size

    @staticmethod
    def denormalize(tensor):
        # Denormalize and convert to a PIL image
        tensor = tensor.cpu().detach().squeeze()
        tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        tensor = torch.clamp(tensor, 0, 1)
        return transforms.ToPILImage()(tensor)
