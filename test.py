import gradio as gr
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch
import numpy as np
from PIL import Image

from models.segmentation_models.ResnetUnet import ResNetUnetmodel_50

device = 'gpu' if torch.cuda.is_available() else 'cpu'

model = ResNetUnetmodel_50
checkpoint = torch.load('weights\segmentation_models\ResNetUnet_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
val_loss = checkpoint['val_loss']


def infer_mask(model, image, transform, device="cuda"):
    """
    Perform inference on a new image to generate a segmentation mask.
    
    Args:
        model (torch.nn.Module): The trained ResNetUnet model.
        image (PIL.Image.Image): The input image.
        transform (albumentations.Compose): Transform to preprocess the image.
        device (str): The device to run the model on ('cuda' or 'cpu').
    
    Returns:
        PIL.Image.Image: The predicted segmentation mask.
    """
    # Ensure model is in evaluation mode
    model.eval()
    model.to(device)
    
    # Preprocess the image
    img_array = np.array(image)
    transformed = transform(image=img_array)
    input_tensor = transformed['image'].unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Postprocess the output
    output = torch.sigmoid(output)  # Apply sigmoid to get probabilities
    output = output.squeeze().cpu().numpy()  # Remove batch and channel dimensions
    
    # Threshold the output to get the binary mask
    binary_mask = (output > 0.5).astype(np.uint8) * 255  # Convert to binary and scale to 0-255
    
    # Convert to PIL image
    mask_image = Image.fromarray(binary_mask)
    return mask_image

img_size = 256
segmentation_transform = A.Compose([
    A.LongestMaxSize(max_size=img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

image = Image.open("images\COVID\covid_1579.png").convert("RGB")  

mask = infer_mask(model, image, segmentation_transform, device="cuda")
mask.show() 