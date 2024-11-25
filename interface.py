import gradio as gr
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image

from models.classification_models.ResNet import *
from models.segmentation_models.ResnetUnet import *

def get_transforms(img_size=256):
    val_transform = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return val_transform

def load_models():
    classification_model = resnet_model
    classification_model.load_state_dict(torch.load('weights/classification_models/resnet50.pt'))
    classification_model.eval()

    segmentation_model = ResNetUnet()
    checkpoint = torch.load('weights\segmentation_models\ResNetUnet_best.pt')
    segmentation_model.load_state_dict(checkpoint['model_state_dict'])
    segmentation_model.eval()

    return classification_model, segmentation_model

def pipeline(input_image):
    classification_model, segmentation_model = load_models()
    transform = get_transforms()
    
    if isinstance(input_image, str):
        image = cv2.imread(input_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = input_image
 
    transformed = transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0)
    
    with torch.inference_mode():
        outputs = classification_model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    class_names = ['COVID', 'Non-COVID', 'Healthy']
    prediction = class_names[pred_class]
    confidence = probs[0][pred_class].item() * 100
    
    if prediction == 'COVID':
        with torch.inference_mode():
            output = segmentation_model(input_tensor)
            output = torch.sigmoid(output)  
            output = output.squeeze().cpu().numpy()  
            binary_mask = (output > 0.5).astype(np.uint8) * 255
            mask_resized = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))
            
            overlay = np.zeros_like(image)
            overlay[mask_resized > 0] = [255, 0, 0] 
            alpha = 0.3 
            blended = cv2.addWeighted(image, 1, overlay, alpha, 0)
            
            return [
                f"The patient is diagnosed with {prediction}",
                f"Confidence: {confidence:.2f}%",
                blended,
                None 
            ]
    elif prediction == 'Non-COVID':
        return [
            prediction,
            f"Confidence: {confidence:.2f}%",
            None,
            "The patient is not diagnosed with COVID-19, but with other lung diseases :("
        ]
    return [
        prediction,
        f"Confidence: {confidence:.2f}%",
        None,
        "The patient is healthy with no infected area :)"
    ]

def create_interface():
    with gr.Blocks(title="COVID-19 Lung Analysis", theme='lone17/kotaemon') as interface:
        with gr.Column(variant="panel"):
            gr.Markdown("# Lungs Radiography Analysis")
            gr.Markdown("""
                Upload/ Drop a chest X-ray image for COVID-19 diagnosis and analysis. 
            """)
        
        with gr.Row(equal_height=True):
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Chest X-ray",
                    height=400
                )
                submit_btn = gr.Button("Analyze Image", variant="primary", scale=1)
            
            with gr.Column(scale=1):                                
                output_image = gr.Image(
                    label="COVID-19 Analysis",
                    visible=False,
                    height=400
                )
                diagnosis_text = gr.Textbox(
                    label="Diagnosis Details",
                    visible=False,
                    container=False
                )
                with gr.Row(equal_height=True):
                    diagnosis_label = gr.Label(label="Diagnosis Conclusion")
                    confidence_label = gr.Label(label="Confidence Score")

        with gr.Accordion("Information", open=False):
            gr.Markdown("""
                ### How to Use
                1. Click the upload button/ Drag and drop a chest X-ray image.
                2. Choose 'Analyze Image'.
                3. Review the results:
                   - For COVID cases: View highlighted infection regions.
                   - For Non-COVID/Healthy cases: Review detailed diagnosis text.
            """)

        def handle_prediction(image):
            prediction, confidence, output_img, output_text = pipeline(image)
            is_covid = output_img is not None
            
            return (
                prediction, 
                confidence,  
                gr.update(value=output_img, visible=is_covid),  
                gr.update(value=output_text, visible=not is_covid)  
            )

        submit_btn.click(
            fn=handle_prediction,
            inputs=[input_image],
            outputs=[
                diagnosis_label,    
                confidence_label,   
                output_image,      
                diagnosis_text   
            ]
        )
        
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)