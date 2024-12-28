import gradio as gr
import torch.nn.functional as F
import albumentations as A
from pipeline import *

def get_css(css_path):
    with open(css_path, 'r') as f:
        custom = f.read()
    
    return custom

def create_interface():
    custom = get_css('design/design.css')
    processor = Pipeline()

    with gr.Blocks(css=custom, theme=gr.themes.Soft(primary_hue='teal', secondary_hue='blue')) as interface:
        with gr.Column(variant="compact"):
            gr.Markdown("# Lungs Radiography Analysis", elem_classes='heading')
            gr.Markdown("""
                Upload/ Drop a chest X-ray image for COVID-19 diagnosis and analysis. 
            """)
        
        with gr.Row(equal_height=True):
            # [MODEL SELECTION]
            with gr.Column(scale=0.2):
                classification_dropdown = gr.Dropdown(
                    choices=['ResNet50', 'VGG16'],
                    value='ResNet50',
                    label='Classification Model',
                )
                segmentation_dropdown = gr.Dropdown(
                    choices=['ResNetUnet', 'AttentionUnet'],
                    value='ResNetUnet',
                    label='Segmentation Model'
                )

            # [UPLOAD IMAGE SECTION]
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Chest X-ray",
                    height=400,
                    elem_classes="upload-image"
                )

                # [BUTTON]
                with gr.Row():
                    submit_btn = gr.Button("Analyze Image", variant="primary", elem_classes='primary-button', scale=2)
                    clear_btn = gr.Button('Clear', variant='secondary', scale=1)
                    
            with gr.Column():
                with gr.Group(elem_classes='results-container'):                    
                    output_image = gr.Image(
                        label="COVID-19 Analysis",
                        visible=False,
                        height=400
                    )

                with gr.Row(equal_height=True):
                    diagnosis_label = gr.Label(label="Diagnosis Conclusion", elem_classes='results-container')
                    confidence_label = gr.Label(label="Confidence Score", elem_classes='results-container')
                
                with gr.Row():
                    diagnosis_text = gr.Textbox(
                                label="Diagnosis Details",
                                visible=False,
                                container=False
                            )
        
        # [HELP SECTION]    
        with gr.Accordion("Information", open=False):
                    gr.Markdown("""
                ### Tutorial
                1. Click the upload button/ Drag and drop a chest X-ray image.
                2. Choose 'Analyze Image'.
                3. Review the results:
                   - For COVID cases: View highlighted infection regions.
                   - For Non-COVID/Healthy cases: Review detailed diagnosis text.
            """)
                    
        def clear_inputs():
            return {
                input_image: None,
                output_image: gr.update(visible=False),
                diagnosis_text: gr.update(visible=False),
                diagnosis_label: None,
                confidence_label: None
            }
        
        def handle_prediction(image, classification_model, segmentation_model, opacity=0.5):
            processor._load_models(classification_model, segmentation_model)            
            prediction, confidence, output_img, analysis_text = processor.process_image(
                image, segmentation_model, overlay_opacity=opacity
            )
            
            confidence_class = (
                "confidence-high" if confidence > 90
                else "confidence-medium" if confidence > 70
                else "confidence-low"
            )
            
            is_covid = output_img is not None
            
            return {
                diagnosis_label: prediction,
                confidence_label: gr.update(
                    value=f"Confidence: {confidence:.2f}%",
                    elem_classes=[confidence_class]
                ),
                output_image: gr.update(value=output_img, visible=is_covid),
                diagnosis_text: gr.update(value=analysis_text, visible=True)
            }

        submit_btn.click(
            fn=handle_prediction,
            inputs=[input_image, classification_dropdown, segmentation_dropdown],
            outputs=[
                diagnosis_label,
                confidence_label,
                output_image,
                diagnosis_text,
            ]
        )
        
        clear_btn.click(
            fn=clear_inputs,
            inputs=[],
            outputs=[
                input_image,
                output_image,
                diagnosis_text,
                diagnosis_label,
                confidence_label
            ]
        )
        
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)