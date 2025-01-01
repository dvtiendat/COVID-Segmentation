# Lungs X-Ray Image Analysis

## Features:
The project will allow you to classify lung radiography images into 3 types

- Normal (Lung is healthy)
- COVID (The person has covid)
- Non-COVID (The person doesn't have COVID but suffers different lung-related issues)
  
If a patient is diagnosed with COVID, the pipeline will segment the COVID-diagnosed radiography image further for better prescriptions and treatments

## Usage: 
`git clone https://github.com/dvtiendat/Lungs-Radiography-Analysis.git`

`pip install -r requirements.txt`

Please kindly go to this link to download the pretrained weights and put them at their respective folders of classification and segmentation: https://drive.google.com/drive/folders/1IXO5PCt506PHAEDNOhU4TYRPuE4VWk3K?usp=sharing

To run the application:
`python3 app.py`
