# Visual Product Matcher

## Overview

Visual Product Matcher is an AI-powered application that helps users find visually similar products using advanced computer vision technology. The application uses deep learning models to analyze product images and find similar items in the database based on visual features.

## Features

- Advanced image similarity search using ResNet50 model
- Visual similarity scoring with percentage matching
- Multiple product categories support
- Modern, responsive UI built with Streamlit
- Easy deployment on Streamlit Cloud

## How It Works

1. Upload an image or provide an image URL
2. The AI analyzes the visual features using a pre-trained ResNet50 model
3. The FAISS search engine finds the most similar products in the database
4. Results are displayed with similarity scores and product information

## Technical Details

- **Model**: ResNet50 (pre-trained on ImageNet)
- **Search Engine**: FAISS (Facebook AI Similarity Search)
- **Framework**: Streamlit + PyTorch
- **Features**: Image embedding, similarity search, visual interface

## Requirements

- Python 3.8+
- Streamlit
- PyTorch
- FAISS
- Pillow
- NumPy
- Joblib

## Deployment

This application is designed for deployment on Streamlit Cloud. The required files and configurations are included in this repository to enable easy deployment.

## Usage

1. Upload an image of a product you're interested in
2. Click "Find Similar Products" to search for matches
3. Browse the results with similarity scores
4. Explore different product categories using the sidebar filters

## How to Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/visual-product-matcher.git
   cd visual-product-matcher
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Deployed Application

The application is deployed on Streamlit Cloud and can be accessed at: https://vispmat.streamlit.app/

## Approach

This visual product matcher uses a pre-trained ResNet50 model to extract image features and FAISS for efficient similarity search. Images are converted to feature vectors, stored in an index, and searched using cosine similarity. The Streamlit interface allows users to upload images and find visually similar products from the database. Key challenges included handling image paths across platforms and optimizing for cloud deployment, solved by normalizing file paths and properly loading images before display.
