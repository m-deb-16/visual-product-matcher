import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
from supabase import create_client, Client
from transformers import ViTModel, ViTImageProcessor
import ast

# -----------------------------
# IMAGE PREPROCESSING PIPELINE (for ViT)
# -----------------------------
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

def transform_image(image: Image.Image):
    """Transform image using ViT processor"""
    inputs = processor(images=image, return_tensors="pt")
    return inputs['pixel_values']

# -----------------------------
# FEATURE EXTRACTOR (Vision Transformer - ViT)
# -----------------------------
class ViTFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.model.eval()
    
    @torch.no_grad()
    def forward(self, x):
        outputs = self.model(pixel_values=x)
        features = outputs.last_hidden_state[:, 0, :]  # CLS token
        return features.squeeze(0)

@torch.no_grad()
def extract_features(image: Image.Image, model: ViTFeatureExtractor):
    """Extract normalized features from a PIL image"""
    img_t = transform_image(image)
    feat = model(img_t).numpy()
    feat = feat / np.linalg.norm(feat)
    return feat.astype('float32')

# -----------------------------
# SUPABASE CLIENT
# -----------------------------
def get_supabase_client() -> Client:
    url = st.secrets['SUPABASE_URL']
    key = st.secrets['SUPABASE_KEY']
    return create_client(url, key)

# -----------------------------
# DATABASE FUNCTIONS
# -----------------------------
def search_similar(image: Image.Image, top_k=6, min_similarity=0.0):
    """Search for similar products using local vector computation"""
    model = ViTFeatureExtractor()
    supabase = get_supabase_client()
    
    # Extract query features
    query_feat = extract_features(image, model)

    # Fetch all products with embeddings
    response = supabase.table('products').select('id,name,category,image_url,features').execute()
    products = response.data

    results = []
    for p in products:
        try:
            # Convert features string to numpy array safely
            feat_list = ast.literal_eval(p['features'])
            feat = np.array(feat_list, dtype=np.float32)
        except Exception as e:
            print(f"Skipping product {p.get('id')} due to error:", e)
            continue

        # Compute cosine similarity
        if np.linalg.norm(feat) == 0 or np.linalg.norm(query_feat) == 0:
            sim = 0
        else:
            sim = float(np.dot(query_feat, feat) / (np.linalg.norm(query_feat) * np.linalg.norm(feat)))

        # Filter by similarity only (no category filter)
        if sim >= min_similarity:
            results.append({
                'id': p['id'],
                'name': p['name'],
                'category': p['category'],
                'image_url': p['image_url'],
                'similarity': sim
            })
    
    # Sort by similarity
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]

def get_all_categories():
    """Return all unique categories"""
    supabase = get_supabase_client()
    response = supabase.table('products').select('category').execute()
    categories = sorted(list({item['category'] for item in response.data if item.get('category')}))
    return categories

def get_product_stats():
    """Return total products and total categories"""
    supabase = get_supabase_client()
    response = supabase.table('products').select('id,category').execute()
    products = response.data
    total_products = len(products)
    total_categories = len(set([p['category'] for p in products if p.get('category')]))
    return {
        'total_products': total_products,
        'total_categories': total_categories
    }

def get_product_by_id(product_id):
    """Return product by ID"""
    supabase = get_supabase_client()
    response = supabase.table('products').select('*').eq('id', product_id).single().execute()
    if response.data:
        p = response.data
        return {
            'id': p['id'],
            'name': p['name'],
            'category': p['category'],
            'image_url': p['image_url']
        }
    return None

def add_product(name, category, image_url, features):
    """Add a new product to Supabase"""
    supabase = get_supabase_client()
    if isinstance(features, np.ndarray):
        features = features.tolist()
    response = supabase.table('products').insert({
        'name': name,
        'category': category,
        'image_url': image_url,
        'features': str(features)
    }).execute()
    if response.error:
        print("Error inserting product:", response.error)
    return response.data
