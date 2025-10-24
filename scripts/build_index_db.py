import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from io import BytesIO
import requests
from supabase import create_client, Client
from transformers import ViTModel, ViTImageProcessor

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY").strip()
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "products.csv")

# -----------------------------
# IMAGE PREPROCESSING PIPELINE
# -----------------------------
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

def transform_image(image: Image.Image):
    """Transform image using ViT processor"""
    inputs = processor(images=image, return_tensors="pt")
    return inputs['pixel_values']

# -----------------------------
# FEATURE EXTRACTOR
# -----------------------------
class ViTFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.model.eval()
        
    @torch.no_grad()
    def forward(self, x):
        outputs = self.model(pixel_values=x)
        # CLS token as feature vector
        features = outputs.last_hidden_state[:, 0, :]
        return features.squeeze(0)

def extract_features_from_image(img: Image.Image, model: ViTFeatureExtractor):
    """Extract normalized features from a PIL Image"""
    try:
        img_t = transform_image(img)
        feat = model(img_t).numpy()
        feat = feat / np.linalg.norm(feat)  # normalize
        return feat.astype('float32')
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

# -----------------------------
# BUILD INDEX
# -----------------------------
def build_index():
    print("🔍 Loading ViT model...")
    model = ViTFeatureExtractor()
    
    # Verify table exists
    try:
        response = supabase.table("products").select("id").limit(1).execute()
        if response.data is not None:
            print("✅ Products table exists in Supabase!")
        else:
            print("✅ Products table exists but is empty.")
    except Exception as e:
        print(f"❌ Products table does not exist. Error: {e}")
        print("   Please create it using supabase_table_schema.sql in Supabase SQL Editor")
        return
    
    # Read CSV
    df = pd.read_csv(CSV_PATH)
    processed_count = 0
    
    for _, row in df.iterrows():
        pid = row['id']
        name = row['name']
        category = row['category']
        image_url = row['image_url']
        
        # Download image
        try:
            r = requests.get(image_url, timeout=30)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            feat = extract_features_from_image(img, model)
        except Exception as e:
            print(f"⚠️ Error downloading/processing image {pid}: {e}")
            continue

        if feat is not None:
            # Upsert into Supabase
            data = {
                "id": int(pid),
                "name": name,
                "category": category,
                "image_url": image_url,
                "features": feat.tolist()
            }
            # Upsert without .error check (Supabase v1+ handles exceptions)
            supabase.table("products").upsert(data, on_conflict="id").execute()
            
            processed_count += 1
            print(f"✅ Inserted/Updated: {pid} - {name}")
    
    print(f"✅ Index built with {processed_count} items in Supabase!")

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    build_index()
    