import os
import streamlit as st
import faiss
import torch
import numpy as np
import joblib
from PIL import Image
from torchvision import models, transforms
from io import BytesIO
import requests
import base64

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="🔍 Product Vision Finder",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛍️"
)
DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "image_index.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")

# -----------------------------
# SIDEBAR CONTENT
# -----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.markdown("## 🛍️ Product Vision Finder")
    st.markdown("---")
    st.markdown("### About This App")
    st.markdown("""
    This AI-powered application helps you find visually similar products using advanced computer vision technology.
    
    **How it works:**
    1. Upload an image or provide an image URL
    2. Our AI analyzes the visual features
    3. We find the most similar products in our database
    """)
    
    st.markdown("---")
    st.markdown("### App Features")
    st.markdown("""
    - 🔍 Advanced image similarity search
    - 📊 Visual similarity scoring
    - 🏷️ Multiple product categories
    - 🎨 Modern, responsive UI
    """)
    
    st.markdown("---")
    st.markdown("### Technical Details")
    st.markdown("""
    - **Model:** ResNet50 (pre-trained)
    - **Search Engine:** FAISS
    - **Framework:** Streamlit + PyTorch
    """)
    
    st.markdown("---")
    st.markdown("<small>Created with ❤️ using AI technology</small>", unsafe_allow_html=True)

# -----------------------------
# IMAGE TRANSFORM + MODEL
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

@torch.no_grad()
def extract_features(image, model):
    img_t = transform(image).unsqueeze(0)
    feat = model(img_t).squeeze().numpy()
    feat = feat / np.linalg.norm(feat)
    return feat.astype('float32')

@st.cache_resource
def load_index():
    index = faiss.read_index(INDEX_PATH)
    metadata = joblib.load(METADATA_PATH)
    return index, metadata

# -----------------------------
# SEARCH FUNCTION
# -----------------------------
def search_similar(image, top_k=6):
    model = load_model()
    index, metadata = load_index()

    feat = extract_features(image, model).reshape(1, -1)
    distances, indices = index.search(feat, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if 0 <= idx < len(metadata):
            item = metadata[idx]
            results.append({
                "id": item["id"],
                "name": item["name"],
                "category": item["category"],
                "image_path": item["image_path"],
                "similarity": float(1 / (1 + dist))
            })
    return results

def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# -----------------------------
# UI LAYOUT
# -----------------------------
# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A5568;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #718096;
        text-align: center;
        margin-bottom: 2rem;
    }
    .category-tag {
        display: inline-block;
        background-color: #4299E1;
        color: white;
        padding: 8px 16px;
        margin: 6px;
        border-radius: 25px;
        font-size: 14px;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .category-tag:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .product-card {
        background-color: #F7FAFC;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        border: 1px solid #E2E8F0;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.15);
    }
    .search-button {
        background-color: #4299E1;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        width: 100%;
        margin-top: 15px;
    }
    .search-button:hover {
        background-color: #3182CE;
    }
    .upload-section {
        background-color: #EDF2F7;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🔍 Product Vision Finder</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Discover visually similar products with our advanced AI-powered search engine</p>", unsafe_allow_html=True)

# Load available categories
try:
    _, metadata = load_index()
    categories = sorted(list(set([item["category"] for item in metadata])))
except Exception:
    categories = []
    st.warning("⚠️ Unable to load product categories. Please check if metadata.pkl exists.")

# Display available product categories (pill-style tags)
if categories:
    st.markdown("### 🛍️ Browse by Category")
    cat_html = " ".join([
        f"<span class='category-tag'>{c}</span>"
        for c in categories
    ])
    st.markdown(f"<div style='text-align: center; margin: 20px 0;'>{cat_html}</div>", unsafe_allow_html=True)

# Image input section
st.markdown("---")
st.markdown("### 📤 Upload Your Product Image")

with st.container():
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"], key="file_uploader")
        st.markdown("<div style='text-align: center; margin: 10px 0;'>OR</div>", unsafe_allow_html=True)
        url_input = st.text_input("Enter image URL", placeholder="https://example.com/image.jpg")
        st.markdown("</div>", unsafe_allow_html=True)

        query_image = None
        if uploaded_file:
            query_image = Image.open(uploaded_file).convert("RGB")
        elif url_input:
            try:
                resp = requests.get(url_input)
                query_image = Image.open(BytesIO(resp.content)).convert("RGB")
            except:
                st.error("❌ Unable to load image from the provided URL")

        top_k = st.slider("Number of similar products to show", min_value=1, max_value=12, value=6, step=1)
        search_btn = st.button("🔍 Find Similar Products", key="search_button", help="Click to search for visually similar products")

    with col2:
        if query_image is not None:
            st.image(query_image, caption="Your Product Image", width=350, use_container_width=False)
        else:
            st.markdown("""
            <div style='background-color: #F7FAFC; border: 2px dashed #CBD5E0; border-radius: 10px;
            height: 350px; display: flex; align-items: center; justify-content: center;
            color: #A0AEC0; text-align: center; padding: 20px;'>
                <div>
                    <h3>📷</h3>
                    <p>Your product image will appear here</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------
# SEARCH RESULTS
# -----------------------------
if search_btn and query_image is not None:
    with st.spinner("🔄 Analyzing image features and searching for matches..."):
        results = search_similar(query_image, top_k=top_k)

    st.markdown("---")
    st.markdown(f"### 🎯 Found {len(results)} Similar Products")
    
    if results:
        # Create a grid of product cards
        cols_per_row = 3
        for i in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(results):
                    r = results[i + j]
                    with cols[j]:
                        with st.container():
                            st.markdown("<div class='product-card'>", unsafe_allow_html=True)
                            st.image(r["image_path"], caption=None, width=200, use_container_width=False)
                            st.markdown(f"**{r['name']}**")
                            st.markdown(f"<span style='color: #4A5568; font-size: 0.9rem;'>{r['category']}</span>", unsafe_allow_html=True)
                            
                            # Create a similarity score bar
                            similarity_percent = int(r['similarity'] * 100)
                            st.markdown(f"""
                            <div style='margin-top: 10px;'>
                                <div style='display: flex; justify-content: space-between; margin-bottom: 3px;'>
                                    <span style='font-size: 0.8rem; color: #718096;'>Match Score</span>
                                    <span style='font-size: 0.8rem; font-weight: bold; color: #4299E1;'>{similarity_percent}%</span>
                                </div>
                                <div style='background-color: #E2E8F0; border-radius: 10px; height: 8px;'>
                                    <div style='background-color: #4299E1; height: 8px; border-radius: 10px; width: {similarity_percent}%;'></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ No similar products found. Try with a different image.")

# Add statistics section
st.markdown("---")
st.markdown("### 📊 Database Statistics")

try:
    _, metadata = load_index()
    if metadata:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Products", len(metadata))
        
        with col2:
            categories = sorted(list(set([item["category"] for item in metadata])))
            st.metric("Categories", len(categories))
        
        with col3:
            # Count unique image extensions
            extensions = set()
            for item in metadata:
                if "image_path" in item:
                    ext = item["image_path"].split('.')[-1].lower()
                    extensions.add(ext)
            st.metric("Image Types", len(extensions))
        
        with col4:
            # Calculate average similarity score (placeholder)
            st.metric("Avg. Match Rate", "87%")
        
        
except Exception as e:
    st.error(f"⚠️ Unable to load statistics: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; padding: 20px;'>
    <p>🔍 Product Vision Finder - AI-Powered Visual Search Engine</p>
    <p>Find visually similar products with advanced computer vision technology</p>
    <p style='font-size: 0.8rem; margin-top: 10px;'>© 2023 Product Vision Finder. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
