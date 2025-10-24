import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import time
from typing import List, Dict, Optional
import logging
from datetime import datetime

# Import database utilities
from db_utils import search_similar, get_all_categories, get_product_stats, get_product_by_id

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="🔍 Product Vision Finder",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛍️"
)

# Initialize session state for search history
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# -----------------------------
# UI LAYOUT - Enhanced with Mobile Responsive Design
# -----------------------------
# Custom CSS for styling and mobile responsiveness
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
        margin: 6px 3px;
        border-radius: 25px;
        font-size: 14px;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        cursor: pointer;
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
        box-shadow: 0 4px 6px rgba(0,0,0.1);
        transition: transform 0.2s;
        border: 1px solid #E2E8F0;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        overflow: hidden;
    }
    
    .product-card img {
        max-width: 100%;
        height: 200px; /* Fixed height for consistent sizing */
        object-fit: cover;
        border-radius: 8px;
        margin-bottom: 10px;
        width: 100%;
        flex-grow: 1;
        display: block;
    }
    
    .product-image {
        width: 100%;
        height: 100%;
        # object-fit: cover;
        # border-radius: 8px;
        # margin-bottom: 10px;
        # display: block;
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0.15);
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
    
    # .upload-section {
    #     background-color: #EDF2F7;
    #     padding: 20px;
    #     border-radius: 10px;
    #     margin-bottom: 20px;
    # }
    
    /* Mobile Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1rem;
        }
        
        [data-testid="column"] {
            gap: 1rem;
        }
        
        .product-card {
            margin-bottom: 10px;
            padding: 10px;
        }
        
        .category-tag {
            margin: 4px 2px;
            font-size: 12px;
            padding: 6px 12px;
        }
    }
    
    /* Tablet Responsive Design */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main-header {
            font-size: 2.2rem;
        }
    }
    
    /* Accessibility improvements */
    .sr-only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
    }
    
    /* Loading spinner */
    .loader {
        border: 5px solid #f3f3f3;
        border-top: 5px solid #4299E1;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        10% { transform: rotate(360deg); }
    }
    
    /* History item styling */
    .history-item {
        padding: 10px;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        margin-bottom: 8px;
        background-color: #F7FAFC;
        cursor: pointer;
    }
    
    .history-item:hover {
        background-color: #EDF2F7;
    }
    
    /* Pagination styling */
    .pagination {
        display: flex;
        justify-content: center;
        margin-top: 20px;
        gap: 10px;
    }
    
    .pagination button {
        padding: 8px 15px;
        border: 1px solid #429E1;
        background-color: white;
        color: #4299E1;
        border-radius: 5px;
        cursor: pointer;
    }
    
    .pagination button.active {
        background-color: #4299E1;
        color: white;
    }
    
    .pagination button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🔍 Product Vision Finder</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Discover visually similar products with our advanced AI-powered search engine</p>", unsafe_allow_html=True)

# Load available categories from PostgreSQL
try:
    categories = get_all_categories()
    categories = sorted(categories)
except Exception as e:
    categories = []
    st.warning(f"⚠️ Unable to load product categories from database: {str(e)}")

# Initialize session state
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = None
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None

# Sidebar for additional functionality
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
    st.markdown("### Search History")
    if st.session_state.search_history:
        for i, search in enumerate(st.session_state.search_history[-5:]):  # Show last 5 searches
            timestamp = search['timestamp'].strftime("%H:%M")
            if st.button(f"🕒 {timestamp} - {search['query_type']}", key=f"history_{i}"):
                # Load the previous search results
                st.session_state.prev_results = search['results']
                st.session_state.prev_query_image = search['query_image']
    else:
        st.write("No search history yet")

# Display available product categories (pill-style tags)
# if categories:
#     st.markdown("### 🛍️ Browse by Category")
#     # Create category filter using buttons
#     cols = st.columns(min(len(categories), 4))
#     selected_categories = []
#     for i, cat in enumerate(categories):
#         with cols[i % len(cols)]:
#             if st.button(f"🏷️ {cat}", key=f"cat_{cat}"):
#                 st.session_state.selected_category = cat
#                 selected_categories = [cat]
#             else:
#                 if f"cat_{cat}" in st.session_state:
#                     selected_categories.append(cat)

def validate_image(image):
    """Validate image dimensions and format"""
    if image is None:
        return False, "Image is None"
    
    # Check if image is too small
    if image.width < 10 or image.height < 10:
        return False, "Image is too small. Minimum dimensions are 10x10 pixels."
    
    # Check if image is too large (optional)
    if image.width > 5000 or image.height > 5000:
        return False, "Image is too large. Maximum dimensions are 5000x5000 pixels."
    
    return True, "Image is valid"

# Image input section with enhanced filtering options
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
        query_type = None
        if uploaded_file:
            try:
                # Validate image
                temp_image = Image.open(uploaded_file)
                is_valid, msg = validate_image(temp_image)
                if not is_valid:
                    st.error(f"❌ {msg}")
                else:
                    query_image = temp_image.convert("RGB")
                    query_type = "file_upload"
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
        elif url_input:
            try:
                resp = requests.get(url_input, timeout=10)
                resp.raise_for_status()
                temp_image = Image.open(BytesIO(resp.content))
                is_valid, msg = validate_image(temp_image)
                if not is_valid:
                    st.error(f"❌ {msg}")
                else:
                    query_image = temp_image.convert("RGB")
                    query_type = "url"
            except requests.exceptions.RequestException:
                st.error("❌ Invalid URL or unable to load image from the provided URL")
            except Exception as e:
                st.error(f"❌ Unable to load image from the provided URL: {str(e)}")

        # Enhanced filtering options
        st.markdown("### 🔍 Filters")
        top_k = st.slider("Number of similar products to show", min_value=1, max_value=20, value=6, step=1)
        min_similarity = st.slider("Minimum Similarity Score", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.2f")
        
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
            """, unsafe_allow_html=True)

# -----------------------------
# SEARCH RESULTS WITH FILTERING
# -----------------------------
if search_btn and query_image is not None:
    # Show loading indicator
    with st.spinner("🔄 Analyzing image features and searching for matches..."):
        start_time = time.time()
        results = search_similar(query_image, top_k=top_k, min_similarity=min_similarity)
        end_time = time.time()
        
        # Log the search
        search_record = {
            'timestamp': datetime.now(),
            'query_type': query_type,
            'query_image': query_image,
            'results': results,
            'search_time': end_time - start_time
        }
        st.session_state.search_history.append(search_record)
        # Keep only the last 20 searches
        if len(st.session_state.search_history) > 20:
            st.session_state.search_history = st.session_state.search_history[-20:]

    st.markdown("---")
    st.markdown(f"### 🎯 Found {len(results)} Similar Products (Search took {end_time - start_time:.2f}s)")
    
    if results:
        # Pagination
        items_per_page = 6
        total_pages = max(1, (len(results) + items_per_page - 1) // items_per_page)
        
        if total_pages > 1:
            page = st.slider("Page", 1, total_pages, 1)
        else:
            page = 1
            
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(results))
        page_results = results[start_idx:end_idx]
        
        # Create a grid of product cards
        cols_per_row = min(3, len(page_results))  # Responsive grid
        for i in range(0, len(page_results), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(page_results):
                    r = page_results[i + j]
                    with cols[j]:
                        with st.container():
                            st.markdown("<div class='product-card'>", unsafe_allow_html=True)
                            # Load and display the image from URL properly for deployment
                            image_url = r["image_url"]
                            try:
                                response = requests.get(image_url, timeout=30)
                                response.raise_for_status()
                                image = Image.open(BytesIO(response.content))
                                st.image(image, caption=None, use_container_width=True, clamp=True, output_format="auto")
                            except Exception as e:
                                st.warning(f"⚠️ Could not load image from URL: {image_url}")
                            
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
                            """, unsafe_allow_html=True)
                            
                            # Add a "View Details" button
                            # if st.button(f"👁️ Details", key=f"details_{r['id']}"):
                            #     st.session_state.selected_product = r
                            # st.markdown("</div>", unsafe_allow_html=True)
        
        # Pagination controls
        if total_pages > 1:
            st.markdown("<div class='pagination'>", unsafe_allow_html=True)
            col_list = st.columns(total_pages)
            for i in range(total_pages):
                with col_list[i]:
                    if st.button(f"{i+1}", key=f"page_{i+1}", type="secondary" if page != i+1 else "primary"):
                        st.session_state.page = i+1
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ No similar products found. Try with a different image or adjust your filters.")

# Show selected product details if available
if st.session_state.selected_product:
    st.markdown("---")
    st.markdown("### 📋 Product Details")
    product = st.session_state.selected_product
    col1, col2 = st.columns([1, 2])
    
    with col1:
        try:
            response = requests.get(product["image_url"], timeout=30)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Selected Product", use_column_width=True)
        except:
            st.warning(f"⚠️ Could not load image from URL: {product['image_url']}")
    
    with col2:
        st.markdown(f"### {product['name']}")
        st.markdown(f"**Category:** {product['category']}")
        st.markdown(f"**Similarity Score:** {product['similarity']:.4f} ({int(product['similarity']*100)}%)")
        st.markdown(f"**Product ID:** {product['id']}")
        
        if st.button("Close Details", key="close_details"):
            st.session_state.selected_product = None

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; padding: 20px;'>
    <p>🔍 Product Vision Finder - AI-Powered Visual Search Engine</p>
    <p>Find visually similar products with advanced computer vision technology</p>
    <p style='font-size: 0.8rem; margin-top: 10px;'>© 2023 Product Vision Finder. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
