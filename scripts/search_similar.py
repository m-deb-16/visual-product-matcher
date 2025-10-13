import os
import faiss
import torch
import numpy as np
import joblib
from PIL import Image
from torchvision import models, transforms

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "image_index.faiss")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")

# -----------------------------
# IMAGE PREPROCESSING PIPELINE
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# -----------------------------
# FEATURE EXTRACTOR (same as build_index.py)
# -----------------------------
def get_feature_extractor():
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model


@torch.no_grad()
def extract_features(image_path, model):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)
    feat = model(img_t).squeeze().numpy()
    feat = feat / np.linalg.norm(feat)
    return feat


# -----------------------------
# SEARCH FUNCTION
# -----------------------------
def search_similar(query_image_path, top_k=5):
    # Load model, index, metadata
    model = get_feature_extractor()
    index = faiss.read_index(INDEX_PATH)
    metadata = joblib.load(METADATA_PATH)

    # Extract query image feature
    query_feat = extract_features(query_image_path, model).astype('float32')
    query_feat = np.expand_dims(query_feat, axis=0)

    # Perform search
    distances, indices = index.search(query_feat, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if 0 <= idx < len(metadata):
            item = metadata[idx]
            results.append({
                "id": item["id"],
                "name": item["name"],
                "category": item["category"],
                "image_path": item["image_path"],
                "similarity": float(1 / (1 + dist))  # convert distance → similarity (0–1)
            })
    return results


# -----------------------------
# TEST EXAMPLE
# -----------------------------
if __name__ == "__main__":
    test_image = "query.jpg"  # put a test image here
    results = search_similar(test_image, top_k=5)

    print("\n🔍 Top Similar Products:")
    for r in results:
        print(f"{r['name']} ({r['category']}) - Similarity: {r['similarity']:.3f}")
        print(f"🖼️ {r['image_path']}\n")
