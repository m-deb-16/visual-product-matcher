# Visual Product Matcher

## Overview

Visual Product Matcher is an AI-powered application that helps users find visually similar products using advanced computer vision technology. The application uses deep learning models to analyze product images and find similar items in the database based on visual features.

## Features

- Advanced image similarity search using ResNet50 model
- Visual similarity scoring with percentage matching
- Multiple product categories support
- Modern, responsive UI built with Streamlit
- Supabase PostgreSQL database with pgvector extension for similarity search
- Easy deployment on Streamlit Cloud with remote database access

## How It Works

1. Upload an image or provide an image URL
2. The AI analyzes the visual features using a pre-trained ResNet50 model
3. The PostgreSQL database with pgvector extension finds the most similar products
4. Results are displayed with similarity scores and product information

## Technical Details

- **Model**: Vision Transformer ViT (google/vit-base-patch16-224)
- **Search Engine**: Supabase PostgreSQL with pgvector extension using HNSW index
- **Framework**: Streamlit + PyTorch + Transformers + Supabase-py
- **Features**: Image embedding, similarity search, visual interface

## Requirements

- Python 3.8+
- Supabase PostgreSQL with pgvector extension
- Streamlit
- PyTorch
- Transformers
- supabase-py
- psycopg2-binary (for vector operations)
- pgvector
- python-dotenv
- Other dependencies listed in requirements_db.txt

## Database Setup

1. Create a Supabase account at https://supabase.com/
2. Create a new project and note down your project URL and API key (do not use the example credentials provided in the .env file)
3. Update the .env file with your actual Supabase credentials:
   ```
   SUPABASE_URL=your-project-name.supabase.co
   SUPABASE_KEY=your-anon-or-service-key
   DATABASE_URL=postgresql://postgres:your_password@your_project_id.supabase.co:5432/postgres
   ```
4. Create the database table using Supabase SQL Editor:

   - Go to your Supabase dashboard
   - Navigate to "SQL Editor"
   - Copy and run the SQL commands from the `supabase_table_schema.sql` file
   - This will create the products table with the pgvector extension and necessary indexes

5. Verify the setup by running the verification script:

```bash
python scripts/setup_db.py
```

6. Build the product index in Supabase (after table is created):

```bash
python scripts/build_index_db.py
```

**Note**: If you see an error like "could not translate host name" when running the verification or build scripts, it means the example credentials in `.env` have been replaced with your own Supabase project credentials. This is expected behavior when using the default template credentials.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/visual-product-matcher.git
   cd visual-product-matcher
   ```

2. Set up the database as described above

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Deployed Application

The application is deployed on Streamlit Cloud and can be accessed at: https://vispmat.streamlit.app/

## Approach

This visual product matcher uses a pre-trained Vision Transformer (ViT) model to extract 768-dimensional image features and Supabase PostgreSQL with the pgvector extension and HNSW indexing for efficient similarity search. Images are converted to feature vectors, stored in the database along with their URLs, and searched using cosine similarity. The application uses the Supabase Python client for database operations with direct PostgreSQL connections for vector operations. The Streamlit interface allows users to upload images and find visually similar products from the database. Key improvements include migrating from a local file system to Supabase for better scalability, deployment compatibility, and remote access.
