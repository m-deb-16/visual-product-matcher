-- Supabase Table Schema for Visual Product Matcher
-- Run this SQL in Supabase SQL Editor to create the table

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create products table with 768-dim vector column
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    image_url TEXT NOT NULL,
    features vector(768)
);

-- Create HNSW index for similarity search
CREATE INDEX IF NOT EXISTS idx_features_hnsw
ON products USING hnsw (features vector_l2_ops);

-- Grant privileges
GRANT ALL PRIVILEGES ON TABLE products TO authenticator;
GRANT USAGE ON SCHEMA public TO postgres, anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO postgres, anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO postgres, anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO postgres, anon, authenticated, service_role;

-- Example of how data would be inserted (from data/products.csv)
-- INSERT INTO products (id, name, category, image_url, features) VALUES
-- (1, 'White backpack hanging on wall', 'Backpack', 'https://plus.unsplash.com/premium_photo-1664110691115-790e20a41744?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=653', '[0.1, 0.2, 0.3, ...]');

-- Example similarity search query that the application would use:
-- SELECT id, name, category, image_url,
--        (1 - (features <=> '[query_vector_values]')) AS similarity
-- FROM products
-- WHERE (1 - (features <=> '[query_vector_values]')) >= 0.0
-- ORDER BY similarity DESC
-- LIMIT 6;