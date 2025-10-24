#!/bin/bash

echo "Setting up Supabase database for Visual Product Matcher..."
echo "Note: You need to create the products table manually using Supabase SQL Editor first."
echo "Run the SQL commands in supabase_table_schema.sql file in your Supabase SQL Editor."

# Install required Python packages
echo "Installing required packages..."
pip install -r requirements_db.txt

# Verify the database table exists
echo "Verifying Supabase table exists..."
python scripts/setup_db.py

# Build the index by importing products into Supabase
echo "Building product index in Supabase..."
python scripts/build_index_db.py

echo "Supabase setup completed successfully!"
echo "You can now run the application with: streamlit run app.py"