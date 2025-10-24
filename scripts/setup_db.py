import os
import streamlit as st
from supabase import create_client, Client

def verify_table_exists():
    """Verify that the products table exists in Supabase"""
    supabase_url = st.secrets['SUPABASE_URL'].strip()
    supabase_key = st.secrets['SUPABASE_KEY'].strip()
    
    if not supabase_url or not supabase_key:
        print("❌ Supabase URL or key not found in environment variables!")
        return False
    
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
    except Exception as e:
        print(f"❌ Error connecting to Supabase: {e}")
        return False
    
    # Check if products table exists
    try:
        response = supabase.table('products').select('id').limit(1).execute()
        if response.data:
            print("✅ Products table exists in Supabase!")
        else:
            print("✅ Products table exists but is empty.")
    except Exception as e:
        print(f"❌ Products table does not exist. Error: {e}")
        return False
    
    print("✅ Basic table verification completed successfully!")
    return True

if __name__ == "__main__":
    verify_table_exists()
