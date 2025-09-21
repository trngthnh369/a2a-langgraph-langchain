import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import re
import chromadb
import uuid
import ast
import time

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ChromaDB setup
chroma_client = chromadb.PersistentClient("db")

def get_embedding(text: str) -> list[float]:
    """Generate embeddings using Gemini API"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating embedding for text: {e}")
        # Fallback: create a simple hash-based embedding
        import hashlib
        hash_object = hashlib.md5(text.encode())
        # Convert to a simple numeric representation (not recommended for production)
        return [float(ord(c)) for c in hash_object.hexdigest()[:384]]  # 384 dimensions

def sanitize_collection_name(name: str) -> str:
    """Sanitize collection name to be ChromaDB-compatible."""
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = name.strip("_")
    return name.lower()

def sanitize_metadata(record):
    """Convert None values in metadata to empty string and exclude embeddings."""
    sanitized_record = {
        k: (str(v) if v is not None else "") for k, v in record.items() 
        if k != "embedding"
    }
    return sanitized_record

def join_string(row):
    """Join product information into a single searchable string"""
    title = row.get('title', '') or ''
    product_promotion = row.get('product_promotion', '') or ''
    product_specs = row.get('product_specs', '') or ''
    current_price = row.get('current_price', '') or ''
    color_options = row.get('color_options', '') or ''

    final_string = ""
    
    if title:
        final_string += f"{title}"

    if product_promotion:
        product_promotion = str(product_promotion).replace("<br>", " ").replace("\n", " ")
        final_string += f" {product_promotion}"

    if product_specs:
        product_specs = str(product_specs).replace("<br>", " ").replace("\n", " ")
        final_string += f" {product_specs}"

    if current_price:
        final_string += f" có giá: {current_price}"

    if color_options:
        final_string += " có màu sắc: "
        try:
            if isinstance(color_options, str) and color_options.startswith('['):
                colors = ast.literal_eval(color_options)
                final_string += ", ".join(colors)
            else:
                final_string += str(color_options)
        except:
            final_string += str(color_options)

    return final_string

def main():
    """Main function to build vector search database"""
    print("Starting vector search database build...")
    
    # Check if CSV file exists
    csv_path = "./hoanghamobile.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        print("Please make sure the CSV file is in the correct location.")
        return
    
    try:
        # Load the CSV file
        print("Loading CSV data...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records from CSV")
        
        # Create information column
        print("Processing product information...")
        df['information'] = df.apply(join_string, axis=1)
        
        # Filter out empty information
        df = df[df['information'].notna() & (df['information'].str.len() > 10)]
        print(f"Filtered to {len(df)} records with valid information")
        
        # Limit to first 50 records for testing (remove this in production)
        df = df.head(50)
        print(f"Using first {len(df)} records for vector database")
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = []
        
        for idx, row in df.iterrows():
            try:
                embedding = get_embedding(row['information'])
                embeddings.append(embedding)
                
                if (idx + 1) % 5 == 0:
                    print(f"Generated embeddings for {idx + 1}/{len(df)} records")
                    time.sleep(1)  # Rate limiting
                    
            except Exception as e:
                print(f"Error generating embedding for row {idx}: {e}")
                # Use a zero vector as fallback
                embeddings.append([0.0] * 384)
        
        df['embedding'] = embeddings
        
        # Prepare metadata
        print("Preparing metadata...")
        metadatas = []
        for _, row in df.iterrows():
            metadata = {
                "information": row["information"],
                "title": str(row.get("title", "")),
                "current_price": str(row.get("current_price", "")),
                "product_specs": str(row.get("product_specs", ""))[:500]  # Limit length
            }
            metadatas.append(metadata)
        
        # Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in range(len(df))]
        
        # ChromaDB setup
        print("Setting up ChromaDB...")
        collection_name = sanitize_collection_name("products")
        
        # Delete existing collection if it exists
        try:
            chroma_client.delete_collection(name=collection_name)
            print("Deleted existing collection")
        except:
            pass
        
        # Create new collection
        collection = chroma_client.get_or_create_collection(name=collection_name)
        
        # Insert data in batches
        print("Inserting data into ChromaDB...")
        batch_size = 10
        
        for i in range(0, len(df), batch_size):
            end_idx = min(i + batch_size, len(df))
            batch_embeddings = embeddings[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            batch_ids = ids[i:end_idx]
            
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
            
            print(f"Inserted batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
        
        print(f"\nSuccessfully built vector search database!")
        print(f"- Total documents: {len(df)}")
        print(f"- Collection name: {collection.name}")
        print(f"- Database location: ./db")
        
        # Test the database
        print("\nTesting the database...")
        test_query = "Nokia 3210"
        test_embedding = get_embedding(test_query)
        
        results = collection.query(
            query_embeddings=[test_embedding],
            n_results=2
        )
        
        if results['metadatas']:
            print(f"Test query '{test_query}' returned {len(results['metadatas'][0])} results")
            print("Database is working correctly!")
        else:
            print("Warning: Test query returned no results")
            
    except Exception as e:
        print(f"Error building vector database: {e}")
        print("Please check your data and try again.")

if __name__ == "__main__":
    main()