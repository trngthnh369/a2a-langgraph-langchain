import asyncio
import pandas as pd
import sys
import os
import argparse
import ast
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend import VectorStore

def join_product_string(row):
    """Product string creation"""
    parts = []
    
    # Title
    if 'title' in row and pd.notna(row['title']):
        parts.append(str(row['title']))
    
    # Promotion
    if 'product_promotion' in row and pd.notna(row['product_promotion']):
        promotion = str(row['product_promotion']).replace('<br>', ' ').replace('\n', ' ')
        parts.append(f"Khuyến mãi: {promotion}")
    
    # Specifications
    if 'product_specs' in row and pd.notna(row['product_specs']):
        specs = str(row['product_specs']).replace('<br>', ' ').replace('\n', ' ')
        parts.append(f"Thông số: {specs}")
    
    # Price
    if 'current_price' in row and pd.notna(row['current_price']):
        parts.append(f"Giá hiện tại: {row['current_price']}")
    
    if 'original_price' in row and pd.notna(row['original_price']):
        parts.append(f"Giá gốc: {row['original_price']}")
    
    # Colors
    if 'color_options' in row and pd.notna(row['color_options']):
        try:
            if isinstance(row['color_options'], str) and row['color_options'].startswith('['):
                colors = ast.literal_eval(row['color_options'])
                parts.append(f"Màu sắc: {', '.join(colors)}")
            else:
                parts.append(f"Màu sắc: {row['color_options']}")
        except:
            parts.append(f"Màu sắc: {row['color_options']}")
    
    # Brand and category
    if 'brand' in row and pd.notna(row['brand']):
        parts.append(f"Thương hiệu: {row['brand']}")
    
    if 'category' in row and pd.notna(row['category']):
        parts.append(f"Danh mục: {row['category']}")
    
    return " | ".join(parts)

async def main():
    """vector database builder"""
    parser = argparse.ArgumentParser(description="Build vector database")
    parser.add_argument("--csv", default="./data/products.csv", help="CSV file path")
    parser.add_argument("--collection", default="products", help="Collection name")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size")
    parser.add_argument("--max-records", type=int, default=200, help="Max records to process")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild collection")
    
    args = parser.parse_args()
    
    print("🚀 Vector Database Builder")
    print(f"📁 CSV File: {args.csv}")
    print(f"📊 Collection: {args.collection}")
    print(f"🔢 Batch Size: {args.batch_size}")
    print(f"📈 Max Records: {args.max_records}")
    print("=" * 50)
    
    if not os.path.exists(args.csv):
        print(f"❌ CSV file not found: {args.csv}")
        return
    
    try:
        # Load and process data
        print("📊 Loading CSV data...")
        df = pd.read_csv(args.csv)
        print(f"✅ Loaded {len(df)} records")
        
        # Limit records if specified
        if args.max_records:
            df = df.head(args.max_records)
            print(f"📏 Limited to {len(df)} records")
        
        # Filter valid records
        df = df[df['title'].notna() & (df['title'].str.len() > 5)]
        print(f"🔍 Filtered to {len(df)} valid records")
        
        # Create content strings
        print("🔨 Processing product information...")
        df['content'] = df.apply(join_product_string, axis=1)
        
        # Filter empty content
        df = df[df['content'].str.len() > 20]
        print(f"✅ Final dataset: {len(df)} records")
        
        # Prepare documents
        documents = []
        for _, row in df.iterrows():
            doc = {
                "content": row['content'],
                "title": str(row.get('title', '')),
                "current_price": str(row.get('current_price', '')),
                "product_specs": str(row.get('product_specs', ''))[:500],
                "brand": str(row.get('brand', '')),
                "category": str(row.get('category', ''))
            }
            documents.append(doc)
        
        # Initialize vector store
        print("🗃️ Initializing vector store...")
        vector_store = VectorStore(args.collection)
        
        if args.rebuild:
            print("🔄 Rebuilding collection...")
            try:
                vector_store.client.delete_collection(args.collection)
                print("🗑️ Deleted existing collection")
            except:
                pass
            vector_store._initialize()
        
        # Build vector database
        print("🔧 Building vector database...")
        start_time = time.time()
        
        total_added = await vector_store.add_documents(documents, args.batch_size)
        
        build_time = time.time() - start_time
        print(f"✅ Database built successfully!")
        print(f"📊 Documents added: {total_added}")
        print(f"⏱️ Build time: {build_time:.2f} seconds")
        print(f"⚡ Speed: {total_added/build_time:.1f} docs/second")
        
        # Test search
        print("\n🧪 Testing search functionality...")
        test_queries = ["iPhone", "Samsung", "Nokia"]
        
        for query in test_queries:
            results = await vector_store.search(query, k=2)
            print(f"🔍 '{query}': {len(results)} results")
            if results:
                print(f"   📱 Top result: {results[0]['metadata'].get('title', 'N/A')}")
        
        print("\n🎉 Vector database setup completed successfully!")
        
    except Exception as e:
        print(f"❌ Error building vector database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
