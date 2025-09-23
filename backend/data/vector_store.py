import uuid
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
import google.genai as genai

from backend.core.config import settings

class VectorStore:
    """ Vector Store with ChromaDB and Gemini embeddings"""
    
    def __init__(self, collection_name: str = "products"):
        self.collection_name = self._sanitize_name(collection_name)
        self.client = None
        self.collection = None
        self._initialize()
        
        # Configure Gemini for embeddings
        
        genai.Client(api_key=settings.google_api_key)
    
    def _initialize(self):
        """Initialize ChromaDB"""
        try:
            chroma_settings = ChromaSettings(
                persist_directory=settings.chroma_db_path,
                anonymized_telemetry=False
            )
            
            self.client = chromadb.PersistentClient(
                path=settings.chroma_db_path,
                settings=chroma_settings
            )
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Product information"}
            )
            
            print(f"✅ Vector store initialized: {self.collection_name}")
            
        except Exception as e:
            print(f"❌ Vector store initialization failed: {e}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini (synchronous)."""
        try:
            result = genai.embed_content(
                model=settings.embedding_model,
                content=text
            )
            return result['embedding']
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            # Fallback to simple hash-based embedding
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            return [float(ord(c)) for c in hash_obj.hexdigest()[:384]]
    
    async def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 50):
        """Add documents to vector store"""
        total_added = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Prepare batch data
            texts = [doc.get("content", "") for doc in batch]
            metadatas = [self._prepare_metadata(doc) for doc in batch]
            ids = [str(uuid.uuid4()) for _ in batch]
            
            # Generate embeddings
            embeddings = []
            for text in texts:
                embedding = self.get_embedding(text)
                embeddings.append(embedding)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            
            total_added += len(batch)
            print(f"📊 Added batch {i//batch_size + 1}: {len(batch)} documents")
        
        print(f"✅ Total documents added: {total_added}")
        return total_added
    
    def search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """Search for similar documents (synchronous)."""
        k = k or settings.vector_search_k
        
        try:
            # Generate query embedding
            query_embedding = self.get_embedding(query)
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0.0,
                        'relevance_score': round(1 - results['distances'][0][i], 3) if results['distances'] else 1.0
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Vector search failed: {e}")
            return []
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize collection name"""
        import re
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        return name.strip("_").lower()
    
    def _prepare_metadata(self, doc: Dict[str, Any]) -> Dict[str, str]:
        """Prepare metadata for ChromaDB"""
        metadata = {}
        for key, value in doc.items():
            if key != "content" and value is not None:
                str_value = str(value)
                if len(str_value) > 500:
                    str_value = str_value[:500] + "..."
                metadata[key] = str_value
        return metadata