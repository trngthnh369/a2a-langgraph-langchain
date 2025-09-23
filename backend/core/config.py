from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # A2A Original Settings
    host: str = Field("localhost", env="HOST")
    port: int = Field(10000, env="PORT")
    
    # API Keys
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    serper_api_key: str = Field(..., env="SERPER_API_KEY")
    
    # Features
    enable_web_search: bool = Field(True, env="ENABLE_WEB_SEARCH")
    enable_rag: bool = Field(True, env="ENABLE_RAG")
    enable_caching: bool = Field(True, env="ENABLE_CACHING")
    
    # Vector Database
    chroma_db_path: str = Field("./data/vector_db", env="CHROMA_DB_PATH")
    embedding_model: str = Field("models/text-embedding-004", env="EMBEDDING_MODEL")
    vector_search_k: int = Field(5, env="VECTOR_SEARCH_K")
    
    # Performance
    cache_ttl: int = Field(3600, env="CACHE_TTL")
    max_context_length: int = Field(4000, env="MAX_CONTEXT_LENGTH")
    
    model_config = SettingsConfigDict(env_file="../../.env")

settings = Settings()