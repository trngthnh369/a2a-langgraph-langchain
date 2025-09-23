import time
from typing import Any, Optional, Dict

class CacheManager:
    """Simple in-memory cache manager"""
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = 1000  # Prevent memory bloat
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        try:
            if key in self.cache:
                item = self.cache[key]
                if time.time() - item['timestamp'] < item.get('ttl', 3600):
                    return item['value']
                else:
                    del self.cache[key]
            return None
        except Exception:
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set cached value"""
        try:
            # Simple cleanup if cache is too large
            if len(self.cache) >= self.max_size:
                # Remove oldest entries
                oldest_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])[:100]
                for old_key in oldest_keys:
                    del self.cache[old_key]
            
            self.cache[key] = {
                'value': value,
                'timestamp': time.time(),
                'ttl': ttl
            }
            return True
        except Exception:
            return False
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()