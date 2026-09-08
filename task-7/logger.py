import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

class QueryLogger:
    def __init__(self, log_file: str = "logs.jsonl"):
        self.log_file = log_file
        
    def log(self, query: str, answer: str, sources: list, status: str, 
            chunks_found: int, answer_length: int):
        """Логирование запроса"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer[:500] if answer else None,
            "answer_length": answer_length,
            "sources": [s.metadata.get('filename', 'unknown') for s in sources] if sources else [],
            "chunks_found": chunks_found,
            "status": status,  # 'success', 'not_found', 'empty'
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        return entry