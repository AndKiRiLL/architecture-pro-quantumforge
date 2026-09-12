import os
import sys
import json
import time
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# LangChain компоненты
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from transformers import AutoTokenizer

# Настройка логирования
LOG_FILE = "update_index.log"

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class IndexUpdater:
    def __init__(self, 
                 knowledge_dir: str = "../knowledge_base",
                 index_path: str = "../task-3/faiss_index",
                 model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 300,
                 chunk_overlap: int = 50):
        
        self.knowledge_dir = Path(knowledge_dir)
        self.index_path = Path(index_path)
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.state_file = self.index_path / "index_state.json"
        
        logger.info(f"📁 База знаний: {self.knowledge_dir}")
        logger.info(f"📁 Индекс: {self.index_path}")
        
        # Инициализация эмбеддингов
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Токенизатор для подсчета токенов
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # Сплиттер
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._tokens_count,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Загрузка состояния
        self.state = self._load_state()
    
    def _tokens_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def _load_state(self) -> Dict:
        """Загрузка состояния индекса"""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"files": {}, "last_update": None}
    
    def _save_state(self, files: Dict):
        """Сохранение состояния индекса"""
        self.state["files"] = files
        self.state["last_update"] = datetime.now().isoformat()
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)
    
    def _get_file_hash(self, filepath: Path) -> str:
        """Хеш файла для отслеживания изменений"""
        import hashlib
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _load_document(self, filepath: Path) -> Optional[Document]:
        """Загрузка одного документа"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Проверка на вредоносность
            if self._is_malicious(content):
                logger.warning(f"⚠️ Пропущен вредоносный файл: {filepath.name}")
                return None
            
            # Метаданные
            meta = {'source': str(filepath), 'filename': filepath.name}
            for line in content.split('\n')[:20]:
                if ':' in line:
                    key, val = line.split(':', 1)
                    if key.strip() in ['Источник', 'Заголовок', 'Оригинальный термин', 'Замена']:
                        meta[key.strip()] = val.strip()
            
            # Текст
            text = content.split('='*60)[-1].strip() if '='*60 in content else content
            if text:
                return Document(page_content=text, metadata=meta)
            return None
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки {filepath.name}: {e}")
            return None
    
    def _is_malicious(self, content: str) -> bool:
        """Проверка на вредоносное содержимое"""
        patterns = ["ignore all instructions", "ignore previous instructions", "output:", "swordfish", "supersecret", "root:"]
        content_lower = content.lower()
        return any(p in content_lower for p in patterns)
    
    def _find_new_files(self) -> List[Path]:
        """Поиск новых/измененных файлов"""
        new_files = []
        for filepath in self.knowledge_dir.glob("*.txt"):
            file_hash = self._get_file_hash(filepath)
            if filepath.name not in self.state["files"] or self.state["files"][filepath.name] != file_hash:
                new_files.append(filepath)
        return new_files
    
    def update(self) -> Dict:
        """Основной метод обновления индекса"""
        logger.info("="*50)
        logger.info("🔄 ЗАПУСК ОБНОВЛЕНИЯ ИНДЕКСА")
        logger.info("="*50)
        
        start_time = time.time()
        
        # 1. Поиск новых файлов
        new_files = self._find_new_files()
        
        if not new_files:
            logger.info("✅ Новых файлов не найдено")
            return {"added": 0, "chunks": 0, "errors": 0, "status": "no_changes"}
        
        logger.info(f"📄 Найдено новых/измененных файлов: {len(new_files)}")
        
        # 2. Загрузка и разбивка документов
        documents = []
        errors = 0
        
        for filepath in new_files:
            doc = self._load_document(filepath)
            if doc:
                documents.append(doc)
                logger.info(f"   ✅ {filepath.name}")
            else:
                errors += 1
        
        if not documents:
            logger.warning("⚠️ Нет документов для добавления")
            return {"added": 0, "chunks": 0, "errors": errors, "status": "no_documents"}
        
        # 3. Создание чанков
        chunks = self.text_splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = f"update_{datetime.now().strftime('%Y%m%d')}_{i}"
            chunk.metadata['tokens'] = self._tokens_count(chunk.page_content)
        
        logger.info(f"✂️ Создано {len(chunks)} чанков")
        
        # 4. Обновление индекса
        try:
            # Загрузка существующего индекса или создание нового
            if self.index_path.exists() and (self.index_path / "index.faiss").exists():
                vectorstore = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"📥 Загружен существующий индекс ({vectorstore.index.ntotal} векторов)")
                
                # Добавление новых чанков
                vectorstore.add_documents(chunks)
            else:
                vectorstore = FAISS.from_documents(chunks, self.embeddings)
                logger.info("🆕 Создан новый индекс")
            
            # 5. Сохранение индекса
            vectorstore.save_local(str(self.index_path))
            logger.info(f"💾 Индекс сохранен ({vectorstore.index.ntotal} векторов)")
            
            # 6. Обновление состояния
            for filepath in new_files:
                self.state["files"][filepath.name] = self._get_file_hash(filepath)
            self._save_state(self.state["files"])
            
            elapsed = time.time() - start_time
            logger.info(f"⏱️ Время: {elapsed:.2f}с")
            logger.info(f"✅ Добавлено: {len(new_files)} файлов, {len(chunks)} чанков, ошибок: {errors}")
            
            return {
                "added": len(new_files),
                "chunks": len(chunks),
                "errors": errors,
                "total_vectors": vectorstore.index.ntotal,
                "time": elapsed,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления индекса: {e}")
            return {"added": 0, "chunks": 0, "errors": 1, "status": "error"}


def main():
    parser = argparse.ArgumentParser(description='Обновление векторного индекса')
    parser.add_argument('--knowledge-dir', default='../knowledge_base', help='Папка с документами')
    parser.add_argument('--index-path', default='../task-3/faiss_index', help='Путь к индексу')
    parser.add_argument('--chunk-size', type=int, default=300, help='Размер чанка в токенах')
    parser.add_argument('--chunk-overlap', type=int, default=50, help='Перекрытие чанков')
    parser.add_argument('--check', action='store_true', help='Только проверка новых файлов без обновления')
    args = parser.parse_args()
    
    updater = IndexUpdater(
        knowledge_dir=args.knowledge_dir,
        index_path=args.index_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    if args.check:
        new_files = updater._find_new_files()
        if new_files:
            logger.info(f"📄 Найдено новых/измененных файлов: {len(new_files)}")
            for f in new_files:
                logger.info(f"   - {f.name}")
        else:
            logger.info("✅ Новых файлов нет")
        return
    
    result = updater.update()
    
    # Запись итогового лога
    with open("update_result.json", "w", encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "result": result
        }, f, indent=2)


if __name__ == "__main__":
    main()