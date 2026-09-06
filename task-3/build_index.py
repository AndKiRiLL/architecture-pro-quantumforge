import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from transformers import AutoTokenizer

class VectorIndexBuilder:
    def __init__(self, knowledge_dir: str = None, model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 300, chunk_overlap: int = 50, output_path: str = "./faiss_index"):
        
        # Определяем пути
        if knowledge_dir is None:
            knowledge_dir = Path(__file__).parent.parent / "knowledge_base"
        self.knowledge_dir = Path(knowledge_dir)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Токенизатор
        print("📥 Загружаю токенизатор...")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # Определяем устройство
        try:
            import torch
            self.use_gpu = torch.cuda.is_available()
            device = 'cuda' if self.use_gpu else 'cpu'
            if self.use_gpu:
                print(f"✅ GPU доступен: {torch.cuda.get_device_name(0)}")
            else:
                print("ℹ️  GPU не найден, использую CPU")
        except:
            self.use_gpu = False
            device = 'cpu'
            print("ℹ️  PyTorch не установлен, использую CPU")
        
        print(f"📥 Загружаю модель эмбеддингов: {model_name}")
        print(f"   Устройство: {device.upper()}")
        print(f"   Размер чанка: {chunk_size} токенов (overlap: {chunk_overlap})")
        print(f"   Папка с документами: {self.knowledge_dir}")
        
        # Эмбеддинг-модель
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.embedding_dim = self.embeddings.client.get_sentence_embedding_dimension()
        print(f"   Размерность эмбеддингов: {self.embedding_dim}")
        
        # Сплиттер с токенами
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.tokens_count,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        print("✅ Готово!\n")
    
    def tokens_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def load_documents(self) -> List[Document]:
        print("📂 Загружаю документы...")
        docs = []
        files = list(self.knowledge_dir.glob("*.txt"))
        print(f"   Найдено {len(files)} .txt файлов")
        
        for fp in files:
            with open(fp, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Парсим метаданные
            meta = {'source': str(fp), 'filename': fp.name}
            for line in content.split('\n')[:20]:
                if ':' in line:
                    key, val = line.split(':', 1)
                    key = key.strip()
                    if key in ['Источник', 'Заголовок', 'Оригинальный термин', 'Замена']:
                        meta[key] = val.strip()
            
            # Извлекаем текст
            text = content.split('='*60)[-1].strip() if '='*60 in content else content
            if text:
                docs.append(Document(page_content=text, metadata=meta))
        
        print(f"✅ Загружено {len(docs)} документов\n")
        return docs
    
    def build(self) -> Optional[FAISS]:
        print("="*60)
        print("🔨 ПОСТРОЕНИЕ ВЕКТОРНОГО ИНДЕКСА")
        print("="*60)
        
        docs = self.load_documents()
        if not docs:
            print("❌ Документы не найдены!")
            return None
        
        # Создаем чанки
        print("✂️  Разбиваю документы на чанки...")
        chunks = self.text_splitter.split_documents(docs)
        
        # Добавляем метаданные чанков
        for i, c in enumerate(chunks):
            c.metadata['chunk_id'] = i
            c.metadata['tokens'] = self.tokens_count(c.page_content)
        
        total_tokens = sum(c.metadata['tokens'] for c in chunks)
        avg_tokens = total_tokens // len(chunks) if chunks else 0
        
        print(f"   ✅ Создано {len(chunks)} чанков")
        print(f"   📊 Средний размер: {avg_tokens} токенов")
        print(f"   📊 Всего токенов: {total_tokens:,}\n")
        
        # Создаем индекс
        print("🔨 Создаю FAISS индекс...")
        start = time.time()
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        elapsed = time.time() - start
        print(f"   ✅ Индекс создан за {elapsed:.2f} секунд\n")
        
        # Сохраняем индекс
        print(f"💾 Сохраняю индекс в {self.output_path}...")
        vectorstore.save_local(str(self.output_path))
        
        # Сохраняем метаданные
        metadata = {
            'model': self.model_name,
            'dimension': self.embedding_dim,
            'chunk_size_tokens': self.chunk_size,
            'chunk_overlap_tokens': self.chunk_overlap,
            'device': 'cuda' if self.use_gpu else 'cpu',
            'documents': len(docs),
            'chunks': len(chunks),
            'total_tokens': total_tokens,
            'avg_tokens_per_chunk': avg_tokens,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'knowledge_base_path': str(self.knowledge_dir)
        }
        
        with open(self.output_path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Индекс сохранен!")
        print(f"   📁 {self.output_path / 'index.faiss'}")
        print(f"   📄 {self.output_path / 'metadata.json'}")
        
        # Итоговая статистика
        print("\n" + "="*60)
        print("📊 ИТОГОВАЯ СТАТИСТИКА")
        print("="*60)
        print(f"Модель эмбеддингов:  {self.model_name}")
        print(f"Размерность:         {self.embedding_dim}")
        print(f"Документов:          {len(docs)}")
        print(f"Чанков:              {len(chunks):,}")
        print(f"Средний размер:      {avg_tokens} токенов")
        print(f"Всего токенов:       {total_tokens:,}")
        print(f"Устройство:          {'GPU' if self.use_gpu else 'CPU'}")
        print(f"Время индексации:    {elapsed:.2f} секунд")
        print("="*60)
        
        return vectorstore

def test_search(index_path: str = "./faiss_index"):
    print("\n" + "="*60)
    print("🔍 ТЕСТОВЫЙ ПОИСК")
    print("="*60)
    
    # Загружаем метаданные
    metadata_path = Path(index_path) / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        print(f"📋 Информация об индексе:")
        print(f"   Модель: {meta.get('model', 'unknown')}")
        print(f"   Размерность: {meta.get('dimension', 'unknown')}")
        print(f"   Чанков: {meta.get('chunks', 'unknown')}")
        print(f"   Создан: {meta.get('timestamp', 'unknown')}")
    
    # Определяем устройство
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        device = 'cpu'
    
    print(f"\n📥 Загружаю индекс с устройства: {device.upper()}")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    print("✅ Индекс загружен\n")
    
    # Тестовые запросы
    queries = [
        "What is the Void Core and what is its purpose?",
        "Tell me about the Void Core superweapon",
        "Who was Mech-7, and what role did he play?",
        "What is the Synth Flux and how does it work?",
        "Who is Xarn Velgor and what is his role?"
    ]
    
    for q in queries:
        print(f"📝 Запрос: {q}")
        results = vectorstore.similarity_search_with_score(q, k=3)
        
        if not results:
            print("   ❌ Ничего не найдено")
            continue
        
        print(f"   Найдено {len(results)} результатов:")
        for i, (doc, score) in enumerate(results, 1):
            filename = doc.metadata.get('filename', 'unknown')
            title = doc.metadata.get('Заголовок', 'Без заголовка')
            chunk_id = doc.metadata.get('chunk_id', 'N/A')
            tokens = doc.metadata.get('tokens', 'N/A')
            
            print(f"\n   [{i}] Score: {score:.4f} | Файл: {filename}")
            print(f"       Заголовок: {title[:80]}...")
            print(f"       Чанк ID: {chunk_id} | Токенов: {tokens}")
            print(f"       Текст: {doc.page_content[:200]}...")
        
        print("-"*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Построение векторного индекса для RAG-бота')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Модель эмбеддингов')
    parser.add_argument('--chunk-size', type=int, default=300, help='Размер чанка в токенах (макс 512)')
    parser.add_argument('--chunk-overlap', type=int, default=50, help='Перекрытие в токенах')
    parser.add_argument('--output', default='./faiss_index', help='Папка для сохранения индекса')
    parser.add_argument('--input', default=None, help='Папка с документами')
    parser.add_argument('--no-test', action='store_true', help='Пропустить тестовый поиск')
    args = parser.parse_args()
    
    # Проверка на превышение лимита
    if args.chunk_size > 512:
        print(f"⚠️  chunk_size={args.chunk_size} превышает лимит 512!")
        print("   Автоматически уменьшаю до 350")
        args.chunk_size = 350
    
    # Строим индекс
    builder = VectorIndexBuilder(
        knowledge_dir=args.input,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        output_path=args.output
    )
    builder.build()
    
    # Тестовый поиск
    if not args.no_test:
        test_search(args.output)