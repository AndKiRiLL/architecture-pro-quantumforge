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

class VectorIndexBuilder:
    def __init__(self, knowledge_dir: str = None, model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 300, chunk_overlap: int = 50, output_path: str = "./faiss_index"):
        
        # Авто-определение путей
        if knowledge_dir is None:
            knowledge_dir = Path(__file__).parent.parent / "knowledge_base"
        self.knowledge_dir = Path(knowledge_dir)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем параметры для метаданных
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Авто-определение GPU
        try:
            import torch
            self.use_gpu = torch.cuda.is_available()
            device = 'cuda' if self.use_gpu else 'cpu'
        except:
            self.use_gpu = False
            device = 'cpu'
        
        print(f"📥 Модель: {model_name} | Устройство: {device.upper()} | Документы: {self.knowledge_dir}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.embedding_dim = self.embeddings.client.get_sentence_embedding_dimension()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            length_function=len, separators=["\n\n", "\n", " ", ""]
        )
    
    def load_documents(self) -> List[Document]:
        docs = []
        for fp in self.knowledge_dir.glob("*.txt"):
            with open(fp, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Парсим метаданные
            meta = {'source': str(fp), 'filename': fp.name}
            for line in content.split('\n')[:20]:
                if ':' in line:
                    key, val = line.split(':', 1)
                    if key.strip() in ['Источник', 'Заголовок', 'Оригинальный термин', 'Замена']:
                        meta[key.strip()] = val.strip()
            
            # Извлекаем текст
            text = content.split('='*60)[-1].strip() if '='*60 in content else content
            if text:
                docs.append(Document(page_content=text, metadata=meta))
        
        print(f"✅ {len(docs)} документов")
        return docs
    
    def build(self) -> Optional[FAISS]:
        print("="*50)
        docs = self.load_documents()
        if not docs:
            return None
        
        # Сплит
        chunks = self.text_splitter.split_documents(docs)
        for i, c in enumerate(chunks):
            c.metadata['chunk_id'] = i
        
        print(f"✂️ {len(chunks)} чанков (ср. {sum(len(c.page_content) for c in chunks)//len(chunks) if chunks else 0} символов)")
        
        # Индекс
        print(f"🔨 Начинаем создание индекса")
        start = time.time()
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print(f"🔨 Индекс создан за {time.time()-start:.2f}с")
        
        # Сохранение
        vectorstore.save_local(str(self.output_path))
        with open(self.output_path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump({
                'model': self.model_name,  # ← исправлено!
                'dimension': self.embedding_dim,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'device': 'cuda' if self.use_gpu else 'cpu',
                'documents': len(docs),
                'chunks': len(chunks),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        print(f"💾 Сохранено: {self.output_path}")
        return vectorstore

def test_search(index_path: str = "./faiss_index"):
    print("\n" + "="*50 + "\n🔍 ТЕСТОВЫЙ ПОИСК\n" + "="*50)
    
    # Авто-определение GPU для теста
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        device = 'cpu'
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    queries = [
        "What is the Void Core and what is its purpose?",
        "Tell me about the Void Core superweapon",
        "What is the Void Core used for in Star Wars?",
        "Who was Mech-7, and what role did he play?",
    ]
    
    for q in queries:
        print(f"\n📝 {q}")
        results = vectorstore.similarity_search_with_score(q, k=2)
        for doc, score in results:
            print(f"   [{score:.3f}] {doc.metadata.get('filename', 'unknown')}")
            print(f"   {doc.page_content[:150]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='all-MiniLM-L6-v2')
    parser.add_argument('--chunk-size', type=int, default=300)
    parser.add_argument('--chunk-overlap', type=int, default=50)
    parser.add_argument('--output', default='./faiss_index')
    parser.add_argument('--input', default=None)
    parser.add_argument('--no-test', action='store_true')
    args = parser.parse_args()
    
    builder = VectorIndexBuilder(
        knowledge_dir=args.input,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        output_path=args.output
    )
    builder.build()
    
    if not args.no_test:
        test_search(args.output)