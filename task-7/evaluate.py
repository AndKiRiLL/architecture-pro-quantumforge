#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self, index_path: str = "../task-3/faiss_index"):
        self.index_path = index_path
        
        logger.info("📥 Загрузка индекса...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = FAISS.load_local(
            index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        logger.info("📥 Загрузка LLM...")
        self.llm = Ollama(model="mistral:7b", temperature=0.7)
        
        self.prompt = PromptTemplate(
            template="""
Ты помощник QuantumForge. Отвечай на основе документов.
Если в документах нет ответа, скажи "Я не знаю".

ДОКУМЕНТЫ: {context}

ВОПРОС: {question}

ОТВЕТ:""",
            input_variables=["context", "question"]
        )
        
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )
        
        self.results = []
    
    def ask(self, question: str) -> Dict:
        """Задать вопрос и сохранить результат"""
        start_time = time.time()
        
        try:
            # Поиск в базе
            docs = self.vectorstore.similarity_search_with_score(question, k=3)
            
            if not docs or docs[0][1] < 0.4:
                return {
                    "answer": "Я не знаю",
                    "sources": [],
                    "score": float(0),
                    "chunks": 0,
                    "time": time.time() - start_time,
                    "status": "not_found"
                }
            
            result = self.qa({"query": question})
            answer = result["result"]
            sources = result.get("source_documents", [])
            
            # Проверка на "Я не знаю"
            if len(answer) < 10 or "не знаю" in answer.lower():
                return {
                    "answer": "Я не знаю",
                    "sources": [],
                    "score": float(docs[0][1]),
                    "chunks": len(docs),
                    "time": time.time() - start_time,
                    "status": "not_found"
                }
            
            return {
                "answer": answer,
                "sources": [doc.metadata.get('filename', 'unknown') for doc in sources],
                "score": float(docs[0][1]),
                "chunks": len(docs),
                "time": time.time() - start_time,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "answer": f"Ошибка: {e}",
                "sources": [],
                "score": float(0),
                "chunks": 0,
                "time": time.time() - start_time,
                "status": "error"
            }
    
    def evaluate_golden_set(self, golden_file: str) -> List[Dict]:
        """Запуск тестирования на золотом наборе"""
        with open(golden_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        logger.info(f"🔍 Запуск тестирования на {len(questions)} вопросах")
        
        for q in questions:
            logger.info(f"📝 {q['id']}: {q['question']}")
            
            result = self.ask(q['question'])
            
            # Проверка корректности ответа
            expected = q.get('expected', 'present')
            status = result['status']
            
            if expected == 'present':
                is_correct = status == 'success'
            else:
                is_correct = status == 'not_found'
            
            logger.info(f"   Статус: {status} | Ожидалось: {expected} | {'✅' if is_correct else '❌'}")
            
            self.results.append({
                "id": q['id'],
                "question": q['question'],
                "expected": expected,
                "answer": result['answer'],
                "sources": result['sources'],
                "status": status,
                "is_correct": is_correct,
                "score": result.get('score', 0),
                "time": result.get('time', 0)
            })
        
        return self.results
    
    def save_results(self, output_file: str = "logs.json"):
        """Сохранение результатов"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total": len(self.results),
            "correct": sum(1 for r in self.results if r['is_correct']),
            "wrong": sum(1 for r in self.results if not r['is_correct']),
            "success_rate": sum(1 for r in self.results if r['is_correct']) / len(self.results) * 100 if self.results else 0,
            "results": self.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Результаты сохранены в {output_file}")
        return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--golden', default='golden_questions.json')
    parser.add_argument('--output', default='evaluation_results.json')
    parser.add_argument('--index', default='../task-3/faiss_index')
    args = parser.parse_args()
    
    evaluator = RAGEvaluator(index_path=args.index)
    evaluator.evaluate_golden_set(args.golden)
    summary = evaluator.save_results(args.output)
    
    print("\n" + "="*50)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*50)
    print(f"Всего вопросов: {summary['total']}")
    print(f"✅ Правильных: {summary['correct']}")
    print(f"❌ Неправильных: {summary['wrong']}")
    print(f"📈 Успешность: {summary['success_rate']:.1f}%")
    print("="*50)

if __name__ == "__main__":
    main()