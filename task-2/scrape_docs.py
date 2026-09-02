import os
import json
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime

class DocumentScraper:
    def __init__(self, output_dir="../knowledge_base", terms_file="./terms_map.json", urls_file="./urls.json"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.replacement_dict = self.load_json(terms_file) or {}
        self.urls_data = self.load_json(urls_file, key='urls') or []
        print(f"✅ Загружено {len(self.replacement_dict)} терминов и {len(self.urls_data)} URL")
    
    def load_json(self, filepath, key=None):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data[key] if key else data
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    def replace_terms(self, text):
        for old, new in sorted(self.replacement_dict.items(), key=lambda x: len(x[0]), reverse=True):
            if ' ' in old:
                text = text.replace(old, new)
            else:
                text = re.sub(r'\b' + re.escape(old) + r'\b', new, text, flags=re.IGNORECASE)
        return text
    
    def clean_text(self, text):
        return re.sub(r'\s+', ' ', text).strip()
    
    def fetch_page(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()
            main = soup.find('main') or soup.find('article') or soup.find('div', class_='mw-parser-output') or soup.find('body')
            text = main.get_text(separator=' ', strip=True) if main else soup.get_text(separator=' ', strip=True)
            return self.clean_text(text)
        except Exception as e:
            print(f"❌ Ошибка {url}: {e}")
            return None
    
    def process(self, url, term=None, replacement=None):
        print(f"📥 {term or url}")
        text = self.fetch_page(url)
        if not text:
            return False
        
        original_len = len(text)
        text = self.replace_terms(text)
        
        # Генерация имени файла
        if term:
            filename = re.sub(r'[^\w\-_]', '_', term.replace(' ', '_')) + '.txt'
        else:
            filename = re.sub(r'[^\w\-_]', '_', urlparse(url).path.strip('/').replace('/', '_') or 'page') + '.txt'
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"""Источник: {url}
Заголовок: {term or 'N/A'}
Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Длина: {original_len} → {len(text)} символов
Замен: {len(self.replacement_dict)} терминов
{'='*60}

{text}""")
        
        print(f"✅ {filename} ({len(text)} символов)")
        return True
    
    def process_single(self):
        url = input("Введите URL: ").strip()
        if url:
            self.process(url)
    
    def process_batch(self):
        if not self.urls_data:
            print("❌ Нет URL в urls.json")
            return
        
        success = 0
        for i, item in enumerate(self.urls_data, 1):
            print(f"\n[{i}/{len(self.urls_data)}]")
            if self.process(item['url'], item.get('term'), item.get('replacement')):
                success += 1
        
        print(f"\n{'='*50}\n✅ Готово: {success}/{len(self.urls_data)}")

def main():
    scraper = DocumentScraper()
    print("="*50)
    print("📄 Извлекатель текста")
    print("="*50)
    
    while True:
        print("\n1. Одна ссылка\n2. Массовая загрузка из urls.json\n3. Выход")
        choice = input("Выбор: ").strip()
        
        if choice == '1':
            scraper.process_single()
        elif choice == '2':
            scraper.process_batch()
        elif choice == '3':
            print("👋 До свидания!")
            break
        else:
            print("❌ Неверный выбор")

if __name__ == "__main__":
    main()