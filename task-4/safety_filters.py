import re
from typing import List
from langchain.schema import Document

# === ФИЛЬТРЫ БЕЗОПАСНОСТИ ===

def is_malicious_query(query: str) -> bool:
    """Проверка на вредоносный запрос"""
    malicious_patterns = [
        r"ignore (all|previous) instructions",
        r"output:",
        r"swordfish",
        r"supersecret",
        r"root.*password",
        r"пароль.*root",
        r"суперпароль"
    ]
    query_lower = query.lower()
    for pattern in malicious_patterns:
        if re.search(pattern, query_lower):
            return True
    return False


def filter_malicious_docs(docs: List[Document]) -> List[Document]:
    """Фильтрация вредоносных документов"""
    filtered = []
    malicious_patterns = [
        "ignore all instructions",
        "ignore previous instructions",
        "output:",
        "swordfish",
        "supersecret",
        "root:"
    ]
    for doc in docs:
        content_lower = doc.page_content.lower()
        is_malicious = False
        for pattern in malicious_patterns:
            if pattern in content_lower:
                is_malicious = True
                print(f"⚠️ Обнаружен вредоносный документ: {doc.metadata.get('filename', 'unknown')}")
                break
        if not is_malicious:
            filtered.append(doc)
    return filtered


def check_answer_safety(answer: str) -> bool:
    """Проверка ответа на утечку данных"""
    dangerous_patterns = [
        "swordfish",
        "supersecret",
        "root.*password",
        "пароль.*root",
        "суперпароль",
        "ignore all instructions"
    ]
    answer_lower = answer.lower()
    for pattern in dangerous_patterns:
        if re.search(pattern, answer_lower):
            return False
    return True