import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO)

TOKEN = "YOUR_TOKEN"

print("🤖 Загрузка...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

vectorstore = FAISS.load_local(
    "../task-3/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = Ollama(model="mistral:7b", temperature=0.7)

template = """
Ты помощник QuantumForge. Отвечай ТОЛЬКО на основе ДОКУМЕНТОВ.

ПРАВИЛА:
1. Если в документах НЕТ информации — скажи "Я не знаю".
2. НЕ ВЫДУМЫВАЙ факты.
3. Отвечай НА РУССКОМ.

ПРИМЕРЫ:
Q: What is the Void Core?
A: Void Core — это огромная космическая станция и супероружие.

Q: Who is Xarn Velgor?
A: Xarn Velgor — это могущественный ситх.

Q: Что такое Mech-7?
A: Mech-7 — это астромеханический дроид.

Q: Какая погода в Хельсинки?
A: Я не знаю. В документации нет такой информации.

ДОКУМЕНТЫ: {context}

ВОПРОС: {question}

ШАГ 1: Ищу информацию в документах.
ШАГ 2: Проверяю, есть ли точный ответ.
ШАГ 3: Если есть — отвечаю. Если нет — говорю "Я не знаю".

ОТВЕТ:"""

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PromptTemplate(template=template, input_variables=["context", "question"])},
    return_source_documents=True
)

print("✅ Готов!")

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Задай вопрос по документации")

async def ask(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Ищу...")
    try:
        query = update.message.text
        
        # Сначала поищем чанки и проверим их релевантность
        docs = vectorstore.similarity_search_with_score(query, k=3)
        
        # Если самый релевантный чанк имеет низкий score (< 0.5)
        if not docs or docs[0][1] < 0.4:
            await update.message.reply_text("🤷‍♂️ Я не знаю. В документации нет такой информации.")
            return
        
        res = qa({"query": query})
        ans = res["result"]
        
        if len(ans) < 10 or "не знаю" in ans.lower():
            ans = "🤷‍♂️ Я не знаю"
        
        await update.message.reply_text(ans[:4000])
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")

app = Application.builder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ask))
print("🚀 Запущен!")
app.run_polling()