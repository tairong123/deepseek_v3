from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader 
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import Ollama
import glob

# 教科書 PDF
textbook_docs = PyPDFLoader("docs/mydoc.pdf").load()

# 就診建議 PDF（Lung-RADS）
lung_rads_docs = PyPDFLoader("docs/lung-rads-assessment-categories.pdf").load()

# 臨床觀察資料（.docx Word）
clinical_docs = [UnstructuredWordDocumentLoader(path).load() for path in glob.glob("data/*.docx")]

# 2. 分段處理
all_docs = sum([textbook_docs, lung_rads_docs] + clinical_docs, [])
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(all_docs)

# 3. 建立向量資料庫
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(split_docs, embedding_model)

# 4. 對話記憶
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

llm = Ollama(model="deepseek-r1:7b")

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)
# 7. 問答介面
def chat():
    print("📄 請輸入模擬觀察內容（如：RUL 有 45mm 結節...），輸入 q 離開")

    while True:
        obs = input("你：")
        if obs.lower() == "q":
            break

        prompt = f"""
請你扮演一位放射科醫師，根據你學習的資料（包含 mydoc.pdf 的教材內容）與 Lung-RADS 評估指引，根據以下觀察內容撰寫診斷報告並以患者的角度給予易懂的建議。

請依照以下格式產出：

---
【肺部所見】：（條列肺部異常）
【心血管】：（條列心血管異常）
【縱隔】：（條列縱隔或淋巴結異常）
【Lung RADS 分類】：(根據給予的特徵條列分類內容)
【總結印象】：（根據觀察總結病灶性質與建議，可參考 lung-rads-assessment-categories.pdf，並寫出Lung
RADS，finding，management）
---

以下為觀察內容描述：
"{obs}"
"""

        result = qa_chain.invoke({"question": prompt})
        print("\n📋 AI 報告：\n")
        print(result["answer"])
        print("\n---\n")

chat()