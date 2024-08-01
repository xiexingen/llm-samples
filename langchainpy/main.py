
from langchain_community.llms import Ollama
# 加载器
from langchain_community.document_loaders import WebBaseLoader
# 文本分割
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 嵌入
from langchain_community.embeddings import OllamaEmbeddings
# 向量数据库
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

ollama = Ollama(
  base_url='http://localhost:11434',
  model="llama3"
)

# 加载文档
loader = WebBaseLoader("https://www.gutenberg.org/files/1727/1727-h/1727-h.htm")
# loader = WebBaseLoader('https://./assets/source.html')
data = loader.load()

# 拆分
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# 嵌入
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
# 向量数据
# 创建向量数据库
try:
  print("Creating vector store...")
  vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)
  print("Vector store created successfully.")
except Exception as e:
  print(f"Error creating vector store: {e}")

# 提问  Neleus 是谁，他的家人是谁？
question="Who is Neleus and who is in Neleus' family?"
docs = vectorstore.similarity_search(question)
len(docs)

# 将问题和文档的相关部分发送到模型中，提问
qachain=RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
res = qachain.invoke({"query": question})
print(res['result'])