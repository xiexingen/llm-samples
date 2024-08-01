from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 1. 初始化llm, 让其流式输出
## temperature 控制文本生成的创造性，为0时响应是可预测，始终选择下一个最可能的单词，这对于事实和准确性非常重要的答案是非常有用的。为 1时生成文本会选择更多的单词，会产生更具创意但不可能预测的答案
## top_p 或 核心采样 决定了生成时要考虑多少可能的单词。高top_p值意味着模型会考虑更多可能的单词，甚至是可能性较低的单词，从而使生成的文本更加多样化
## 较低的temperature和较高的top_p，可以产生具有创意的连贯文字。由于temperature较低，答案通常具有逻辑性和连贯性，但由于top_p较高，答案仍然具有丰富的词汇和观点。比较适合生成信息类文本，内容清晰且能吸引读者。
## 较高的temperature和较低的top_p，可能会把单词以难以预测的方式组合在一起。 生成的文本创意高，会出现意想不到的结果，适合创作
llm = Ollama(
  model="llama3",
  temperature=0.1,
  top_p=0.4,
  callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

# 获取RAG检索内容并分块
##`BeautifulSoup'解析网页内容：按照标签、类名、ID 等方式来定位和提取你需要的内容
## chunk_overlap：分块的重叠部分,重叠有助于降低将语句与与其相关的重要上下文分开的可能性。
## chunk_size： 分块的大小，合理的分词设置会提高RAG的效果
import bs4
#Load HTML pages using `urllib` and parse them with `BeautifulSoup'
from langchain_community.document_loaders import WebBaseLoader
#文本分割
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader(
    web_paths=("https://vuejs.org/guide/introduction.html#html",),
    bs_kwargs=dict(
      parse_only=bs4.SoupStrainer(
          class_=("content",),
          # id=("article-root",)
      )
    ),
)
docs = loader.load()
# chunk_overlap：分块的重叠部分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 内容基于本地的词嵌入模型 nomic-embed-text 嵌入向量数据库中

# 向量嵌入 ::: conda install onnxruntime -c conda-forge
from langchain_community.vectorstores import Chroma
# 有许多嵌入模型
from langchain_community.embeddings import OllamaEmbeddings
# 基于ollama运行嵌入模型 nomic-embed-text ： A high-performing open embedding model with a large token context window.
vectorstore = Chroma.from_documents(
  documents=splits,
  embedding=OllamaEmbeddings(model="nomic-embed-text") # herald/dmeta-embedding-zh
)
# 相似搜索
texts= vectorstore.similarity_search("vue")

from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate(
    input_variables=['context', 'question'],
    template=
    """你可以在此提问，如果不知道我会说:不知道
    without any explanation Question: {question} Context: {context} Answer:"""
)

from langchain.chains import RetrievalQA
# 向量数据库检索器
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)
# what is Composition API？
question = "什么是Vue?"
result = qa_chain.invoke({"query": question})

# 再来个它不知道的
question = "what is react?"
result = qa_chain.invoke({"query": question})
