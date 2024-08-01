from langchain_community.document_loaders import WebBaseLoader

# 加载数据
loader = WebBaseLoader("https://www.nba.com/player/2544/lebron-james")
data = loader.load()

# 将数据转换为向量数据库
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 使用 HuggingFace 的嵌入模型, 例如 sentence-transformers/all-mpnet-base-v2
embeddings_model = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-mpnet-base-v2",
  model_kwargs= {'device': 'cpu'},
  encode_kwargs={'normalize_embeddings': False}
)

# 创建 Chroma 向量数据库
vectorstore = Chroma.from_documents(documents=data, embedding=embeddings_model, persist_directory="./chroma_db")

# 构建RAG管道
#在这个步骤中，我们将导入所需的各种包，如chatolama、output parser、chat prompt等，并导入Transformers包。接着，使用chatolama加载Gemma 2模型，并从Lang chain Hub导入一个RAG prompt。我们会创建一个QA链，传入我们选择的LLM（Gemma 2）、检索器和RAG prompt，然后运行这个链，传入问题并检视结果

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain import hub
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM,AutoTokenizer, pipeline

# 使用 ChatOllama 模型
llm =ChatOllama(model="gemma2:9b")

# 拉取 RAG 提示模板
prompt = hub.pull("rlm/rag-prompt")

# 使用向量数据库创建检索器
vectorstore =Chroma(persist_directory="./chroma_db", embedding_function=HuggingFaceEmbeddings())

# 创建 RetrievalQA 链
qa_chain =RetrievalQA.from_chain_type(
  llm,
  retriever=vectorstore.as_retriever(),
  chain_type_kwargs={"prompt": prompt}
)

# 提问并获取结果
question ="这些统计数据属于谁？"
result = qa_chain({"query": question})

# 输出结果
print(result["result"])