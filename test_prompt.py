from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# LangChainのRetrievalQAチェーンが使用するプロンプトを確認
docs = DirectoryLoader("test_docs", glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
qa = RetrievalQA.from_chain_type(ChatOpenAI(temperature=0), retriever=vectorstore.as_retriever())

# デフォルトのプロンプトテンプレートを表示
print("Default prompt template:")
print(qa.combine_documents_chain.llm_chain.prompt.messages[0].prompt.template)