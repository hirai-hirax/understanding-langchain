from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

def qa_system(folder: str, question: str) -> str:
    docs = DirectoryLoader(folder, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
    return RetrievalQA.from_chain_type(ChatOpenAI(temperature=0), retriever=vectorstore.as_retriever()).invoke({"query": question})["result"]

if __name__ == "__main__":
    print(qa_system(input("Folder: "), input("Question: ")))