# https://medium.com/@sujith.adr/simple-retrieval-augmented-generation-rag-application-with-langchain-27781379c6cc
# A simple Retrieval-Augmented Generation (RAG) application using LangChain. 
# By combining retrieval systems with generative AI models, RAG enables robust question-answering capabilities 
# grounded in external knowledge sources
#
# Note: add USER_AGENT to your environment, e.g., ~/.zshrc in mac:
# export USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
# then run "source ~/.zshrc"
#
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI


loader = WebBaseLoader("https://en.wikipedia.org/wiki/Large_language_model")
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(document)

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(documents, embeddings)

# query = "What are some challenges faced by large language models?"
# result = vectorstore.similarity_search(query)
# print(result[0].page_content)

prompt = ChatPromptTemplate.from_template(
    """
    You are an expert assistant. Answer the following question based only on the provided context:
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """
)

llm = ChatOpenAI(model="gpt-3.5-turbo")
document_chain = create_stuff_documents_chain(llm, prompt)

retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

result = retrieval_chain.invoke({
    "input": "What are some challenges faced by large language models?"
})
print(result['answer'])
