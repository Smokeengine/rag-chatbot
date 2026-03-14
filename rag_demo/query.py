from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings

def ask(query):
    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=OllamaEmbeddings(model='nomic-embed-text')
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # use the new API safely
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant data found in your knowledge base."

    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer concisely using only the context above."

    llm = OllamaLLM(model="mistral")
    response = llm.invoke(prompt)
    return response

if __name__ == "__main__":
    while True:
        q = input("Ask: ").strip()
        if q.lower() in ["exit", "quit", "q"]:
            break
        print(ask(q))
