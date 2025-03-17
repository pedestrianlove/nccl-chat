from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
import gradio as gr

# import the .env file
from dotenv import load_dotenv

load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

embeddings_model = OllamaEmbeddings(
    base_url="http://192.168.1.12:11434",
    model="nomic-embed-text",
)

# initiate the model
llm = ChatOllama(
    base_url="http://192.168.1.12:11434",
    model="deepseek-r1:32b-qwen-distill-q4_K_M",
    temperature=0.5,
)

# connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Set up the vectorstore to be the retriever
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={"k": num_results})


# call this function for every message added to the chatbot
def stream_response(message, history):
    # print(f"Input: {message}. History: {history}\n")

    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    # add all the chunks to 'knowledge'
    knowledge = ""

    for doc in docs:
        knowledge += doc.page_content + "\n\n"

    # make the call to the LLM (including prompt)
    if message is not None:
        partial_message = ""

        rag_prompt = f"""
        You are an assistent which answers questions based on knowledge which is provided to you.
        While answering, you don't use your internal knowledge, 
        but solely the information in the "The knowledge" section.
        You don't mention anything to the user about the povided knowledge.

        The question: {message}

        Conversation history: {history}

        The knowledge: {knowledge}

        """

        # stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content.replace(
                "<think>", "&lt;think&gt;"
            ).replace("</think>", "&lt;/think&gt;")
            yield partial_message


# initiate the Gradio app
chatbot = gr.ChatInterface(
    stream_response,
    textbox=gr.Textbox(
        placeholder="Send to the LLM...",
        container=False,
        autoscroll=True,
        scale=7,
    ),
    analytics_enabled=False,
    type="messages",
    save_history=True,
)

# launch the Gradio app
try:
    chatbot.queue().launch(
        share=False, server_name="labtr.taila54574.ts.net", debug=True
    )
except Exception as e:
    print(e)
