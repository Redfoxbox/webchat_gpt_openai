from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext, StorageContext, load_index_from_storage
from langchain.chat_models import ChatOpenAI
import gradio as gr
import os
from loguru import logger
from pydantic import BaseModel

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(
        max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(
        temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        prompt_helper=prompt_helper
    )
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context)

    return index


class Message(BaseModel):
    role: str
    content: str

def clear():
    return ""


async def predict(input, history):

    history.append(Message(role="user", content=input))
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    logger.info("Query:"+input)
    response = query_engine.query(input)
    logger.info(response)
    history.append(Message(role="assistant", content=response.response))
    messages = [(history[i].content, history[i+1].content)
                for i in range(0, len(history)-1, 2)]
    return messages, history


with gr.Blocks(title="Webchat") as demo:
    logger.info("Starting Demo...")
    gr.Markdown(
        """
    #Webchat!
    Egyedi adatokat használó OpenAI gpt-3.5-turbo
    """)
    chatbot = gr.Chatbot(label="WebGPT")
    state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Írja be a kérdését, és nyomja meg az enter billentyűt").style(
            container=False)
    txt.submit(predict, [txt, state], [chatbot, state])
    txt.submit(clear, None, [txt], queue=False)


if len(os.listdir('./storage')) == 0:
    logger.info("Storage folder is empty, start construct index")
    index = construct_index("docs")
    logger.info("Persist index")
    index.storage_context.persist(persist_dir="./storage")
else:
    logger.info("Try to load index")
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    logger.info("Index loaded")


logger.info("Starting webserver " +
            os.environ["SERVER_HOST"]+":"+os.environ["SERVER_PORT"])
demo.launch(share=False, debug=True,
            server_name=os.environ["SERVER_HOST"], server_port=int(os.environ["SERVER_PORT"]), auth=(os.environ["USER"], os.environ["PASSWORD"]))
