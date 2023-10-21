from llama_index import SimpleDirectoryReader, Document, ServiceContext, PromptHelper, LLMPredictor, VectorStoreIndex, StorageContext, GPTVectorStoreIndex, StorageContext, load_index_from_storage, set_global_service_context
from langchain.chat_models import ChatOpenAI
import gradio as gr
from gradio import Interface, Textbox, Dropdown, HTML, File, Checkbox, Button

import sys
import os
import json
import time
from llama_index.llms import OpenAI
import openai
from langchain.prompts import ChatPromptTemplate
from llama_index.llms.base import ChatMessage, ChatResponse, MessageRole
from transformers import pipeline
import numpy as np
from gtts import gTTS

import numpy as np
import tempfile

import uuid
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import os
from gtts import gTTS
from io import BytesIO
from fastapi import FastAPI
import uvicorn

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

MAX_RETRIES = 10
service_context = None    
storage_context = StorageContext.from_defaults(persist_dir='./storage') 
index = load_index_from_storage(storage_context=storage_context) # load index
uploaded_index = None
chat_history_global = []


# for pop up messages
def info_fn(message): 
    return gr.Info(message)


def load_documents_from_directory(directory='docs'):
    documents = []
    file_names = os.listdir(directory)
    
    for file_name in file_names:
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r') as file:
            data = json.load(file)
            for url, content in data.items():
                document = Document(
                    text=content,
                    metadata={"file_name": file_name, "url": url}
                )
                documents.append(document)
    return documents


# when user uploads their own files
def load_documents_from_uploaded_files(uploaded_files):
    documents = []
    if isinstance(uploaded_files, list):
        files_to_process = uploaded_files
        print(f"Number of uploaded files: {len(uploaded_files)}")
    else:
        files_to_process = [uploaded_files]
        print("Processing a single uploaded file.")

    for uploaded_file in files_to_process:
        file_ext = os.path.splitext(uploaded_file.name)[1]

        if file_ext == ".json":
            with open(uploaded_file.name, 'r') as file:
                data = json.load(file)
                for url, content in data.items():
                    document = Document(
                        text=content,
                        metadata={"file_name": uploaded_file.name, "url": url}
                    )
                    documents.append(document)
        elif file_ext == ".txt":
            with open(uploaded_file.name, 'r') as file:
                content = file.read()
                document = Document(
                    text=content,
                    metadata={"file_name": uploaded_file.name}
                )
                documents.append(document)
    for document in documents:
        print(f"Document text: {document.text}") #debugging line 
   
    return documents




def construct_index():
    global index
    documents = load_documents_from_directory()
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    print("Index created")
    index.storage_context.persist()
    print("Persist called")
    return index


# chat prompt template provides the chatbot with context 
def format_chatbot_prompt(agent_message, input_text):
    message_tuples = [
        (MessageRole.SYSTEM.value, agent_message),
        (MessageRole.USER.value, "Respond to the following question: {question}")
    ]
    chat_template = ChatPromptTemplate.from_messages(message_tuples)
    return chat_template.format(question=input_text)



def process_query_results(results):
    response_text = results.response
    if isinstance(results.metadata, dict):
        for uuid in results.metadata.keys():
            if "url" in results.metadata[uuid]:
                url = results.metadata[uuid]["url"]
                print(f"UUID: {uuid}, URL: {url}")
    return response_text




def handle_file_upload(uploaded_files=None):
    global uploaded_index  

    if uploaded_files:
        documents = load_documents_from_uploaded_files(uploaded_files)
        uploaded_index = VectorStoreIndex.from_documents(documents, service_context=service_context) # creating a temp index to swap with the default
        print("Uploaded index created with new documents")
        return info_fn("Document uploaded and chatbot retrained successfully!")
    else:
        print("No files uploaded")
        return info_fn("Failed to upload document. Please try again.")
    

def chatbot(agent_message, llm_choice, input_text="", response_mode="respond by text"):
    global index, uploaded_index  

    llm = OpenAI(model=llm_choice, temperature=0.3)
    service_context = ServiceContext.from_defaults(llm=llm)

    active_index = uploaded_index if uploaded_index else index
    print(f"Active Index: {active_index}")  # Debug print

    # Prepare the prompt
    prompt = format_chatbot_prompt(agent_message, input_text) 

    # Create the query engine before any conditional branches
    if response_mode == "respond by text":
        query_engine = active_index.as_query_engine(streaming=True)
    else:
        query_engine = active_index.as_query_engine(streaming=False)

    results = query_engine.query(prompt)

    if response_mode == "respond by text":
        bot_message = ""
        for text in results.response_gen:
            bot_message += text
            yield bot_message  # This is streaming the message parts as they come
            time.sleep(0.1)
    else:
        # For non-streaming modes, we only want to yield once with the complete message.
        try:
            final_bot_message = process_query_results(results)
            print(f"Bot message: {final_bot_message}")  # Debug print
        except (AttributeError, IndexError, KeyError) as e:
            print(f"Exception caught: {type(e).__name__}, {str(e)}")
            final_bot_message = "An error occurred."

        # Yield the final message (only once)
        yield final_bot_message


def speak_text(text):

    print(f"Text received by speak_text: {text}")

    # Create a buffer
    mp3_fp = BytesIO()
    
    tts = gTTS(text=text, lang='en', slow=False)

    # Save to buffer
    tts.write_to_fp(mp3_fp)
    
    # Get buffer contents (bytes) and reset pointer to the beginning
    mp3_fp.seek(0)
    audio_data = mp3_fp.read()

    return audio_data

        
def respond(user_text, agent_message, llm_choice, response_mode, chat_history):
    print(f"Received user_text: {user_text}")  # Debug print

    if isinstance(user_text, list) and user_text:
        user_text = user_text[0]
        print(f"Extracted user_text: {user_text}")  # Debug print

    if chat_history is None:
        chat_history = []

    if not llm_choice or llm_choice == "":
        bot_message = "Please choose an LLM model before sending a message."
        chat_history.append((str(user_text), bot_message))
        return "", chat_history, None

    # initiate chat_history with the user's message first
    chat_history.append((str(user_text), ""))

    # Call the chatbot function and process the response
    for bot_message in chatbot(agent_message, llm_choice, user_text, response_mode):
        print(f"Received bot_message: {bot_message}")  # Debug print

        if response_mode == "respond by text":
            # Update the last chat history entry with the bot's response
            chat_history[-1] = (chat_history[-1][0], bot_message)
            yield "", chat_history, None

        elif response_mode == "respond by text and voice":
            
            audio_data = speak_text(bot_message)  # Get the audio data for the bot's message
            # Update the last chat history entry with the bot's response
            chat_history[-1] = (chat_history[-1][0], bot_message)
            yield "", chat_history, audio_data







transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
# function to convert our voice input into text using whisper 
def transcribe(stream, new_chunk):
    sr, y = new_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y
    return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]




def process_input(agent_message, llm_choice, text_input, audio_chunk):
    global chat_history_global 

    if text_input:  # If there's text input use it
        responses = list(respond(text_input, agent_message, llm_choice, chat_history_global))
        _, chat_history_global = responses[-1]
    elif audio_chunk:  # Else if no text then voice message
        _, transcribed_text = transcribe(None, audio_chunk)
        responses = list(respond(transcribed_text, agent_message, llm_choice, chat_history_global))
        _, chat_history_global = responses[-1]
    
    return chat_history_global



def handle_voice_input(agent_message, llm_choice, response_mode, voice_chunk, chatbot_component):
    _, transcribed_text = transcribe(None, voice_chunk)

    responses = respond(transcribed_text, agent_message, llm_choice, response_mode, chatbot_component)

    # Yield each result as it comes
    for _, chat_history, audio_data in responses:
        yield "", chat_history, audio_data






def create_gradio_app():
    global index
    global service_context

    with gr.Blocks() as app:
        gr.Markdown("""
        <center>

        # Tax Strategists of America Chatbot

        </center>
        """)

        with gr.Tab("Configuration"):
            agent_message_input = gr.Textbox(
                label="Who Am I?", value="You are a customer service agent for the company Tax Strategists of America"
            )
            llm_choice_input = gr.Dropdown(
                choices=["gpt-4", "gpt-3.5-turbo", "text-davinci-003"], label="Choose LLM Model"
            )
            uploaded_files_input = gr.File(
                label="Upload your documents (JSON or TXT format)"
            )
            process_files_button = gr.Button(value="Retrain me on your own data")
            response_mode = gr.Radio(
                choices=["respond by text", "respond by text and voice"],
                label="Chatbot Response Mode",
                default="respond by text"
            )

        with gr.Tab("Chatbot") as chat_tab:
            msg_input = gr.Textbox(
                label="Type here",
                placeholder="Input a message then press enter"
            )
            voice_input = gr.Audio(source="microphone", label="Speak here")
            process_speech_button = gr.Button(value="Process Speech")
            chatbot_component = gr.Chatbot(
                [],
                elem_id="chatbot",
                bubble_full_width=False,
                avatar_images= (None, os.path.join(os.path.dirname(__file__), "seamgen-chatbot-icon.png"))
            )
            audio_component = gr.Audio(autoplay=True, label="Chatbot Voice Response")

        process_files_button.click(handle_file_upload, inputs=[uploaded_files_input])
        msg_input.submit(
            respond,
            [
                msg_input,
                agent_message_input,
                llm_choice_input,
                response_mode,
                chatbot_component
            ],
            [
                msg_input,
                chatbot_component,
                audio_component  
            ]
        )
        process_speech_button.click(
            handle_voice_input,
            inputs=[agent_message_input, llm_choice_input, response_mode, voice_input, chatbot_component],
            outputs=[msg_input, chatbot_component, audio_component]  
        )


        return app


app = FastAPI()

@app.get("/")
def read_main():
    return {"message": "This is your main app"}

# Create and mount the Gradio interface
gradio_interface = create_gradio_app()
gr.mount_gradio_app(app, gradio_interface, path="/gradio")

# The main check is used to run the server when the script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 