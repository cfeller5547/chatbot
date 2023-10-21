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
import soundfile as sf
import sounddevice as sd
import numpy as np
import tempfile

import uuid
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import os
from gtts import gTTS

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

    if response_mode == "respond by text":
        query_engine = active_index.as_query_engine(streaming=True)
        prompt = format_chatbot_prompt(agent_message, input_text)
        results = query_engine.query(prompt)

        bot_message = ""
        for text in results.response_gen:
            bot_message += text
            yield bot_message
            time.sleep(0.1)
    else:
        query_engine = active_index.as_query_engine(streaming=False)
        prompt = format_chatbot_prompt(agent_message, input_text)
        results = query_engine.query(prompt)
        try:
            bot_message = process_query_results(results)
            print(f"Bot message: {bot_message}")  # Debug print
            yield bot_message
        except (AttributeError, IndexError, KeyError) as e:
            print(f"Exception caught: {type(e).__name__}, {str(e)}")
            yield "An error occurred."







def speak_text(text):
    print(f"Text received by speak_text: {text}") 
    tts = gTTS(text=text, lang='en', slow=False)
    # Create a temporary file to save the audio
    temp_fd, temp_filename = tempfile.mkstemp(suffix=".mp3")
    os.close(temp_fd)  # Close the file descriptor, as it's not needed
    try:
        tts.save(temp_filename)
        # Read the audio data from the temporary file
        with open(temp_filename, 'rb') as temp_mp3_file:
            audio_data = temp_mp3_file.read()
        return audio_data
    finally:
        # Ensure the temporary file is deleted
        os.remove(temp_filename)
        
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
    if response_mode == "respond by text":
        chat_history.append((str(user_text), ""))  
        for bot_message in chatbot(agent_message, llm_choice, user_text, response_mode):
            chat_history[-1] = (chat_history[-1][0], bot_message)  # Replace the last tuple with a new tuple
            yield "", chat_history, None

    else:
        chat_history.append((str(user_text), ""))
        for bot_message in chatbot(agent_message, llm_choice, user_text, response_mode):
            print(f"Appending to chat history: User: {str(user_text)}, Bot: {bot_message}")
            chat_history.append((str(user_text), bot_message))

            audio_data = None
            if response_mode == "respond by text and voice":
                audio_data = speak_text(bot_message)
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






def main():
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
                avatar_images=(None, os.path.join(os.path.dirname(__file__), "seamgen-chatbot-icon.png"))
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


    app.queue().launch(share=True)

if __name__ == "__main__":
    main()

