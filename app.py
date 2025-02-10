import dash
from dash import html, dcc, Input, Output, State, callback, ALL
import dash_bootstrap_components as dbc
from datetime import datetime
import os
import tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np
import wavio
import wave
import threading
import queue
import time
import uuid
import base64
import re
from dotenv import load_dotenv
from pypdf import PdfReader
import camelot  # New: using Camelot for PDF table extraction.
import pandas as pd

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PDFMinerLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from openai import OpenAI
from dash.dependencies import ClientsideFunction

# Load environment variables
load_dotenv()


# Enhanced Global Store
class GlobalStore:
    def __init__(self):
        self.processed_pdf = False
        self.vectorstore = None
        self.rag_chain = None
        self.openai_client = None
        self.current_chat_id = None
        self.chat_sessions = {}
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.temp_dir = tempfile.mkdtemp()

    def get_formatted_history(self, max_tokens=2000):
        """Get formatted chat history with token management."""
        if not self.current_chat_id:
            return ""

        history = self.chat_sessions.get(self.current_chat_id, {}).get('messages', [])
        formatted_history = []
        token_count = 0

        for msg in reversed(history[-10:]):  # Get last 10 messages, newest first
            msg_text = f"{msg['role']}: {msg['content']}"
            msg_tokens = len(msg_text) // 4  # Approximate token count

            if token_count + msg_tokens > max_tokens:
                break

            formatted_history.insert(0, msg_text)
            token_count += msg_tokens

        return "\n".join(formatted_history)

    def start_new_chat(self):
        new_chat_id = str(uuid.uuid4())
        self.current_chat_id = new_chat_id
        self.chat_sessions[new_chat_id] = {
            'messages': [],
            'title': 'New Chat',
            'timestamp': datetime.now()
        }
        return new_chat_id

    def add_message(self, role, content):
        if self.current_chat_id and self.current_chat_id in self.chat_sessions:
            self.chat_sessions[self.current_chat_id]['messages'].append({
                'role': role,
                'content': content
            })
            if role == 'user' and len(self.chat_sessions[self.current_chat_id]['messages']) == 1:
                self.chat_sessions[self.current_chat_id]['title'] = get_chat_title(content)


global_store = GlobalStore()

# Styles
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "300px",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "overflow-y": "auto",
    "z-index": 1
}

CHAT_TITLE_STYLE = {
    "padding": "10px",
    "margin-bottom": "5px",
    "border-radius": "5px",
    "cursor": "pointer",
    "transition": "background-color 0.2s",
    "white-space": "nowrap",
    "overflow": "hidden",
    "text-overflow": "ellipsis",
    "font-size": "0.9rem"
}

CONTENT_STYLE = {
    "margin-left": "320px",
    "padding": "2rem 1rem",
    "max-width": "calc(100% - 320px)"
}


# Enhanced PDF Processing Functions
def extract_tables_from_pdf(pdf_path):
    """Extract tables from PDF using Camelot."""
    tables = []
    try:
        # Use Camelot to extract tables from all pages with the 'stream' flavor.
        extracted_tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        for table in extracted_tables:
            if not table.df.empty:
                table_str = "TABLE_START\n"
                table_str += table.df.to_markdown(index=False)
                table_str += "\nTABLE_END"
                tables.append(table_str)
    except Exception as e:
        print(f"Camelot extraction error: {str(e)}")
    return tables


def preprocess_text(text):
    """Enhance text with special markers for numbers and data."""
    text = re.sub(r'(\d+\.?\d*)', r'NUM[\1]', text)
    text = re.sub(r'(\d+\.?\d*)\s*%', r'PERCENTAGE[\1]', text)
    text = re.sub(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', r'RANGE[\1 to \2]', text)
    text = re.sub(r'(\d+\.?\d*)\s*(kg|g|mg|L|ml|km|m|cm|mm)', r'MEASUREMENT[\1 \2]', text)
    return text


def enhanced_pdf_extraction(pdf_path):
    """Extract content from PDF using multiple methods."""
    content = []
    tables = extract_tables_from_pdf(pdf_path)

    try:
        loader = PDFMinerLoader(pdf_path)
        pages = loader.load()
        for page in pages:
            content.append(preprocess_text(page.page_content))
    except:
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text()
                content.append(preprocess_text(text))
        except Exception as e:
            print(f"PDF extraction error: {str(e)}")

    combined_content = []
    for i, text in enumerate(content):
        combined_content.append(text)
        if i < len(tables):
            combined_content.append(tables[i])

    return combined_content


# Enhanced Audio Functions

def start_recording(duration, filename):
    """Record audio with improved error handling using fixed duration."""
    sample_rate = 44100
    channels = 1
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}")
        audio_queue.put(indata.copy())

    try:
        with sd.InputStream(samplerate=sample_rate, channels=channels, callback=audio_callback):
            sd.sleep(int(duration * 1000))

        audio_data = []
        while not audio_queue.empty():
            audio_data.append(audio_queue.get())

        if audio_data:
            audio_data = np.vstack(audio_data)
            wavio.write(filename, audio_data, sample_rate, sampwidth=2)
            return True, "Recording completed successfully!"
    except Exception as e:
        return False, f"Error during recording: {str(e)}"


def record_until_silence(filename, silence_threshold=0.01, silence_duration=1.0,
                         sample_rate=44100, channels=1, max_duration=20):
    """
    Record audio until silence is detected or the maximum duration is reached.

    This function continuously listens and computes the root mean square (RMS)
    amplitude of each audio block. When the amplitude stays below the
    silence_threshold for at least silence_duration seconds, the recording stops.
    """
    audio_queue = queue.Queue()
    recorded_frames = []
    silence_start = None
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}")
        audio_queue.put(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=channels, callback=callback):
        while True:
            try:
                block = audio_queue.get(timeout=0.1)
            except queue.Empty:
                block = None

            if block is not None:
                recorded_frames.append(block)
                amplitude = np.sqrt(np.mean(block ** 2))
                if amplitude < silence_threshold:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= silence_duration:
                        break
                else:
                    silence_start = None

            if time.time() - start_time >= max_duration:
                break

    if recorded_frames:
        audio_data = np.concatenate(recorded_frames, axis=0)
        wavio.write(filename, audio_data, sample_rate, sampwidth=2)
        return True, "Recording completed successfully!"
    else:
        return False, "No audio recorded"


def transcribe_audio(filename, api_key):
    """Transcribe audio with OpenAI Whisper."""
    try:
        client = OpenAI(api_key=api_key)
        with open(filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return True, transcript.text
    except Exception as e:
        return False, f"Error during transcription: {str(e)}"


# Enhanced Chat Functions
def format_message(role, content):
    """Format chat messages with improved styling."""
    base_style = {
        "padding": "15px",
        "margin": "10px",
        "border-radius": "10px",
        "max-width": "80%",
        "white-space": "pre-wrap"
    }

    if role == "user":
        style = {
            **base_style,
            "background-color": "#e9ecef",
            "margin-left": "auto",
            "color": "black"
        }
    else:
        style = {
            **base_style,
            "background-color": "#ffffff",
            "color": "black"
        }

    return html.Div(content, style=style)


def get_chat_title(question):
    """Generate a chat title."""
    title = question[:40] + "..." if len(question) > 40 else question
    return title


def generate_sidebar_chat_history():
    """Generate sidebar chat history with improved UI."""
    chat_sessions = []
    for chat_id, session in sorted(
            global_store.chat_sessions.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
    ):
        chat_sessions.append(
            html.Div(
                [
                    html.I(className="fas fa-message me-2", style={"color": "#6c757d"}),
                    session['title']
                ],
                style={
                    **CHAT_TITLE_STYLE,
                    "background-color": "#ffffff" if chat_id != global_store.current_chat_id else "#e9ecef",
                    "border": "1px solid #dee2e6",
                    "color": "#212529",
                },
                id={'type': 'chat-session', 'index': chat_id},
                n_clicks=0
            )
        )
    return chat_sessions


# Enhanced QA System Initialization
def initialize_qa_system(contents):
    """Initialize QA system with improved context handling."""
    try:
        # Decode and save PDF
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        pdf_path = os.path.join(global_store.temp_dir, "uploaded.pdf")

        with open(pdf_path, 'wb') as f:
            f.write(decoded)

        # Enhanced PDF processing
        extracted_content = enhanced_pdf_extraction(pdf_path)

        # Improved text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\nTABLE_END\n", "\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

        # Create documents
        documents = []
        for content in extracted_content:
            splits = text_splitter.split_text(content)
            for split in splits:
                documents.append(Document(page_content=split))

        # Initialize embeddings and vectorstore
        embeddings = OpenAIEmbeddings(api_key=global_store.api_key)
        global_store.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings
        )

        # Enhanced prompt template
        prompt_template = ChatPromptTemplate.from_template("""
        You are a concise and precise assistant specialized in dairy farming. Your task is to analyze the provided document or data to answer inquiries about dairy cow management effectively, while maintaining strong topic continuity throughout the conversation.

        CONTEXT:
        Previous conversation thread: {chat_history}
        Current document context: {context}
        Current question: {input}

        CONVERSATION GUIDELINES:
        1. Always reference and build upon relevant points from previous messages
        2. If the current question seems disconnected, explicitly relate it back to previous topics
        3. Maintain the conversation's thematic flow by connecting new information to established context
        4. Acknowledge previous discussion points when introducing new, related information

        RESPONSE FORMAT:

        ANSWER:
        [Provide a direct, clear answer that:
        - Directly addresses the current question
        - References relevant parts of previous conversation
        - Connects to earlier discussed topics where applicable
        - Builds upon previously established context]

        KEY POINTS:
        [Include only if critical for understanding. Focus on points that:
        - Connect to previously discussed topics
        - Build upon earlier established information
        - Provide new, relevant context
        Examples:
        • Key fact about lactating cow dry matter intake that relates to previous discussion
        • Relevant nutritional guidelines that connect to earlier topics]

        FOLLOW-UP:
        [Conclude with a proactive question that:
        - Builds upon both the current answer and previous conversation threads
        - Helps maintain topic continuity
        - Encourages deeper exploration of related themes
        - Bridges current topic with previously discussed points]

        Example follow-up:
        "Could you specify your current feed regimen to help tailor these recommendations to your situation?"

        Remember to maintain conversation coherence while being succinct and focused in your explanations.
        """)

        # Initialize chat model with improved parameters
        llm = ChatOpenAI(
            api_key=global_store.api_key,
            temperature=0.1,
            max_tokens=1000
        )

        # Create document chain
        document_chain = create_stuff_documents_chain(
            llm,
            prompt_template,
            document_variable_name="context"
        )

        # Create retrieval chain with improved search
        retriever = global_store.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "lambda_mult": 0.7
            }
        )

        global_store.rag_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

        global_store.processed_pdf = True
        return True, "PDF processed successfully with enhanced context handling!"

    except Exception as e:
        return False, f"Error processing PDF: {str(e)}"


# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
)
server = app.server

# Layouts
sidebar = html.Div([
    html.H4("Configuration", className="mb-3"),
    dbc.Card([
        dbc.CardBody([
            dbc.Input(
                id="api-key-input",
                type="password",
                placeholder="Enter your OpenAI API Key",
                value=global_store.api_key,
                className="mb-2"
            ),
            html.Div(id="api-key-status", className="mb-3"),
        ])
    ], className="mb-4"),
    dbc.Button(
        [html.I(className="fas fa-plus me-2"), "New Chat"],
        id="new-chat-button",
        color="light",
        className="w-100 mb-3",
    ),
    html.H4("Chat History", className="mb-3"),
    html.Div(
        id="sidebar-chat-history",
        style={'height': 'calc(100vh - 300px)', 'overflow-y': 'auto'}
    )
], style=SIDEBAR_STYLE)

content = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(
                html.H1("Dairy Farm Chat System", className="text-center mb-4"),
                width=12
            ),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Upload Document"),
                    dbc.CardBody([
                        dcc.Upload(
                            id='upload-pdf',
                            children=html.Div([
                                html.I(className="fas fa-file-pdf me-2"),
                                'Drag and Drop or ',
                                html.A('Select PDF')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '10px',
                                'textAlign': 'center'
                            },
                            multiple=False
                        ),
                        html.Div(id="upload-status", className="mt-2")
                    ])
                ], className="mb-4"),
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Chat Interface"),
                    dbc.CardBody([
                        html.Div(
                            id="chat-history",
                            style={
                                'height': '400px',
                                'overflow-y': 'auto',
                                'margin-bottom': '20px',
                                'padding': '10px',
                                'border': '1px solid #dee2e6',
                                'border-radius': '10px'
                            }
                        ),
                        dbc.InputGroup([
                            dbc.Input(
                                id="question-input",
                                placeholder="Ask a question about dairy farming...",
                                type="text",
                                n_submit=0
                            ),
                            dbc.Button(
                                "Send",
                                id="send-button",
                                color="primary"
                            )
                        ], className="mb-3")
                    ])
                ], className="mb-4"),
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Voice Controls"),
                    dbc.CardBody([
                        dbc.RadioItems(
                            id="input-method",
                            options=[
                                {"label": "Text", "value": "text"},
                                {"label": "Voice", "value": "voice"}
                            ],
                            # Changed default value to "voice" so voice controls show by default.
                            value="voice",
                            inline=True,
                            className="mb-3"
                        ),
                        html.Div([
                            dcc.Slider(
                                id="recording-duration",
                                min=1,
                                max=10,
                                step=1,
                                value=5,
                                marks={i: f"{i}s" for i in range(1, 11)},
                                className="mb-3"
                            ),
                            dbc.Button(
                                "🎤 Start Recording",
                                id="record-button",
                                color="danger",
                                className="w-100"
                            )
                        ], id="voice-controls", style={'display': 'none'})
                    ])
                ])
            ], width=12)
        ])
    ], fluid=True, className="px-4")
], style=CONTENT_STYLE)

# Hidden dummy element for auto-scroll
dummy = html.Div(id="dummy", style={"display": "none"})

# Main app layout with dummy included
app.layout = html.Div([
    sidebar,
    content,
    dummy
], style={"overflow-x": "hidden"})

# Clientside callback for auto-scrolling chat history
app.clientside_callback(
    """
    function(children) {
        var chatDiv = document.getElementById("chat-history");
        if (chatDiv) {
            chatDiv.scrollTop = chatDiv.scrollHeight;
        }
        return "";
    }
    """,
    Output("dummy", "children"),
    Input("chat-history", "children")
)


@callback(
    Output("voice-controls", "style"),
    Input("input-method", "value")
)
def toggle_voice_controls(input_method):
    if input_method == "voice":
        return {"display": "block"}
    return {"display": "none"}


# Callbacks
@callback(
    [Output("chat-history", "children"),
     Output("sidebar-chat-history", "children")],
    [Input({"type": "chat-session", "index": ALL}, "n_clicks")],
    [State("chat-history", "children")]
)
def load_chat_session(n_clicks, current_chat):
    if not dash.callback_context.triggered:
        return current_chat or [], generate_sidebar_chat_history()

    triggered_id = dash.callback_context.triggered[0]["prop_id"]
    if not triggered_id:
        return current_chat or [], generate_sidebar_chat_history()

    chat_id = eval(triggered_id.split('.')[0])['index']
    global_store.current_chat_id = chat_id

    if chat_id in global_store.chat_sessions:
        messages = [
            format_message(msg['role'], msg['content'])
            for msg in global_store.chat_sessions[chat_id]['messages']
        ]
        return messages, generate_sidebar_chat_history()

    return current_chat or [], generate_sidebar_chat_history()


@callback(
    [Output("chat-history", "children", allow_duplicate=True),
     Output("sidebar-chat-history", "children", allow_duplicate=True),
     Output("question-input", "value")],
    [Input("send-button", "n_clicks"),
     Input("question-input", "n_submit"),
     Input("record-button", "n_clicks"),
     Input("new-chat-button", "n_clicks")],
    [State("question-input", "value"),
     State("chat-history", "children"),
     State("sidebar-chat-history", "children"),
     State("input-method", "value"),
     State("recording-duration", "value")],
    prevent_initial_call=True
)
def process_input(send_clicks, enter_pressed, record_clicks, new_chat_clicks,
                  question, current_chat, sidebar_chat, input_method, duration):
    if not dash.callback_context.triggered or not global_store.processed_pdf:
        return current_chat or [], sidebar_chat or [], question or ""

    triggered_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "new-chat-button":
        global_store.start_new_chat()
        return [], generate_sidebar_chat_history(), ""

    if triggered_id == "record-button" and input_method == "voice":
        audio_file = os.path.join(global_store.temp_dir, f"recording_{int(time.time())}.wav")
        # Record until silence is detected instead of a fixed duration.
        success, message = record_until_silence(audio_file)
        if not success:
            return (
                current_chat + [format_message("assistant", f"Recording failed: {message}")],
                sidebar_chat,
                ""
            )

        success, transcribed_text = transcribe_audio(audio_file, global_store.api_key)
        try:
            os.remove(audio_file)
        except:
            pass

        if not success:
            return (
                current_chat + [format_message("assistant", f"Transcription failed: {transcribed_text}")],
                sidebar_chat,
                ""
            )

        question = transcribed_text

    if triggered_id in ["send-button", "question-input", "record-button"] and question:
        try:
            if global_store.current_chat_id is None:
                global_store.start_new_chat()

            chat_history = global_store.get_formatted_history()

            response = global_store.rag_chain.invoke({
                "input": question,
                "chat_history": chat_history
            })

            answer = response.get("answer", "I couldn't find a relevant answer in the document.")

            global_store.add_message("user", question)
            global_store.add_message("assistant", answer)

            messages = []
            if current_chat:
                messages = current_chat if isinstance(current_chat, list) else [current_chat]

            messages.extend([
                format_message("user", question),
                format_message("assistant", answer)
            ])

            return messages, generate_sidebar_chat_history(), ""

        except Exception as e:
            error_message = f"Error: {str(e)}"
            return (
                current_chat + [format_message("assistant", error_message)],
                sidebar_chat,
                ""
            )

    return current_chat or [], sidebar_chat or [], question or ""


@callback(
    Output("api-key-status", "children"),
    Input("api-key-input", "value")
)
def update_api_key(api_key):
    if api_key:
        global_store.api_key = api_key
        return html.Div("API Key set successfully!", style={"color": "green"})
    return html.Div("Please enter your API Key", style={"color": "red"})


@callback(
    Output("upload-status", "children"),
    Input("upload-pdf", "contents"),
    State("upload-pdf", "filename")
)
def process_upload(contents, filename):
    if contents is None:
        return ""
    if not filename.lower().endswith('.pdf'):
        return html.Div("Please upload a PDF file", style={"color": "red"})

    success, message = initialize_qa_system(contents)
    color = "green" if success else "red"
    return html.Div(message, style={"color": color})


if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
