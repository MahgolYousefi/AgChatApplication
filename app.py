# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import dash
from dash import html, dcc, Input, Output, State, callback, ALL
import dash_bootstrap_components as dbc
from datetime import datetime
import os
import chromadb
from chromadb.config import Settings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from openai import OpenAI
import base64
import tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np
import wavio
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Initialize the Dash app with bootstrap
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)
server = app.server
# Initialize global variables
class GlobalStore:
    def __init__(self):
        self.processed_pdf = False
        self.vectorstore = None
        self.rag_chain = None
        self.openai_client = None
        self.current_chat_id = None
        self.chat_sessions = {}  # Store multiple chat sessions
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.temp_dir = tempfile.mkdtemp()

    def start_new_chat(self):
        # Generate new chat ID
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
            # Update chat title with first question if it's a user message
            if role == 'user' and len(self.chat_sessions[self.current_chat_id]['messages']) == 1:
                self.chat_sessions[self.current_chat_id]['title'] = get_chat_title(content)

    def get_current_chat_history(self):
        if self.current_chat_id and self.current_chat_id in self.chat_sessions:
            return self.chat_sessions[self.current_chat_id]['messages']
        return []

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

def format_message(role, content):
    """Format chat messages for display with enhanced styling"""
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
            "background-color": "#ffffff",  # Change to white background
            "color": "black"  # Change text color to black for better readability
        }

    return html.Div(content, style=style)

def get_chat_title(question):
    """Generate a title from the question"""
    title = question[:40] + "..." if len(question) > 40 else question
    return title

def generate_sidebar_chat_history():
    """Generate sidebar chat history display with clickable items"""
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

def initialize_qa_system(contents):
    """Initialize the QA system with uploaded PDF and enhanced prompt"""
    try:
        # Decode and save PDF temporarily
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        pdf_path = os.path.join(global_store.temp_dir, "uploaded.pdf")

        with open(pdf_path, 'wb') as f:
            f.write(decoded)

        # Load and process PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(pages)

        # Create and store embeddings
        embeddings = OpenAIEmbeddings(api_key=global_store.api_key)
        global_store.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )

        # Initialize chat model
        llm = ChatOpenAI(
            api_key=global_store.api_key,
            temperature=0.3,
            max_tokens=500
        )

        # Create prompt template
        prompt_template = ChatPromptTemplate.from_template("""
        You are a concise and precise assistant specialized in dairy farming. Your task is to analyze the provided document or data to answer inquiries about dairy cow management effectively.

        Previous conversation: {chat_history}
        Current question: {input}
        Document context: {context}

        Respond in this format:

        ANSWER:
        [Provide a direct, clear answer based on the document's information or standard dairy farming practices. Make this the first element of the response.]

        KEY POINTS:
        [Include only if they are critical for understanding the answer. Use bullet points for clarity and brevity.]
        • [Key fact about lactating cow dry matter intake]
        • [Relevant nutritional guidelines or requirements]

        [Conclude with a proactive, engaging question that seeks further input from the user to refine the discussion or address additional specifics of their query. This question should naturally follow from the previous parts of the conversation and be framed to solicit necessary details or preferences from the user.]

        For example:
        "Could you specify the current feed regimen for your cows to better tailor recommendations to your specific situation?"

        Ensure all responses are succinct, avoiding unnecessary explanations to maintain focus and efficiency.
        """)

        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt_template)

        # Create retrieval chain
        retriever = global_store.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        global_store.rag_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

        global_store.processed_pdf = True
        return True, "PDF processed successfully!"

    except Exception as e:
        return False, f"Error processing PDF: {str(e)}"# Sidebar layout
sidebar = html.Div(
    [
        html.H4("Configuration", className="mb-3"),
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        dbc.Input(
                            id="api-key-input",
                            type="password",
                            placeholder="Enter your OpenAI API Key",
                            value=global_store.api_key,
                            className="mb-2"
                        ),
                        html.Div(id="api-key-status", className="mb-3"),
                    ]
                )
            ],
            className="mb-4"
        ),
        dbc.Button(
            [html.I(className="fas fa-plus me-2"), "New Chat"],
            id="new-chat-button",
            color="light",
            className="w-100 mb-3",
        ),
        html.H4("Chat History", className="mb-3"),
        html.Div(
            id="sidebar-chat-history",
            style={
                'height': 'calc(100vh - 300px)',
                'overflow-y': 'auto'
            }
        )
    ],
    style=SIDEBAR_STYLE
)

# Content layout
content = html.Div(
    [
        dbc.Container(
            [
                html.H1("PDF Chat System", className="text-center mb-4"),
                dbc.Row(
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Upload Document"),
                                    dbc.CardBody(
                                        [
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
                                        ]
                                    )
                                ],
                                className="mb-4"
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader("Chat Interface"),
                                    dbc.CardBody(
                                        [
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
                                            dbc.InputGroup(
                                                [
                                                    dbc.Input(
                                                        id="question-input",
                                                        placeholder="Ask a question about your PDF...",
                                                        type="text",
                                                        n_submit=0
                                                    ),
                                                    dbc.Button(
                                                        "Send",
                                                        id="send-button",
                                                        color="primary"
                                                    )
                                                ],
                                                className="mb-3"
                                            )
                                        ]
                                    )
                                ],
                                className="mb-4"
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader("Voice Controls"),
                                    dbc.CardBody(
                                        [
                                            dbc.RadioItems(
                                                id="input-method",
                                                options=[
                                                    {"label": "Text", "value": "text"},
                                                    {"label": "Voice", "value": "voice"}
                                                ],
                                                value="text",
                                                inline=True,
                                                className="mb-3"
                                            ),
                                            html.Div(
                                                id="voice-controls",
                                                style={'display': 'none'},
                                                children=[
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
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )
                        ],
                        width=12
                    )
                )
            ],
            fluid=True,
            className="px-4"
        )
    ],
    style=CONTENT_STYLE
)

# Main app layout
app.layout = html.Div(
    [
        sidebar,
        content
    ],
    style={"overflow-x": "hidden"}
)# Callbacks
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

    # Extract chat_id from the triggered component
    chat_id = eval(triggered_id.split('.')[0])['index']

    # Update current chat session
    global_store.current_chat_id = chat_id

    # Get messages from the selected chat session
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

    # Handle New Chat button click
    if triggered_id == "new-chat-button":
        global_store.start_new_chat()
        return [], generate_sidebar_chat_history(), ""

    if triggered_id == "record-button":
        return current_chat or [], sidebar_chat or [], question or ""

    # Process input for both Send button and Enter key
    if triggered_id in ["send-button", "question-input"] and question:
        try:
            # Create new chat session if none exists
            if global_store.current_chat_id is None:
                global_store.start_new_chat()

            # Get chat history for context
            chat_history = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in global_store.get_current_chat_history()[-5:]
            ])

            response = global_store.rag_chain.invoke({
                "input": question,
                "chat_history": chat_history
            })

            answer = response.get("answer", "I couldn't find a relevant answer in the document.")

            # Add messages to current chat session
            global_store.add_message("user", question)
            global_store.add_message("assistant", answer)

            # Update main chat display
            messages = []
            if current_chat:
                messages = current_chat if isinstance(current_chat, list) else [current_chat]

            messages.extend([
                format_message("user", question),
                format_message("assistant", answer)
            ])

            # Update sidebar with all chat sessions
            sidebar_messages = generate_sidebar_chat_history()

            return messages, sidebar_messages, ""

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

@callback(
    Output("voice-controls", "style"),
    Input("input-method", "value")
)
def toggle_voice_controls(input_method):
    if input_method == "voice":
        return {"display": "block"}
    return {"display": "none"}

if __name__ == '__main__':
    app.run_server(debug=True)
    
