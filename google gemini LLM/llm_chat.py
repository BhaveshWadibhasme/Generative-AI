import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai

# Set up API key
os.environ["GEMINI_API_KEY"] = "use_your_api_keys"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model with safety settings and generation config
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    safety_settings=safety_settings,
    generation_config=generation_config,
)

def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_gpt_answer(context, question):
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(f"Context: {context}\n\nQuestion: {question}\n\nAnswer:")
    return response.text

# Streamlit UI
st.title("PDF Chatbot using Google Gemini Model")

# Upload PDF Section
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# Store uploaded PDFs in a temporary directory
pdf_dir = tempfile.mkdtemp()

pdf_paths = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        pdf_path = os.path.join(pdf_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_paths.append(pdf_path)

# Extract text from PDFs
if pdf_paths:
    st.success("PDFs uploaded successfully")
    pdf_text = extract_text_from_pdfs(pdf_paths)
else:
    pdf_text = ""

# Chat Section
if pdf_text:
    st.header("Chat with PDF Content")
    question = st.text_input("Ask a question about the PDFs")

    if question:
        answer = get_gpt_answer(pdf_text, question)
        st.write("Answer:", answer)
else:
    st.warning("Please upload PDF files to start the chat.")
