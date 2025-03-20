import streamlit as st
from transformers import pipeline
import time

# Page configuration
st.set_page_config(
    page_title=" Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# Debugging Check
st.write("If you see this, Streamlit is working!")

# Custom CSS for UI Styling
st.markdown("""
<style>
    .main {
        background-image: linear-gradient(to bottom right, #1e3a8a, #7e22ce);
        padding: 0;
        margin: 0;
    }
    .stApp {
        background-image: linear-gradient(to bottom right, rgba(30, 58, 138, 0.7), rgba(109, 40, 217, 0.7));
    }
    div[data-testid="stHeader"] {
        display: none;
    }
    .title {
        color: red;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        font-family: 'Inter', sans-serif;
    }
    .subtitle {
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        font-size: 1.25rem;
        margin-bottom: 2rem;
        font-family: 'Inter', sans-serif;
    }
    .card {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .button-container {
        display: flex;
        justify-content: center;
    }
    .stButton>button {
        background: linear-gradient(to right, #7e22ce, #6d28d9);
        color: black;
        border: none;
        border-radius: 0.375rem;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    .summary-box {
        background-color: black;
        border-radius: 0.375rem;
        padding: 1rem;
        border: 1px solid #e5e7eb;
    }
    .footer {
        color: rgba(255, 255, 255, 0.6);
        text-align: center;
        font-size: 0.875rem;
        margin-top: 4rem;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='title'> Text Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Transform lengthy content into concise summaries using deep learning</p>", unsafe_allow_html=True)

# Function to load the summarization model
@st.cache_resource
def load_summarizer():
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Create a card container
st.markdown("<div class='card'>", unsafe_allow_html=True)

# Text input section
st.markdown("<label style='font-weight: 500; color: #374151; margin-bottom: 0.5rem;'>Your Text</label>", unsafe_allow_html=True)
text_input = st.text_area("", placeholder="Paste your article, document, or long text here...", height=200, label_visibility="collapsed")

# Initialize session state
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'loading_model' not in st.session_state:
    st.session_state.loading_model = False

# Debugging: Check session state
st.write(f"Model Loaded: {st.session_state.model_loaded}")
st.write(f"Summary Exists: {bool(st.session_state.summary)}")

# Load model button
if not st.session_state.model_loaded and not st.session_state.loading_model:
    if st.button("Load AI Model"):
        with st.spinner("Loading AI model..."):
            st.session_state.loading_model = True
            summarizer = load_summarizer()
            if summarizer:
                st.session_state.summarizer = summarizer
                st.session_state.model_loaded = True
                st.session_state.loading_model = False
                st.success("AI model loaded successfully!")
                time.sleep(1)
                st.rerun()

# Summarize button (only show if model is loaded)
if st.session_state.model_loaded:
    st.markdown("<div class='button-container'>", unsafe_allow_html=True)
    summarize_button = st.button("Summarize Text", disabled=not text_input.strip())
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Process text when summarize button is clicked
    if summarize_button and text_input.strip():
        with st.spinner("Generating summary..."):
            try:
                result = st.session_state.summarizer(text_input, max_length=150, min_length=30)
                st.session_state.summary = result[0]['summary_text']
                st.success("Summary generated successfully!")
            except Exception as e:
                st.error(f"Error during summarization: {e}")
                st.info("Try using a shorter text or different content.")

# Display summary if available
if st.session_state.summary:
    st.markdown("<label style='font-weight: 500; color: black; margin-bottom: 0.5rem; margin-top: 1.5rem;'>Summary</label>", unsafe_allow_html=True)
    st.markdown(f"<div class='summary-box'>{st.session_state.summary}</div>", unsafe_allow_html=True)

# Close card container
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<p class='footer'></p>", unsafe_allow_html=True)