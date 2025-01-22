from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
import numpy as np
import pickle
from FeatureExtraction import FeatureExtraction
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from langchain_utils import summarize_with_langchain, qa_with_langchain  # Import LangChain functions
from urllib.parse import urlparse

try:
    from langchain_community.document_loaders import UnstructuredFileLoader
except ImportError:
    raise ImportError("The 'unstructured' package is not found. Please install it with 'pip install unstructured'.")

# Set Streamlit page configuration
st.set_page_config(page_title="PhishNet AI", layout="wide")

# Load environment variables
load_dotenv()

genai.configure(api_key="AIzaSyA7Q2_eC2-RXD9sG1rZjMl2FqE0eMQwkB0")  # Your provided API key

# Function to load Gemini Pro model and get responses
@st.cache_resource
def initialize_genai_model():
    model = genai.GenerativeModel("gemini-pro")
    return model.start_chat(history=[])

chat = initialize_genai_model()

def get_gemini_response(question):
    try:
        response = chat.send_message(question, stream=True)
        if not response:
            raise ValueError("No response received from the model.")
        return ''.join([chunk.text for chunk in response if hasattr(chunk, 'text')])
    except Exception as e:
        # Attempt to rewind and retry once
        try:
            chat.rewind()
            response = chat.send_message(question, stream=True)
            if not response:
                raise ValueError("No response received from the model.")
            return ''.join([chunk.text for chunk in response if hasattr(chunk, 'text')])
        except Exception as retry_e:
            # Handle safety ratings and no valid text
            if hasattr(retry_e, 'safety_ratings'):
                return "The response was flagged for safety concerns and could not be processed."
            return f"An error occurred: {retry_e}"

# Train or Load the Phishing Detection Model
@st.cache_resource
def train_and_save_model(uploaded_file=None):
    dataset_path = 'C:\\PHISH-NET-AI-\\phishing.csv'
    try:
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_csv(dataset_path)
    except Exception as e:
        st.error(f"An error occurred while reading the dataset: {e}")
        raise

    # Separate features and labels
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Train a Gradient Boosting Classifier
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    # Save the trained model
    with open('C:\\PHISH-NET-AI-\\model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    st.success("Model trained and saved successfully.")
    return model

# Load existing model or train a new one if not found
def load_or_train_model():
    model_path = 'C:\\PHISH-NET-AI-\\model.pkl'
    try:
        with open(model_path, 'rb') as model_file:
            return pickle.load(model_file)
    except (FileNotFoundError, EOFError, ValueError):
        st.warning("Model not found or incompatible. Training a new model.")
        return train_and_save_model()

phishing_model = load_or_train_model()

# Function to check if a URL is phishing or safe
def check_phishing(url):
    try:
        # Parse the domain from the URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Check if the URL is one of the official URLs
        official_urls = ["google.com", "www.google.com", "linkedin.com", "www.linkedin.com"]
        if domain in official_urls:
            return {
                "is_safe": True,
                "prob_safe": 100,
                "prob_phishing": 0
            }

        # Extract features from the URL
        feature_extractor = FeatureExtraction(url)
        features = np.array(feature_extractor.getFeaturesList()).reshape(1, -1)

        # Adjust features dynamically to match model input size
        required_features = phishing_model.n_features_in_
        if features.shape[1] < required_features:
            features = np.pad(features, ((0, 0), (0, required_features - features.shape[1])), constant_values=0)
        elif features.shape[1] > required_features:
            features = features[:, :required_features]

        # Model prediction
        prediction = phishing_model.predict(features)[0]
        proba_safe = phishing_model.predict_proba(features)[0, 0]
        proba_phishing = phishing_model.predict_proba(features)[0, 1]

        is_safe = proba_safe > 0.6

        return {
            "is_safe": is_safe,
            "prob_safe": round(proba_safe * 100, 2),
            "prob_phishing": round(proba_phishing * 100, 2),
        }
    except Exception as e:
        return {"error": str(e)}

def classify_phishing_reason(url):
    try:
        question = f"Why is the URL '{url}' considered unsafe?"
        response = get_gemini_response(question)
        return response
    except Exception as e:
        return f"An error occurred while classifying the reason: {e}"

# Add custom CSS for enhanced UI
def add_custom_css():
    st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #1e1e2f, #121212);
        color: #fff;
        font-family: 'Arial', sans-serif;
    }
    .main {
        background: #222;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        padding: 10px;
    }
    .stAlert {
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

add_custom_css()

# Streamlit UI setup
st.title("💬 PhishNetUI - Chatbot with Phishing Detector")
st.caption("🚀 A Streamlit chatbot powered by Google Gemini AI")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = get_gemini_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

# Tabs for organization
tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "🔍 Phishing Detection", "📄 LangChain Features"])

# Chatbot Tab
with tab1:
    st.subheader("Chatbot Section")
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    col1, col2 = st.columns([8, 2])

    with col1:
        input_text = st.text_input("Ask the chatbot something:", key="chat_input")

    with col2:
        file_uploaded = st.file_uploader("", key="document_uploader")
        
        if file_uploaded:
            st.success("Document uploaded successfully!")

    submit_chat = st.button("Ask")

    if submit_chat and input_text:
        response = get_gemini_response(input_text)
        
        st.session_state['chat_history'].append(("You", input_text))
        st.session_state['chat_history'].append(("Bot", response))

    if file_uploaded:
        try:
            summary = summarize_with_langchain(file_uploaded)
            if summary:
                st.subheader("Document Summary")
                st.write(summary)
                
                question = st.text_input("Ask a question about the document:")
                if st.button("Get Answer"):
                    answer = qa_with_langchain(question, summary)
                    if answer:
                        st.subheader("Answer")
                        st.write(answer)
                    else:
                        st.warning("Please upload a document first.")
                    
        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")

    st.subheader("Chat History")
    
    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")

# Phishing Detection Tab
with tab2:
    st.subheader("Phishing Detection Section")
    
    url_input = st.text_input("Enter a URL to check if it is safe:", key="phishing_url_input")
    
    submit_url = st.button("Check URL")

    if submit_url and url_input:
        
        result = check_phishing(url_input)
        
        if "error" in result:
            st.error(f"An error occurred: {result['error']}")
            
        elif result["is_safe"]:
            st.success(f"The URL is safe ({result['prob_safe']}% confidence).")
            
        else:
            st.error(f"The URL is phishing ({result['prob_phishing']}% confidence).")
            reason = classify_phishing_reason(url_input)
            st.info(f"Reason: {reason}")

# LangChain Features Tab
with tab3:
    
    st.subheader("LangChain Features")
    
    file_upload = st.file_uploader("Upload a document for summarization or QA:")
    
    if file_upload:
        
        summary = summarize_with_langchain(file_upload)
        
        if summary:
            st.subheader("Summary")
            st.write(summary)

            question = st.text_input("Ask a question about the document:")
            
            if st.button("Get Answer"):
                answer = qa_with_langchain(question, summary)
                
                if answer:
                    st.subheader("Answer")
                    st.write(answer)
                else:
                    st.warning("Please upload a document first.")
