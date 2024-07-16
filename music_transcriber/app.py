import streamlit as st
import os
from werkzeug.utils import secure_filename
from MusicAssist import process_music  # Import the main function from MusicAssist.py
import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up upload and output folders relative to the script location
base_dir = os.path.dirname(os.path.abspath(__file__))
upload_folder = os.path.join(base_dir, 'inputs')
output_folder = os.path.join(base_dir, 'output')
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Streamlit App
st.title("Music Transcriber")
st.subheader("Upload an audio file to transcribe it into instrument-separated MIDI and .mp3 files.")

uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'mp4', 'flac'])
st.write("")
st.subheader("How it works and usage tips")
st.write("Music Transcriber first extracts any piano, voice, drums and/or bass from your audio file. Whatever is remaining will get put into the 'other' category. All five of these tracks will then be converted to midi. The midi conversion works best for single timbre instruments. This means if whats left in the 'other' folder is more than one instrument, its midi conversion will not work successfully. This means if you're not transcribing piano, drums, bass, or voice; the best arrangements are solo performances, guitar duets (with voice, piano, bass, or drums), or even string quartets. Play around with it! Here are some things you can do with the extracted stems and midi files: Drag the MIDI files into any musical notation software to learn one or more parts from sheet music. Or drag the MIDI into your digital audio workstation and change the sounds.")
st.write("Contact jpellechia04@gmail.com for any questions!")

@st.cache_resource
def load_model(): 
    from spleeter.separator import Separator
    tf.compat.v1.config.experimental_run_functions_eagerly(False)
    separator = Separator('spleeter:5stems')
    return separator

# Load the model when the app starts
with st.spinner('Loading model...'):
    separator = load_model()

if uploaded_file is not None:
    # Save uploaded file
    filename = secure_filename(uploaded_file.name)
    file_path = os.path.join(upload_folder, filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File {filename} uploaded successfully.")

    try:
        logger.info("calling process music...")
        result = process_music(file_path, separator, use_concurrency=False)
        logger.info("processing music done")
    except Exception as e:
        st.error(f'Error processing file: {e}')

    # Check if the output folder exists and contains the expected zip file
    zip_filename = f"{os.path.splitext(filename)[0]}.zip"
    zip_filepath = os.path.join(output_folder, zip_filename)
    
    logger.info("searching for  zip")
    if os.path.exists(zip_filepath):
        with open(zip_filepath, 'rb') as f:
            st.download_button('Download processed files', f, file_name=zip_filename)
    else:
        st.error(f"Zip file not found: {zip_filepath}")