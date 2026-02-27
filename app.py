import streamlit as st
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.generate_music import generate_notes, create_midi

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="AI Mood Music Generator",
    page_icon="🎵",
    layout="centered"
)

# --------------------------------
# PREMIUM CSS + PARTICLES
# --------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #000000);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
}

@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

h1 {
    text-align: center;
    font-size: 42px;
    background: -webkit-linear-gradient(#00f5a0, #00d9f5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stButton>button {
    background: linear-gradient(90deg, #00f5a0, #00d9f5);
    color: black;
    font-size: 18px;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 20px #00f5a0;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# TITLE
# --------------------------------
st.markdown("<h1>🎵 AI Mood Music Generator</h1>", unsafe_allow_html=True)
st.write("Create AI-generated music based on your mood.")

st.divider()

# --------------------------------
# MOOD SELECTOR
# --------------------------------
mood = st.radio(
    "Select Your Mood",
    ["happy", "sad", "energetic"],
    horizontal=True
)

# --------------------------------
# TEMPO SLIDER (NEW)
# --------------------------------
tempo = st.slider(
    "Select Tempo (BPM)",
    min_value=60,
    max_value=180,
    value=120,
    step=5
)

# --------------------------------
# PATHS
# --------------------------------
MODEL_PATH = "music_model.h5"
NOTES_PATH = "notes.pkl"
OUTPUT_FOLDER = "generated_music"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------------------------------
# GENERATE BUTTON
# --------------------------------
if st.button("✨ Generate Music"):

    with st.spinner("Creating your AI music... 🎶"):

        try:
            model = load_model(MODEL_PATH)

            with open(NOTES_PATH, "rb") as f:
                notes = pickle.load(f)

            predicted_notes = generate_notes(model, notes, mood)

            output_file = os.path.join(
                OUTPUT_FOLDER,
                f"{mood}_output.mid"
            )

            create_midi(predicted_notes, output_file, mood, tempo)

            if os.path.exists(output_file):

                st.success("🎉 Music Generated Successfully!")

                # -------------------------------
                # WAVEFORM VISUALIZATION
                # -------------------------------
                st.subheader("🎼 Waveform Preview")

                numeric_wave = []

                for n in predicted_notes[:200]:
                   try:
                       # chord case
                       if "." in str(n):
                           numeric_wave.append(int(str(n).split(".")[0]))
                       else:
                           numeric_wave.append(int(n))
                   except:
                        # if note like C4 convert to pseudo number
                        numeric_wave.append(len(str(n)) * 5)

                waveform = np.array(numeric_wave, dtype=float)
                waveform = waveform / np.max(waveform)
                

                fig, ax = plt.subplots()
                ax.plot(waveform)
                ax.set_title("Generated Music Pattern")
                ax.set_xlabel("Time")
                ax.set_ylabel("Amplitude")

                st.pyplot(fig)

                # -------------------------------
                # DOWNLOAD
                # -------------------------------
                with open(output_file, "rb") as f:
                    st.download_button(
                        label="⬇ Download Music",
                        data=f,
                        file_name=f"{mood}_music.mid",
                        mime="audio/midi"
                    )

        except Exception as e:
            st.error(f"Error: {str(e)}")