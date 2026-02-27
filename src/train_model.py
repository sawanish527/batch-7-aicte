import os
import pickle
import numpy as np

from music21 import converter, instrument, note, chord

from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import LSTM, Dense, Dropout # pyright: ignore[reportMissingImports]
from tensorflow.keras.utils import to_categorical # type: ignore

# -------------------------
# Extract notes from single MIDI
# -------------------------
def extract_notes_from_midi(file_path):
    notes = []
    try:
        midi = converter.parse(file_path)

        parts = instrument.partitionByInstrument(midi)
        if parts:
            elements = parts.parts[0].recurse()
        else:
            elements = midi.flat.notes

        for element in elements:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return notes


# -------------------------
# Load all dataset notes
# -------------------------
def load_all_notes(dataset_path):
    all_notes = []

    for mood in os.listdir(dataset_path):
        mood_path = os.path.join(dataset_path, mood)

        if not os.path.isdir(mood_path):
            continue

        for file in os.listdir(mood_path):
            if file.endswith(".mid"):
                file_path = os.path.join(mood_path, file)
                all_notes += extract_notes_from_midi(file_path)

    return all_notes


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(BASE_DIR, "dataset")
    model_path = os.path.join(BASE_DIR, "models")
    notes_path = os.path.join(BASE_DIR, "notes.pkl")

    os.makedirs(model_path, exist_ok=True)

    print("Loading notes...")
    notes = load_all_notes(dataset_path)

    if len(notes) < 200:
        print("❌ Not enough notes found. Add more MIDI files.")
        exit()

    print("Total notes:", len(notes))

    # Save notes
    with open(notes_path, "wb") as f:
        pickle.dump(notes, f)

    # -------------------------
    # Prepare sequences
    # -------------------------
    sequence_length = 100

    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)

    note_to_int = {note: num for num, note in enumerate(pitchnames)}

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]

        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    print("Patterns:", n_patterns)

    X = np.reshape(network_input, (n_patterns, sequence_length, 1))
    X = X / float(n_vocab)

    y = to_categorical(network_output)

    # -------------------------
    # Build LSTM model
    # -------------------------
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam")

    print(model.summary())

    # -------------------------
    # Train
    # -------------------------
    model.fit(X, y, epochs=20, batch_size=64)

    # Save model inside models folder
    model.save(os.path.join(model_path, "music_model.h5"))

    print("✅ Model saved in models/music_model.h5")
