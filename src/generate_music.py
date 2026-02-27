import pickle
import numpy as np
import os
import sys

from music21 import note, chord, stream, instrument
from tensorflow.keras.models import load_model
from music21 import tempo as mtempo



# --------------------------------
# Paths
# --------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "music_model.h5")
notes_path = os.path.join(BASE_DIR, "notes.pkl")
output_folder = os.path.join(BASE_DIR, "generated_music")

os.makedirs(output_folder, exist_ok=True)


# --------------------------------
# Generate Notes (Mood Controlled)
# --------------------------------
def generate_notes(model, notes, mood, n_notes=500):

    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)

    note_to_int = {note: num for num, note in enumerate(pitchnames)}
    int_to_note = {num: note for num, note in enumerate(pitchnames)}

    sequence_length = 100

    start = np.random.randint(0, len(notes) - sequence_length)
    pattern = notes[start:start + sequence_length]
    pattern = [note_to_int[n] for n in pattern]

    output_notes = []

    for _ in range(n_notes):

        prediction_input = np.reshape(pattern, (1, sequence_length, 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        prediction = prediction.flatten()

        # 🎛 Mood-based temperature
        if mood == "happy":
            temperature = 1.2
        elif mood == "sad":
            temperature = 0.7
        elif mood == "energetic":
            temperature = 1.5
        else:
            temperature = 1.0

        # Stable softmax sampling
        prediction = np.asarray(prediction).astype("float64")
        prediction = np.log(prediction + 1e-9) / temperature
        prediction = np.exp(prediction - np.max(prediction))
        prediction = prediction / np.sum(prediction)

        index = np.random.choice(len(prediction), p=prediction)

        result = int_to_note[index]
        output_notes.append(result)

        pattern.append(index)
        pattern = pattern[1:]

    return output_notes


# --------------------------------
# Create MIDI (Mood Instrument)
# --------------------------------

def create_midi(predicted_notes, file_name, mood, tempo_value):

    offset = 0
    output = []

    # 🎹 Mood-based instrument
    if mood == "sad":
        inst = instrument.Piano()
    elif mood == "happy":
        inst = instrument.AcousticGuitar()
    elif mood == "energetic":
        inst = instrument.ElectricGuitar()
    else:
        inst = instrument.Piano()

    midi_stream = stream.Stream()

    # 🎵 Add tempo to MIDI
    midi_stream.append(mtempo.MetronomeMark(number=tempo_value))

    for pattern in predicted_notes:

        try:
            if "." in str(pattern):
                notes_in_chord = str(pattern).split(".")
                chord_notes = []

                for n in notes_in_chord:
                    new_note = note.Note(int(n))
                    new_note.storedInstrument = inst
                    chord_notes.append(new_note)

                new_chord = chord.Chord(chord_notes)
                new_chord.offset = offset
                midi_stream.append(new_chord)

            else:
                try:
                    new_note = note.Note(int(pattern))
                except:
                    new_note = note.Note(pattern)

                new_note.offset = offset
                new_note.storedInstrument = inst
                midi_stream.append(new_note)

            offset += np.random.choice([0.25, 0.5, 0.75])

        except:
            continue

    print("Total notes written:", len(predicted_notes))

    midi_stream.write("midi", file_name)

    print("✅ Music saved at:", file_name)

# --------------------------------
# Main Runner
# --------------------------------
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("❌ Please provide mood (happy/sad/energetic)")
        sys.exit()

    mood = sys.argv[1]

    print(f"🎵 Generating music for mood: {mood}")

    # Load model
    model = load_model(model_path)

    # Load notes
    with open(notes_path, "rb") as f:
        notes = pickle.load(f)

    # Generate notes
    predicted_notes = generate_notes(model, notes, mood)
    print("Generated notes:", len(predicted_notes))

    # Output file
    output_file = os.path.join(output_folder, f"{mood}_output.mid")

    # Create MIDI
    create_midi(predicted_notes, output_file, mood)
