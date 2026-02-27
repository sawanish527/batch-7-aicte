from music21 import converter, instrument, note, chord
import os

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
        print(f"Skipping file (error): {file_path}")

    return notes

def load_notes_from_dataset(dataset_path):
    """
    Reads all MIDI files from dataset folder
    and returns a dictionary with mood-wise notes
    """
    mood_notes = {}

    for mood in os.listdir(dataset_path):
        mood_folder = os.path.join(dataset_path, mood)
        if not os.path.isdir(mood_folder):
            continue

        all_notes = []

        for file in os.listdir(mood_folder):
            if file.endswith(".mid"):
                file_path = os.path.join(mood_folder, file)
                notes = extract_notes_from_midi(file_path)
                all_notes.extend(notes)

        mood_notes[mood] = all_notes
        print(f"{mood} -> {len(all_notes)} notes extracted")

    return mood_notes

if __name__ == "__main__":
    import os

    CURRENT_FILE = os.path.abspath(__file__)
    SRC_DIR = os.path.dirname(CURRENT_FILE)
    BASE_DIR = os.path.dirname(SRC_DIR)

    dataset_path = os.path.join(BASE_DIR, "dataset")

    print("Looking for dataset at:", dataset_path)

    mood_notes = load_notes_from_dataset(dataset_path)

    for mood, notes in mood_notes.items():
        print(mood, "sample notes:", notes[:10])


