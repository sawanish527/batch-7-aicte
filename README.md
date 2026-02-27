##  Dataset Information

The model was trained on a custom curated MIDI dataset categorized into:
- Happy (10 files)
- Sad (11 files)
- Energetic (8 files)

Total Files: 29 MIDI files

Each MIDI file was processed using the music21 library to extract note sequences.

##  Model Architecture

- LSTM Layers: 2
- Hidden Units: 256
- Dropout: 0.3
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Epochs: 50
- Batch Size: 64

##  How to Train the Model

1. Place MIDI dataset inside `dataset/` folder.
2. Run:
   python src/train_model.py
3. Model will be saved as:
   models/music_model.h5
