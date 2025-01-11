import os
import numpy as np
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation

# Function to get all MIDI files from a folder
def get_midi_files(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified folder does not exist: {path}")
    
    midi_files = [f for f in os.listdir(path) if f.endswith(".mid")]
    if not midi_files:
        raise ValueError(f"No MIDI files found in folder: {path}. Please add valid `.mid` files.")
    
    print(f"Found {len(midi_files)} MIDI files in {path}.")
    return [os.path.join(path, midi) for midi in midi_files]

# Function to extract notes and chords from MIDI files
def extract_notes(midi_paths):
    notes = []
    for midi_path in midi_paths:
        try:
            midi = converter.parse(midi_path)
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
    if not notes:
        raise ValueError("No notes extracted from the MIDI files.")
    return notes

# Prepare sequences of notes for training
def prepare_sequences(notes, sequence_length):
    note_names = sorted(set(notes))
    note_to_int = {note: number for number, note in enumerate(note_names)}
    int_to_note = {number: note for note, number in note_to_int.items()}
    
    network_input = []
    network_output = []
    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[note] for note in sequence_in])
        network_output.append(note_to_int[sequence_out])

    if not network_input or not network_output:
        raise ValueError("Sequences could not be generated. Check the dataset size or sequence length.")
    
    X = np.reshape(network_input, (len(network_input), sequence_length, 1))
    X = X / float(len(note_names))  # Normalize input data
    y = np.zeros((len(network_output), len(note_names)))
    for i, output in enumerate(network_output):
        y[i][output] = 1

    return X, y, note_to_int, int_to_note

# Build the LSTM model
def build_model(input_shape, output_size):
    model = Sequential()
    model.add(LSTM(512, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Generate music using the trained model
def generate_music(model, sequence_length, note_to_int, int_to_note, seed=None, num_notes=500):
    prediction_input = seed or network_input[np.random.randint(0, len(network_input)-1)]
    prediction_output = []

    for _ in range(num_notes):
        input_reshaped = np.reshape(prediction_input, (1, len(prediction_input), 1))
        input_reshaped = input_reshaped / float(len(note_to_int))
        predicted_probs = model.predict(input_reshaped, verbose=0)
        index = np.argmax(predicted_probs)
        result = int_to_note[index]
        prediction_output.append(result)
        prediction_input.append(index)
        prediction_input = prediction_input[1:]

    return prediction_output

# Create a MIDI file from generated notes
def create_midi(notes, file_name="generated_music.mid"):
    midi_stream = stream.Stream()
    for pattern in notes:
        if '.' in pattern or pattern.isdigit():
            chord_notes = [note.Note(int(n)) for n in pattern.split('.')]
            midi_stream.append(chord.Chord(chord_notes))
        else:
            midi_stream.append(note.Note(pattern))
    midi_stream.write('midi', fp=file_name)

# Main script
try:
    midi_folder = r"P:\lofi-music"  # Update this path to the correct folder
    print(f"Looking for MIDI files in: {midi_folder}")
    midi_files = get_midi_files(midi_folder)
    
    notes = extract_notes(midi_files)
    print(f"Total notes and chords extracted: {len(notes)}")

    sequence_length = 50
    X, y, note_to_int, int_to_note = prepare_sequences(notes, sequence_length)
    print(f"Input shape: {X.shape}, Output shape: {y.shape}")

    model = build_model((X.shape[1], X.shape[2]), y.shape[1])
    model.fit(X, y, epochs=200, batch_size=32)

    generated_notes = generate_music(model, sequence_length, note_to_int, int_to_note)
    print("Generated Notes:", generated_notes)

    create_midi(generated_notes, file_name="generated_lofi_music.mid")
    print("MIDI file generated: generated_lofi_music.mid")
except Exception as e:
    print(f"An error occurred: {e}")
