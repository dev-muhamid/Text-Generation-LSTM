import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

SEQ_LENGTH = 100
test_x = np.array([1, 2, 0, 4, 3, 7, 10])

# one hot encoding
test_y = to_categorical(test_x)
print(test_x)
print(test_y)

# Using keras functional model
def create_functional_model(n_layers, input_shape, hidden_dim, n_out, **kwargs):
    drop        = kwargs.get('drop_rate', 0.2)
    activ       = kwargs.get('activation', 'softmax')
    mode        = kwargs.get('mode', 'train')
    hidden_dim  = int(hidden_dim)

    inputs      = Input(shape = (input_shape[1], input_shape[2]))
    model       = LSTM(hidden_dim, return_sequences = True)(inputs)
    model       = Dropout(drop)(model)
    model       = Dense(n_out)(model)

# Using keras sequential model
def create_model(n_layers, input_shape, hidden_dim, n_out, **kwargs):
    drop        = kwargs.get('drop_rate', 0.2)
    activ       = kwargs.get('activation', 'softmax')
    mode        = kwargs.get('mode', 'train')
    hidden_dim  = int(hidden_dim)
    model       = Sequential()
    flag        = True

    if n_layers == 1:
        model.add( LSTM(hidden_dim, input_shape = (input_shape[1], input_shape[2])) )
        if mode == 'train':
            model.add( Dropout(drop) )

    else:
        model.add( LSTM(hidden_dim, input_shape = (input_shape[1], input_shape[2]), return_sequences = True) )
        if mode == 'train':
            model.add( Dropout(drop) )
        for i in range(n_layers - 2):
            model.add( LSTM(hidden_dim, return_sequences = True) )
            if mode == 'train':
                model.add( Dropout(drop) )
        model.add( LSTM(hidden_dim) )

    model.add( Dense(n_out, activation = activ) )

    return model

def train(model, X, Y, n_epochs, b_size, vocab_size, **kwargs):
    loss            = kwargs.get('loss', 'categorical_crossentropy')
    opt             = kwargs.get('optimizer', 'adam')

    model.compile(loss = loss, optimizer = opt)

    filepath        = "weights-improvement-{epoch:02d}-{loss:.4f}.h5"
    checkpoint      = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
    callbacks_list  = [checkpoint]
    X               = X / float(vocab_size)
    model.fit(X, Y, epochs = n_epochs, batch_size = b_size, callbacks = callbacks_list)

def generate_text_with_prompt(model, filename, ix_to_char, char_to_int, vocab_size, prompt_text):
    model.load_weights(filename)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
    
    # Convert prompt to pattern
    pattern = [char_to_int.get(char, 1) for char in prompt_text.lower()[-SEQ_LENGTH:]]
    if len(pattern) < SEQ_LENGTH:
        pattern = [1] * (SEQ_LENGTH - len(pattern)) + pattern
    
    print("Seed:", '"' + prompt_text + '"')
    output = []
    for i in range(250):
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(vocab_size)
        prediction = model.predict(x, verbose = 0)
        index = np.argmax(prediction)
        output.append(index)
        pattern.append(index)
        pattern = pattern[1:]
    
    print("Generated:", '"' + ''.join([ix_to_char[value] for value in output]) + '"')

def generate_text(model, X, filename, ix_to_char, vocab_size):
    model.load_weights(filename)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
    
    start = np.random.randint(0, len(X) - 1)
    pattern = np.ravel(X[start]).tolist()
    
    print("Seed:", '"' + ''.join([ix_to_char[value] for value in pattern]) + '"')
    output = []
    for i in range(250):
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(vocab_size)
        prediction = model.predict(x, verbose = 0)
        index = np.argmax(prediction)
        output.append(index)
        pattern.append(index)
        pattern = pattern[1:]
    
    print("Generated:", '"' + ''.join([ix_to_char[value] for value in output]) + '"')

import zipfile

filename = 'game_of_thrones.txt.zip'

# Extract the first text file from the ZIP archive
with zipfile.ZipFile(filename, 'r') as zip_ref:
    with zip_ref.open(zip_ref.namelist()[0]) as text_file:  # Assuming there's only one text file
        data = text_file.read().decode('utf-8')  # Decode bytes to string (optional, adjust encoding if needed)

# Now you have the text data from the first text file in the ZIP in the 'data' variable  

data        = data.lower()
# Find all the unique characters
chars       = sorted(list(set(data)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
ix_to_char  = dict((i, c) for i, c in enumerate(chars))
vocab_size  = len(chars)

print("List of unique characters : \n", chars)

print("Number of unique characters : \n", vocab_size)

print("Character to integer mapping : \n", char_to_int)

list_X      = []
list_Y      = []

# Python append is faster than numpy append. Try it!
for i in range(0, len(data) - SEQ_LENGTH, 1):
    seq_in  = data[i : i + SEQ_LENGTH]
    seq_out = data[i + SEQ_LENGTH]
    list_X.append([char_to_int[char] for char in seq_in])
    list_Y.append(char_to_int[seq_out])

n_patterns  = len(list_X)
print("Number of sequences in data set : \n", n_patterns)
print(list_X[0])
print(list_X[1])

X           = np.reshape(list_X, (n_patterns, SEQ_LENGTH, 1)) # (n, 100, 1)
# Encode output as one-hot vector
Y           = to_categorical(list_Y)

print(X[0])
print(Y[0])

print("Shape of input data ", X.shape, "\nShape of output data ", Y.shape)

model   = create_model(1, X.shape, 256, Y.shape[1], mode = 'train')

train(model, X[:1024], Y[:1024], 10, 512, vocab_size)

# Find the best model
best_model = None
for filename in os.listdir():
    if filename.endswith('.h5'):
        if best_model is None or filename > best_model:
            best_model = filename

if best_model:
    print(f"Using best model: {best_model}")
    
    while True:
        choice = input("\nChoose option:\n1. Generate with custom prompt\n2. Generate random text\n3. Exit\nEnter choice (1-3): ")
        
        if choice == '1':
            prompt = input("Enter your prompt: ")
            generate_text_with_prompt(model, best_model, ix_to_char, char_to_int, vocab_size, prompt)
        elif choice == '2':
            generate_text(model, X, best_model, ix_to_char, vocab_size)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
else:
    print("No trained models found.")