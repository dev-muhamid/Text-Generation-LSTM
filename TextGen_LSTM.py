import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

SEQ_LENGTH = 100
hidden_dim=128
drop_rate=0.2
n_out=51
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

# Define the input layer
inputs = keras.layers.Input(shape=(SEQ_LENGTH, 1))


# Define the LSTM layer
lstm = LSTM(hidden_dim, return_sequences=True)(inputs)

# Define the dropout layer
dropout = Dropout(drop_rate)(lstm)

# Define the dense layer
dense = Dense(51, activation='softmax')(dropout)

# Create the model
model = Model(inputs=inputs, outputs=dense)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')
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

    filepath        = "weights-improvement-{epoch:02d}-{loss:.4f}.keras"
    checkpoint      = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
    callbacks_list  = [checkpoint]
    X               = X / float(vocab_size)
    model.fit(X, Y, epochs = n_epochs, batch_size = b_size, callbacks = callbacks_list)

    def generate_text(model, X, filename, ix_to_char, vocab_size):

    # Load the weights from the epoch with the least loss
    # 
     model.load_weights(filename)
     model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    start   = np.random.randint(0, len(X) - 1)
    pattern = np.ravel(X[start]).tolist()

    # We seed the model with a random sequence of 100 so it can start predicting
    print ("Seed:")
    print ("\"", ''.join([ix_to_char[value] for value in pattern]), "\"")
    output = []
    for i in range(250):
        x           = np.reshape(pattern, (1, len(pattern), 1))
        x           = x / float(vocab_size)
        prediction  = model.predict(x, verbose = 0)
        index       = np.argmax(prediction)
        result      = index
        output.append(result)
        pattern.append(index)
        pattern = pattern[1 : len(pattern)]

    print("Predictions")
    print ("\"", ''.join([ix_to_char[value] for value in output]), "\"")

import zipfile

filename = filename = 'C:/Users/mr708/VScode_Projects/PythonProjects/game_of_thrones.txt.zip'

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

train(model, X[:1024], Y[:1024], 2, 512, vocab_size)


# Iterating through each model and generating text
for filename in os.listdir():
    if filename.endswith('.hdf5'):
        print("Model Name:", filename)
        generate_gexf(model, X, filename, ix_to_char, vocab_size)
