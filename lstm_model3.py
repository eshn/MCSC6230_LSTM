from __future__ import print_function
from utils import parse, sample
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
import sys
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 0 for training from input.txt
# 1 for generating from user input (userinput.txt) for the next 50 words
# 2 for generating from user input (userinput.txt) until end of sentence
# 3 for generating from user input (userinput.txt) until end of paragraph
# if userinput.txt is not found, will use a random point in the literature as start point
MODE = 1
EPOCHS = 100
USERINPUT = 'userinput.txt'
WEIGHTS = 'weights/lstm_model3.hdf5'

def generate_text(sentence, model):
    x_pred = np.zeros((1, maxlen, len(wordlist)))
    if len(sentence) > maxlen:
        sentence = sentence[-maxlen:]
        for t, char in enumerate(sentence):
            if char in wordlist:
                x_pred[0, t, word_indices[char]] = 1.
    else:
        # x_pred = np.zeros((1, maxlen, len(wordlist)))
        if len(sentence) < maxlen:
            for t, char in enumerate(sentence):
                if char in wordlist:
                    x_pred[0, maxlen - len(sentence) + 1 + t, word_indices[char]] = 1.
        else:
            for t, char in enumerate(sentence):
                if char in wordlist:
                    x_pred[0, t, word_indices[char]] = 1.
    preds = model.predict(x_pred, verbose=0)[0]

    next_word = indices_word[np.argmax(preds)]

    sentence = sentence[1:]
    sentence.append(next_word)

    sys.stdout.write(str(next_word) + ' ')
    sys.stdout.flush()
    return sentence

with open('input.txt', 'r') as f:
    text = f.readlines()
    f.close()

# Creates dictionary
wordlist, text_parsed = parse(text, punc = 1)
word_indices = dict((c, i) for i, c in enumerate(wordlist))
indices_word = dict((i, c) for i, c in enumerate(wordlist))
# Windowing
maxlen = 10
step = 2
sentences = []
next_words = []
for i in range(0, len(text_parsed) - maxlen, step):
    sentences.append(text_parsed[i: i + maxlen])
    next_words.append(text_parsed[i + maxlen])
print('# of sentence fragments:', len(sentences))   # Sentences is a list of lists
print('# of words (with punctuation):', len(text_parsed))
print('RAM of hot vectors:', (np.floor(len(sentences)-maxlen / step)+1)*maxlen*len(wordlist) / 1000000, 'MB')

print(len(text_parsed), len(wordlist), len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(wordlist)), dtype=np.bool)
y = np.zeros((len(sentences), len(wordlist)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, word_indices[char]] = 1
    y[i, word_indices[next_words[i]]] = 1

## Building LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(wordlist))))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(len(wordlist)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001))

if os.path.exists(WEIGHTS):
    model.load_weights(WEIGHTS)

## Trains LSTM
if MODE == 0:
    for e in range(EPOCHS):
        print('\n')
        print('Iteration ', e+1, ' of ', EPOCHS)
        model_checkpoint = ModelCheckpoint(
            filepath=WEIGHTS,
            monitor='loss',
            verbose=1,
            save_best_only=True)
        model.fit(x, y,
            batch_size=256,
            epochs=1,
            verbose=1,
            callbacks=[model_checkpoint])
        start_index = random.randint(0, len(text_parsed) - maxlen - 1)
        sentence = text_parsed[start_index: start_index + maxlen]
        if (e + 1) % 25 == 0:
            for words in sentence:
                sys.stdout.write(str(words) + ' ')
            print()
            for i in range(50):
                x_pred = np.zeros((1, maxlen, len(wordlist)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, word_indices[char]] = 1.
                preds = model.predict(x_pred, verbose=0)[0]

                next_index = sample(preds, 0.5)
                next_word = indices_word[next_index]

                sentence = sentence[1:]
                sentence.append(next_word)

                sys.stdout.write(str(next_word) + ' ')
                sys.stdout.flush()

elif MODE == 1:
    if os.path.exists(USERINPUT):
        print('Using user input \n')
        with open(USERINPUT, 'r') as f:
            text = f.readlines()
            f.close()
        l, sentence = parse(text, punc=1)
    else:
        print('Using literature \n')
        start_index = random.randint(0, len(text_parsed) - maxlen - 1)
        sentence = text_parsed[start_index: start_index + maxlen]
    for i in range(50):
        sentence = generate_text(sentence, model)

elif MODE == 2:
    if os.path.exists(USERINPUT):
        print('Using user input \n')
        with open(USERINPUT, 'r') as f:
            text = f.readlines()
            f.close()
        l, sentence = parse(text, punc=1)
    else:
        print('Using literature \n')
        start_index = random.randint(0, len(text_parsed) - maxlen - 1)
        sentence = text_parsed[start_index: start_index + maxlen]
        if sentence[-1] == '.':
            sentence = text_parsed[start_index-1 : start_index + maxlen-1]
    while sentence[-1] != '.':
        sentence = generate_text(sentence, model)


elif MODE == 3:
    if os.path.exists(USERINPUT):
        print('Using user input \n')
        with open(USERINPUT, 'r') as f:
            text = f.readlines()
            f.close()
        l, sentence = parse(text, punc=1)
    else:
        print('Using literature \n')
        start_index = random.randint(0, len(text_parsed) - maxlen - 1)
        sentence = text_parsed[start_index: start_index + maxlen]
        if sentence[-1] == '\n':
            sentence = text_parsed[start_index-1 : start_index + maxlen-1]
    while sentence[-1] != '\n' and sentence[-2] != '\n':
        sentence = generate_text(sentence, model)