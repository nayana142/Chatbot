import random 
import json
import pickle
import numpy as np 
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization
from tensorflow.keras.optimizers import SGD,Adam

 

# Load the data
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents2.json').read())

words=[]
classes=[]
documents=[]
ignore_words=['?','!',',','.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word_list=nltk.word_tokenize('pattern')
        words.extend(word_list)
        # Add documents
        documents.append((word_list,intent['tag']))
        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words=[lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words ]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle and convert to np.array
random.shuffle(training)
training = np.array(training, dtype=object)

# Split the features and labels
train_x = list(training[:, 0])
train_y = list(training[:, 1])
# train_x = list(training[:,:len(words)])
# train_y = list(training[:,len(words):])

# Build the model
model = Sequential()
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model with Adam optimizer
adam = Adam(learning_rate=0.001)

# Compile the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist=model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=10, verbose=1)
model.save('chatbot_model.h5',hist)


