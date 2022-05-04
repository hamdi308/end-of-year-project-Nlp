import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import gradient_descent_v2
import random
import tensorflow as tf

words=[]
classes = []
documents = []
ignore_words = ['?', '!', '+', '*', '/']
data_file = open('data.json',encoding="utf8").read() #n7ellou fichier json en mode read
intents = json.loads(data_file) # n7ottou les listes lmawjoudin f fichier json f dictionnaire we7ed bech l prg python ynejem ya9raha (format python)

for intent in intents['intents']: # liste intents feha liste we7eda esmha intent 
    for pattern in intent['patterns']: #ne5dou kol objet f liste intent 3andou proprietee pattern
        w = nltk.word_tokenize(pattern) #nfekkou l kelmett lkol ta3 les proprietee pattern lkol w n7otouhom f liste
        words.extend(w) #n3biw tableau words b kelmett ta3 les pattern lkol
        documents.append((w, intent['tag'])) # na3mlou couple l kol liste ta3 kelmett men pattern m3ah l tag mte3ou n7ottouh f liste documents 
        if intent['tag'] not in classes:
            classes.append(intent['tag']) #n7ottou les tag l kol f liste classes  
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words] #l kelmett f liste words ne5dou l prefix mte3hom w nrodouhom minuscule
words = sorted(words) # nadhmou lekelmet elli mawjoudin f liste words 7asbb l pid mte3hom ascii
classes = sorted(classes) # nadhmou lekelmet elli mawjoudin f liste classes 7asbb l pid mte3hom ascii
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes) 
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('texts.pkl','wb')) #n7otoou liste words f fichier texts.pkl sous forme binaire
pickle.dump(classes,open('labels.pkl','wb')) #n7otoou liste classes f fichier labels.pkl sous forme binaire

training = [] #liste n7ottou feha les elements ba3d trainig
output_empty = [0] * len(classes) #tableau fere88 taille mte3ou 9ad taiile ta3 liste classes n7ottou fih les sorties
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words] #ntal3ou kelmett commun lel kelmett elli 3anna f pattern w n7otouhom f liste jdida pattern_words
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0) #tableau fih 0 w 1 si leklma elli ktebha l users mawjouda f pattern_words sinon te5ou 0
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row]) #array fih 2 lignes
print(output_row)    
random.shuffle(training) #n5altou l array training
training = np.array(training)
print(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = gradient_descent_v2.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=380, batch_size=5, verbose=1)
hist = model.fit(np.array(train_x), np.array(train_y), validation_split=0.9, epochs=65, batch_size=5, verbose=1)
model.save('model.h5', hist)
print(len(train_y[0]))
print("model created")