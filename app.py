from ctypes import c_ssize_t
import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from autocorrect import Speller
spell = Speller(lang='en')
from keras.models import load_model
model = load_model('model.h5')
import json
import random
import itertools 
from flask import Flask, render_template, request
intents = json.loads(open('data.json',encoding="utf8").read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_wordsL = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    if "and" in sentence_words:
        sentence_wordsL = [lemmatizer.lemmatize(spell(word.lower())) for word in sentence_words]
        i=sentence_words.index("and")
        sentence_words1=sentence_wordsL[:i]
        sentence_words2=sentence_wordsL[i+1:]
        return (sentence_words1,sentence_words2)
    else :
        return sentence_wordsL

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    if str(type(sentence_words))=='<class \'list\'>':
    # bag of words - matrix of N words, vocabulary matrix
        bag = [0]*len(words)  
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s: 
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)
        return np.array(bag)
    else :
        bag1 = [0]*len(words)  
        bag2 = [0]*len(words) 
        for (s,p) in zip(sentence_words[0],sentence_words[1]):
            for i,w in enumerate(words):
                if w == s: 
                    # assign 1 if current word is in the vocabulary position
                    bag1[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)
                if w == p :
                     bag2[i] = 1      
        return (np.array(bag1) , np.array(bag2))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    if str(type(p)) == '<class \'numpy.ndarray\'>':
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list
    elif str(type(p)) == '<class \'tuple\'>' :
        res1 = model.predict(np.array([p[0]]))[0]
        ERROR_THRESHOLD = 0.25
        results1 = [[i1,r1] for i1,r1 in enumerate(res1) if r1>ERROR_THRESHOLD]
        # sort by strength of probability
        results1.sort(key=lambda x: x[1], reverse=True)
        return_list1 = []
        for r1 in results1:
            return_list1.append({"intent": classes[r1[0]], "probability": str(r1[1])})
        res2 = model.predict(np.array([p[1]]))[0]
        ERROR_THRESHOLD = 0.25
        results2 = [[i2,r2] for i2,r2 in enumerate(res2) if r2>ERROR_THRESHOLD]
        # sort by strength of probability
        results2.sort(key=lambda x: x[1], reverse=True)
        return_list2 = []
        for r2 in results2:
            return_list2.append({"intent": classes[r2[0]], "probability": str(r2[1])})  
        return (return_list1 , return_list2)

def getResponse(ints, intents_json):
 if str(type(ints))=='<class \'list\'>':
  try:
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(tag == i['tag']):
            result = random.choice(i['responses'])
  except IndexError:
    result ='Sorry i can\'t understand you '
 elif str(type(ints))=='<class \'tuple\'>':
      try :
        tag1 = ints[0][0]['intent']
        tag2 = ints[1][0]['intent']
        list_of_intents = intents_json['intents']
    #result=str(ints)
    #return result
    
      
        for i in list_of_intents:
            if(tag1==i['tag']):
                result1 = random.choice(i['responses']) 
    
        for i in list_of_intents:
            if(tag2==i['tag']):
                result2 = random.choice(i['responses'])  
        result= result1 + " and " + result2  
      except IndexError:
        result ='Sorry i can\'t understand your question '   
 return result                     

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
