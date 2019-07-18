from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse,Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from django.http import JsonResponse
from rasa_nlu.model import Interpreter
import manage
import json
import os
import pickle

from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras import backend as K
import pandas as pd
import psycopg2
from emotions_model import Preprocess_your_Data_with_Gensim
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100
# libs for database interface

import datetime
#'''Function to insert the form data 'values' into table 'table'

def predstore(table1,table2,columns1,columns2,values1,values2):
    connection = psycopg2.connect("dbname='data_emotions' user='postgres' password='123' host='localhost' port='5433'")
    mark = connection.cursor()
    statement1 = 'INSERT INTO ' + table1 + ' (' + columns1 + ') VALUES (%s,%s,%s,%s)'
    statement2 = 'INSERT INTO ' + table2 + ' (' + columns2 + ') VALUES (%s,%s,%s)'
    mark.execute(statement1,values1)
    mark.execute(statement2,values2)
    connection.commit() 
    return "insertion valide"

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
#filename = os.path.abspath('emotions_model/tok_lstm_glove_prétraitement.sav')
filename = os.path.abspath('./tok_prétraitement2.sav')
#filename = 'C:/Users/amel/Desktop/PFE/tok_lstm_glove_sans_prétraitement.sav'
#loaded_tok = pickle.load(open(filename, 'rb'))
       
       
#x_input = np.array(['I"m happy'])
#seq= loaded_tok.texts_to_sequences(x_input)
#seqs = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
file = os.path.abspath('./finalized_model_prétraitement2.sav')
#file = 'C:/Users/amel/Desktop/PFE/lstm_glove_model_prétraitement.sav'

# load the model from disk
@api_view(["POST"])
def prediction(text):
  try:
      msg=json.loads(text.body)
      #file1 = os.path.abspath('webservice/current')
      #interpreter = Interpreter.load(file1)
      interpreter = Interpreter.load(os.path.abspath('./rasa_model'))
      intent=interpreter.parse(msg["message"])
      loaded_model = pickle.load(open(file, 'rb'))
      loaded_tok = pickle.load(open(filename, 'rb'))
      #x_input = np.array([msg["message"]])
      x_input = np.array([Preprocess_your_Data_with_Gensim.transformText(msg["message"])])
      seq= loaded_tok.texts_to_sequences(x_input)
      seqs = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
      probability = loaded_model.predict(seqs)
      r=str(probability)
      class_pred = loaded_model.predict_classes(seqs)

      K.clear_session()
      
      y=probability[0][class_pred[0]]
      x=str(y)
      if class_pred[0]==0 :
         classe='anger'
      if class_pred[0]==1 :
         classe='disgust'
      if class_pred[0]==2 :
         classe='fear'
      if class_pred[0]==3 :
         classe='guilt'
      if class_pred[0]==4 :
         classe='joy'
      if class_pred[0]==5 :
         classe='sadness' 
      if class_pred[0]==6 :
         classe='shame' 

      table1 = "prediction_emotion_store"
      table2 = "feedback_emotion_store"
      fields1 = "id, label, text ,prob"
      fields2 = "id, label, text"
      values1 = (msg["id"] , classe , msg["message"] , x)
      values2 = (msg["id"] , classe , msg["message"])
      predstore(table1,table2,fields1,fields2,values1,values2)  



      #df = pd.read_csv('C:/Users/amel/Desktop/PFE/data_sans_prétraitement.csv', delimiter=';')
      #df.loc[len(df)]=[classe,msg["message"]]
      #df.to_csv('C:/Users/amel/Desktop/PFE/data_sans_prétraitement.csv', sep=';', index=False)
      #HttpResponse(JSON.parse(JSON.stringify({"id":msg["id"],"message":msg["message"],"label":classe,"intent":intent['intent']['name'],"probability":x})), content_type='application/json')     
      return HttpResponse(json.dumps({"id":msg["id"],"message":msg["message"],"label":classe,"intent":intent['intent']['name'],"probability":x}), content_type='application/json')
      #return HttpResponse(json.dumps({"id":msg["id"],"message":msg["message"],"label":classe,"probability":x}), content_type='application/json')
      
      # HttpResponse(json.dumps({"label":classe}), content_type='application/json')
  except ValueError as e:
      return Response(e.args[0],status.HTTP_400_BAD_REQUEST)

@api_view(["GET"])
def annotation_emotion(text):
  try:
     msg=json.loads(text.body)
     message=msg["message"]
     annotation=msg["annotation"]
     df = pd.read_csv('C:/Users/amel/Desktop/PFE/data_sans_prétraitement.csv', delimiter=';')
     df.loc[len(df)]=[msg["annotation"],msg["message"]]
     df.to_csv('C:/Users/amel/Desktop/PFE/data_sans_prétraitement.csv', sep=';', index=False)
     return HttpResponse(json.dumps({"message":msg["message"],"annotation":msg["annotation"]}), content_type='application/json')
  except ValueError as e:
       return Response(e.args[0],status.HTTP_400_BAD_REQUEST)      
