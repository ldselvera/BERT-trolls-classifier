from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
  
def prediction(question, model, tokenizer):
  maxlen = 100 # max number of words in a question to use
  print("1",question)
  quest= []
  quest.append(question)
  print("2",quest)
  quest = tokenizer.texts_to_sequences(quest)
  quest = pad_sequences(quest, maxlen=maxlen)
  print("3",quest)

  pred_val = model.predict([quest], batch_size=1024, verbose=1)
  print(pred_val)
  pred_val=(pred_val>0.3).astype(int)
  print(pred_val)
  
  if pred_val[0] == 1:
    result = 'Improper Question'
  else:
    result = 'Proper Question'
  return result