import keras.models
from keras.models import model_from_json
import pickle

def load_info():
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model/model.h5")

    with open('model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    return loaded_model, tokenizer