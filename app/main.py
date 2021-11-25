import pathlib
import json
import tensorflow as tf
import numpy as np
from fastapi import FastAPI
from typing import Optional
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

app = FastAPI()

# Where are the directories
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'models'
SPAM_SMS_MODEL_DIR = MODEL_DIR / 'spam-sms'
SPAM_SMS_MODEL_PATH = SPAM_SMS_MODEL_DIR / 'spam-model.h5'
SPAM_SMS_TOKENIZER_PATH = SPAM_SMS_MODEL_DIR / 'spam-classifer-tokenizer.json'
SPAM_SMS_METADATA_PATH = SPAM_SMS_MODEL_DIR / 'spam-classifer-metadata.json'

# Initialize model, tokenizer and metadata
AI_MODEL = None
AI_TOKENIZER = None
AI_METADATA = {}
labels_legend_inverted = {}

# Function for loading model on start up
@app.on_event('startup')
def on_startup():
    
    # make variables global
    global AI_MODEL, AI_TOKENIZER, AI_METADATA, labels_legend_inverted

    # load spam sms model
    if SPAM_SMS_MODEL_PATH.exists():
        AI_MODEL = load_model(SPAM_SMS_MODEL_PATH, compile=False)
        #print(AI_MODEL.summary())

    # load spam sms tokenizer
    if SPAM_SMS_TOKENIZER_PATH.exists():
        tokenizer_string = SPAM_SMS_TOKENIZER_PATH.read_text()
        AI_TOKENIZER = tokenizer_from_json(tokenizer_string)

    # load spam sms metadata
    if SPAM_SMS_METADATA_PATH.exists():
        AI_METADATA = json.loads(SPAM_SMS_METADATA_PATH.read_text())
        labels_legend_inverted = AI_METADATA['labels_legend_inverted']

def predict(q:str):
    
    # 1. convert to list of sequences
    sequences = AI_TOKENIZER.texts_to_sequences([q])

    # 2. pad sequences
    maxlen = AI_METADATA['max_sequence']
    input = pad_sequences(sequences, maxlen = maxlen)
    
    # 3. predict 
    preds = AI_MODEL.predict(input)[0]
    top_idx = np.argmax(preds)
    max_pred_dict= {labels_legend_inverted[str(top_idx)]: float(preds[top_idx])}
    print(max_pred_dict)
    preds_dict = {labels_legend_inverted[str(i)]: float(x) for i, x in enumerate(preds)}

    # 4. convert to labels + probabilities
    return json.loads(json.dumps({'predictions': preds_dict, 'top prediction': max_pred_dict}, cls=NumpyEncoder))

# main API call endpoint
@app.get('/') # /?q=this is awesome
def read_index(q: Optional[str] = None):
    query = q or 'hello world'
    print(f'query is: {q}')
    preds_dict = predict(query)

    # FastAPI tries to dump below into json. That's one of the reasons it needs decorators:
    # -> function gets wrapped by app.get function, which probably converts dict to json. 
    return {'query': query, **AI_METADATA, 'results': preds_dict}


# Copy method from docs because failing to import
def tokenizer_from_json(json_string):
    """Parses a JSON tokenizer configuration file and returns a
    tokenizer instance.
    # Arguments
        json_string: JSON string encoding a tokenizer configuration.
    # Returns
        A Keras Tokenizer instance
    """
    tokenizer_config = json.loads(json_string)
    config = tokenizer_config.get('config')

    word_counts = json.loads(config.pop('word_counts'))
    word_docs = json.loads(config.pop('word_docs'))
    index_docs = json.loads(config.pop('index_docs'))
    # Integer indexing gets converted to strings with json.dumps()
    index_docs = {int(k): v for k, v in index_docs.items()}
    index_word = json.loads(config.pop('index_word'))
    index_word = {int(k): v for k, v in index_word.items()}
    word_index = json.loads(config.pop('word_index'))

    tokenizer = Tokenizer(**config)
    tokenizer.word_counts = word_counts
    tokenizer.word_docs = word_docs
    tokenizer.index_docs = index_docs
    tokenizer.word_index = word_index
    tokenizer.index_word = index_word

    return tokenizer

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
        