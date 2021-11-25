import json
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import encoders


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

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


@dataclass
class AIModel:

    model_path: Path
    tokenizer_path: Optional[Path] = None
    metadata_path: Optional[Path] = None

    model = None
    tokenizer = None
    metadata = None

    def __post_init__(self):
        
        # load model
        if self.model_path.exists():
            self.model = load_model(self.model_path, compile=False)

        # load tokenizer if passed as argument
        if self.tokenizer_path:
            if self.tokenizer_path.exists():
                tokenizer_string = self.tokenizer_path.read_text()
                self.tokenizer = tokenizer_from_json(tokenizer_string)
        
        # load metadata if passed as argument
        if self.metadata_path:
            if self.metadata_path.exists():
                metadata_string = self.metadata_path.read_text()
                self.metadata = json.loads((metadata_string))

    # return model
    def get_model(self):
        if not self.model:
            raise Exception("Model not implemented")
        return self.model

    # return tokenizer
    def get_tokenizer(self):
        if not self.tokenizer:
            raise Exception("Tokenizer not implemented")
        return self.tokenizer

     # return metadata
    def get_metadata(self):
        if not self.metadata:
            raise Exception("Metadata not implemented")
        return self.metadata

    # return metadata
    def get_legend_inverted(self):
        metadata = self.get_metadata()
        legend = metadata.get('labels_legend_inverted')
        return legend

    # text to sequences
    def get_sequences_from_text(self, texts: List[str]):
        tokenizer = self.get_tokenizer()
        sequences = tokenizer.texts_to_sequences(texts)
        return sequences
    
    # sequences to input
    def get_input_from_sequences(self, sequences):
        metadata = self.get_metadata()
        maxlen = metadata.get('max_sequence')
        input = pad_sequences(sequences, maxlen = maxlen)
        return input

    # get pred on single label
    def get_label_pred(self, idx, val):
        legend = self.get_legend_inverted()
        return {legend[str(idx)]: val}

    # get top prediction
    def get_top_prediction(self, preds):
        idx_max = np.argmax(preds)
        return self.get_label_pred(idx_max, preds[idx_max])

    # predict function
    def predict_text(self, query: str, include_top=True, convert_to_json=True):
        
        # 1. Preprocessing
        sequence = self.get_sequences_from_text([query])
        x_input = self.get_input_from_sequences(sequence)
        
        # 2. Prediction
        preds = self.get_model().predict(x_input)[0]
        labeled_preds = [self.get_label_pred(i, x) for i, x in enumerate(list(preds))]

        # 3. Top prediction
        top_pred = self.get_top_prediction(preds)

        # create results dictionary
        results = {
            'predictions': labeled_preds
        }

        if include_top:
            results['top'] = top_pred

        if convert_to_json:
            print(type(encoders.encode_to_json(results, as_py=True)))
            print(type(encoders.encode_to_json(results, as_py=False)))
            results = encoders.encode_to_json(results, as_py=True)

        return results

def main():

    import pathlib

    # Where are the directories
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
    MODEL_DIR = BASE_DIR / 'models'
    SPAM_SMS_MODEL_DIR = MODEL_DIR / 'spam-sms'
    SPAM_SMS_MODEL_PATH = SPAM_SMS_MODEL_DIR / 'spam-model.h5'
    SPAM_SMS_TOKENIZER_PATH = SPAM_SMS_MODEL_DIR / 'spam-classifer-tokenizer.json'
    SPAM_SMS_METADATA_PATH = SPAM_SMS_MODEL_DIR / 'spam-classifer-metadata.json'

    model = AIModel(
        SPAM_SMS_MODEL_PATH, 
        SPAM_SMS_TOKENIZER_PATH, 
        SPAM_SMS_METADATA_PATH
    )

    print(model.get_model().summary())
    print(model.get_legend_inverted())
    print(model.predict_text('FREE MONEY GOLD GOLD GOLD CALL'))
    print('model build v1.0 succesfull')

if __name__ == '__main__':
    
    main()



