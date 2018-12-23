import json
import re
import string
from json import JSONDecodeError
from pyvi import ViTokenizer, ViPosTagger
from ftfy import fix_text


def read_data_from_file(file_name):
    x, y = [], []
    pos_count = 0
    with open(file_name) as lines:
        for line in lines:
            try:
                sent = json.loads(line)
                star = sent['star']
                comment = sent['comment']
                if star > 3:
                    if pos_count > 540:
                        continue
                    pos_count += 1
                    sentiment = 1
                elif star < 3:
                    sentiment = 0
                else:
                    continue
                x.append(preprocess_text(comment))
                y.append(sentiment)
            except TypeError:
                print('type error')
            except JSONDecodeError:
                print('decode error')
    print('Collected records: ' + str(len(x)))
    return x, y


def preprocess_text(text):
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)
    text = fix_text(text)
    re.sub('^[0-9]+', '', text)
    table = str.maketrans({key: ' ' for key in string.punctuation})
    text = text.translate(table)
    text = text.lower()
    return ViTokenizer.tokenize(text)


def load_dataset():
    return read_data_from_file('./data/finalData')
