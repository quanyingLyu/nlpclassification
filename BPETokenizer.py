from collections import OrderedDict, defaultdict
import pickle
import re
from tqdm import tqdm
import json
import nltk
from nltk.tokenize import word_tokenize


class BPETokenizer():
    def __init__(self):
        self.b2i=OrderedDict()
        self.i2b=OrderedDict()
        self.next_id = 0

        self.sp_s2i = {}
        self.sp_i2s = {}

    def pair_stats(self, tokens, stats):
        pass

    def train(self, tokens, voca_size):
        
        return
    

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            f.write(pickle.dumps((self.b2i, self.sp_s2i, self.next_id)))


    def load(self, file_name):
        with open(file_name, 'rb') as f:
            self.b2i, self.sp_s2i, self.next_id = pickle.load(f)
        self.i2b = {v:k for k, v in self.b2i.items()}
        self.sp_i2s = {v:k for k, v in self.sp_s2i.items()}



def processText(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'<.*?>', '', text)

    words = word_tokenize(text.lower())

    words = [w for w in words if w not in stopwords]

    return ' '.join(words)



def get_text(evidence, train, progress):
    text = ''
    values_list = list(evidence.values())
    half_values = values_list[:len(values_list) // 3]
    for value in half_values:
        text += processText(value)
        progress.update(1)
    for value in train.values():
        text += processText(value['claim_text'])
        progress.update(1)
    return text



if __name__ == '__main__':
    stopwords = nltk.corpus.stopwords.words('english')
    with open("COMP90042_2024-main\\data\\train-claims.json") as f:
        train_claims_text = json.load(f)

    with open("COMP90042_2024-main\\data\\evidence.json") as f:
        evidence_claims_text = json.load(f)

    progress = tqdm(total=len(evidence_claims_text.values())//25 + len(train_claims_text.values()))
    text = get_text(evidence_claims_text, train_claims_text, progress)

    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=5000)

    tokenizer.add_special_tokens(['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASKED]'])

    tokenizer.save('bbpe.bin')

    tokenizer = BPETokenizer()
    tokenizer.load('bbpe.bin')

    ids = tokenizer.encode('hello world')
    print(ids)
    print(tokenizer.decode(ids))



