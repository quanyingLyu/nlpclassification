from collections import OrderedDict
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
        for i in range(len(tokens)-1):
            new_token = tokens[i] + tokens[i+1]
            if new_token in stats:
                stats[new_token] += 1
            else:
                stats[new_token] = 1


    def merge_pair(self, tokens, new_token):
        merged_token = []
        i = 0
        while i < len(tokens):
            if i+1 < len(tokens) and tokens[i] + tokens[i+1] == new_token:
                merged_token.append(tokens[i] + tokens[i+1])
                i += 2
            else:
                merged_token.append(tokens[i])
                i += 1
        return merged_token


    def train(self, text, vocab_size):
        for i in range(256):
            self.b2i[bytes([i])] = i
        self.next_id = 256

        token_list = [bytes([c]) for c in text.encode('utf-8')]
        progress = tqdm(total=vocab_size-256)
        
        while True:
            if self.next_id > vocab_size:
                
                break
            
            stats = {}
            self.pair_stats(token_list, stats)

            if len(stats) == 0:
                break
            
            new_token = max(stats, key=stats.get)
            new_token_list = self.merge_pair(token_list, new_token)
            token_list = new_token_list
            self.b2i[new_token] = self.next_id
            self.next_id += 1

            progress.update(1)

        self.i2b = {v: k for k, v in self.b2i.items()}


    def voca_size(self):
        return self.next_id
    

    def vocabulary(self):
        return self.b2i
    

    def special_voca(self):
        return self.sp_s2i


    def add_special_tokens(self, special_tokens):
        for token in special_tokens:
            self.sp_s2i[token] = self.next_id
            self.next_id += 1
        self.i2s = {v: k for k, v in self.sp_s2i.items()}


    def encode(self, text):
        encode_id = []

        pattern = '(' + '|'.join([re.escape(tok) for tok in self.sp_s2i.keys()]) + ')'

        splits = re.split(pattern, text)

        for sub_text in splits:
            if sub_text in self.sp_s2i:
                encode_id.append(self.sp_s2i[sub_text])
            else:
                tokens = [bytes([c]) for c in sub_text.encode('utf-8')]
                while True:
                    stats = {}
                    self.pair_stats(tokens, stats)

                    if len(stats) == 0:
                        break

                    new_token = None
                    for merged_token in stats:
                        if merged_token in self.b2i and (new_token is None or stats[merged_token] > stats[new_token]):
                            new_token = merged_token

                    if new_token is None:
                        break

                    tokens = self.merge_pair(tokens, new_token)
                
                encode_id.extend([self.b2i[tok] for tok in tokens])
            
        return encode_id


    def decode(self, ids):
        bytes_list = []
        for id in ids:
            if id in self.sp_i2s:
                bytes_list.append(self.sp_i2s[id].encode('utf-8'))
            else:
                bytes_list.append(self.i2b[id])
        return  b''.join(bytes_list).decode('utf-8', errors='replace')


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
    half_values = values_list[:len(values_list) // 25]
    for value in half_values:
        text += processText(value)
        progress.update(1)
    for value in train.values():
        text += processText(value['claim_text'])
        progress.update(1)
    return text


if __name__ == "__main__":
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