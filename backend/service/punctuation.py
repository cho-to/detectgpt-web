import torch
import tqdm
import re
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
from transformers import BertTokenizerFast
from service.model import CNNLstmBert

segment_size = 100
output_size = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'model/model_best'

class PunctuationGenerator:
    def __init__(self):
        print('*setting model*')
        self.model = CNNLstmBert(output_size).to(DEVICE)
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese', do_lower_case=True)
        self.model.eval()
        saved_checkpoint = torch.load(MODEL_PATH)
        self.model.load_state_dict(saved_checkpoint, strict=False)
    
    async def preprocessing(self, data):
        # replace undefined punctuation to defined punctuation
        data = data.replace("〈", "《").replace("〉", "》").replace('（', '(').replace('）',')').replace('+','')
        data = data.replace('【','(').replace('】', ')').replace('[', '(').replace(']', ')').replace('〔', '(').replace('〕', ')')
        data = data.replace('《 文摘报 》', '').replace('文摘报', '').replace('■', '').replace('/','').replace('~','').replace('●','')
        data = data.replace(' ', '').replace('　', '').replace('\u3000', '').replace('\xa0', '').replace('\n','')
        data = data.replace('‘','“').replace('’','”').replace('？', '?').replace('！', '!').replace("，", ",").replace("：", ":").replace("；", ";")

        # remove () and inside
        data = re.sub(pattern=r'\([^)]*\)', repl='', string=data)

        return data

    async def generate(self, data):
        # tokenizing model for model input
        words = self.tokenizer.encode(data)
        words = words[1:-1]
        words = torch.tensor([words]).long()

        punc_dec = ['', ',', '。','!','?',';',':','“','”','…','—','、','·','《','》']
        # generate output
        with torch.no_grad():
            words = words.to(DEVICE)
            _, output = self.model(words, DEVICE)
            output = list(output.argmax(dim=1).cpu().data.numpy().flatten())
            print(output)
        
        # decoding punctuation
        result = ''
        for char, punc in zip(list(data), output):
            result += char + punc_dec[punc]

        return result