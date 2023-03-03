import os.path
import time
from torch.optim import AdamW
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import os
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import  DebertaV2Tokenizer
import torch
from transformers import get_scheduler
import numpy as np
import random

"""
지워야될 태그들 
- c/ : 휴지구간이 확보되지 않은 연속발성(0.3초 미만) 

- n/ : 발성 이외의 단발적인 소음이 포함된 음성데이터

- N/ : 음성 구간의 50% 이상 잡음이 포함된 음성데이터

- u/ : 단어의 내용을 알아 들을 수 없는 음성데이터

-  l/ : 발성중 '음음 소리가 포함된 상황 (small 'L')

- b/ : 발성 중 숨소리, 김침 소리가 포함된 음성데이터

- * : 단어 중 일부만 알아 듣거나, 알아들었으나 정확하지 않은 음성데이터

- + : 발성 중 말을 반복적으로 더듬는 음성데이터

- / : 간투사
"""


class TextDataset(Dataset):
    def __init__(self,df):
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

        self.utterance = [self.tokenizer(text,return_tensors='pt', max_length=256, padding="max_length")for text in df["Utterance"]]
        self.speaker = [self.tokenizer(text,return_tensors='pt',max_length=256, padding="max_length")for text in df["Speaker"]]
        self.speaker_utterance_base = [f"{speaker}: {utterance}" for speaker, utterance in
                                  zip(df["Speaker"], df["Utterance"])]
        self.speaker_utterance = [self.tokenizer(text,return_tensors='pt',max_length=256, padding="max_length")for text in self.speaker_utterance_base]

        self.labels = {'surprise':0, 'anger':1, 'neutral':2, 'joy':3, 'sadness':4, 'fear':5, 'disgust':6}
        self.df_labels = df["Emotion"]

    def __len__(self):
        return len(self.utterance)

    def __getitem__(self, item):
        texts = self.speaker_utterance[item]
        labels = self.labels[self.df_labels[item]]
        return texts, labels

class TextLoader(DataLoader):
    def __init__(self):
        self.train_df = pd.read_csv('train.csv', encoding="cp949")
        self.dev_df = pd.read_csv('dev.csv', encoding="cp949")

        self.train_data = TextDataset(self.train_df)
        self.dev_data = TextDataset(self.dev_df)

    def get_train(self):
        super(TextLoader,self).__init__(self.train_data,
                                      batch_size=4,
                                      shuffle=False)

    def get_dev(self):
        super(TextLoader,self).__init__(self.dev_data,
                                    batch_size=4,
                                    shuffle=False)


class CustomDataset(Dataset):
    def __init__(self, data, model_path, mode="train"):
        self.dataset = data
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.mode = mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx == 0:  # 첫번째 시작지점 0에서 -1하면 Value Error 뜨니 예외처리
            text = f'[SEP]\n' \
                   f'{self.dataset["Speaker"][idx]}: {self.dataset["Utterance"][idx]}[SEP]\n' \
                   f'{self.dataset["Speaker"][idx + 1]}: {self.dataset["Utterance"][idx + 1]}'

        else:
            try:
                # 앞뒤 대화가 같은 대화면
                if self.dataset['Dialogue_ID'][idx] == self.dataset['Dialogue_ID'][idx - 1] and \
                        self.dataset['Dialogue_ID'][idx] == self.dataset['Dialogue_ID'][idx + 1]:
                    text = f'{self.dataset["Speaker"][idx - 1]}: {self.dataset["Utterance"][idx - 1]}[SEP]\n' \
                           f'{self.dataset["Speaker"][idx]}: {self.dataset["Utterance"][idx]}[SEP]\n' \
                           f'{self.dataset["Speaker"][idx + 1]}: {self.dataset["Utterance"][idx + 1]}'

                # 맨 처음 대화 시작이라면
                elif self.dataset['Dialogue_ID'][idx] != self.dataset['Dialogue_ID'][idx - 1] and \
                        self.dataset['Dialogue_ID'][idx] == self.dataset['Dialogue_ID'][idx + 1]:
                    text = f'[SEP]\n' \
                           f'{self.dataset["Speaker"][idx]}: {self.dataset["Utterance"][idx]}[SEP]\n' \
                           f'{self.dataset["Speaker"][idx + 1]}: {self.dataset["Utterance"][idx + 1]}'

                # 대화의 끝이라면
                elif self.dataset['Dialogue_ID'][idx] == self.dataset['Dialogue_ID'][idx - 1] and \
                        self.dataset['Dialogue_ID'][idx] != self.dataset['Dialogue_ID'][idx + 1]:
                    text = f'[SEP]\n' \
                           f'{self.dataset["Speaker"][idx]}: {self.dataset["Utterance"][idx]}[SEP]\n' \
                           f'{self.dataset["Speaker"][idx + 1]}: {self.dataset["Utterance"][idx + 1]}'

                # 대화른 혼자 하는거면 (Dialogue가 한개일때)
                elif self.dataset['Dialogue_ID'][idx] != self.dataset['Dialogue_ID'][idx - 1] and \
                        self.dataset['Dialogue_ID'][idx] != self.dataset['Dialogue_ID'][idx + 1]:
                    text = f'[SEP]\n' \
                           f'{self.dataset["Speaker"][idx]}: {self.dataset["Utterance"][idx]}[SEP]\n' \
                           f''

            except:  # 인덱스가 끝이라서 Index Error나면
                text = f'{self.dataset["Speaker"][idx - 1]}: {self.dataset["Utterance"][idx - 1]}[SEP]\n' \
                       f'{self.dataset["Speaker"][idx]}: {self.dataset["Utterance"][idx]}[SEP]\n' \
                       f''

        if self.mode == "train":
            inputs = self.tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
            inputs['labels'] = self.dataset['Target'][idx]
            input_ids = inputs['input_ids'][0]
            attention_mask = inputs['attention_mask'][0]
            y = self.dataset['Target'][idx]
            return input_ids, attention_mask, y
        else:
            inputs = self.tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
            input_ids = inputs['input_ids'][0]
            attention_mask = inputs['attention_mask'][0]
            return input_ids, attention_mask


pd.set_option('display.width', 270)
if __name__ == "__main__" :
    path = "../KEMDy20_v1_1/annotation/Sess01_eval.csv"
    df = pd.read_csv(path)
    print(df.columns)
    columns = df.columns
    print(df['Total Evaluation'][:5])
    print(df['Segment ID'][:5])
    print(df[columns[5]])
    print(df[' .1'][:5].drop(index=0))
