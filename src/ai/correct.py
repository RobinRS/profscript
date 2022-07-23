from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cpu")
df = pd.read_csv('data/sentences.csv', on_bad_lines='skip', sep=';', encoding='utf8')
print(df.head())
print(df.info())

tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")

sample_txt = "Das it ein Tet."

tokens = tokenizer.tokenize(sample_txt)
print(tokens)

token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)

print(tokenizer.sep_token, tokenizer.sep_token_id)
print(tokenizer.cls_token, tokenizer.cls_token_id)
print(tokenizer.pad_token, tokenizer.pad_token_id)
print(tokenizer.unk_token, tokenizer.unk_token_id)

encoding = tokenizer.encode_plus(
                    sample_txt,
                    max_length=128,
                    truncation=True,
                    add_special_tokens=True,
                    padding='longest',
                    return_attention_mask=True,
                    return_token_type_ids=False,
                    return_tensors="pt")

print(encoding.keys())

print(encoding['input_ids'])
print(encoding['attention_mask'])

print(tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]))


## prep training data
class VLDataSet(Dataset):

  def __init__(self, texts, corrects, tokenizer, max_len):
    self.texts = texts
    self.corrects = corrects
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.texts)
  
  def __getitem__(self, item):
    text = str(self.incorrect[item])
    target = str(self.correct[item])

    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      truncation=True,
      return_token_type_ids=False,
      padding='longest',
      return_attention_mask=True,
      return_tensors='pt',
    )

    encoding_target = self.tokenizer.encode_plus(
      target,
      add_special_tokens=True,
      max_length=self.max_len,
      truncation=True,
      return_token_type_ids=False,
      padding='longest',
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'text_text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'target_text': target,
      'targets': encoding_target['input_ids'].flatten(),
      'attention_mask': encoding_target['attention_mask'].flatten(),
    }

MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 20

df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
df_validator, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

print(df_train.shape, df_validator.shape, df_test.shape)

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = VLDataSet(df.incorrect.to_numpy(), df.correct.to_numpy(), tokenizer=tokenizer, max_len=max_len)
    
    return DataLoader(ds, batch_size=batch_size, num_workers=1, shuffle=True)

BATCH_SIZE = 16

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
validator_data_loader = create_data_loader(df_validator, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

data = next(iter(train_data_loader))
data.keys()

print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)

PRETRAINED_MODEL_NAME = "bert-base-german-cased"

model = AutoModelForMaskedLM.from_pretrained(PRETRAINED_MODEL_NAME)

last_hidden_state, pooled_output = model(
  input_ids=encoding['input_ids'], 
  attention_mask=encoding['attention_mask']
)

print(last_hidden_state.shape)

class SequenceCorrection(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = AutoModelForMaskedLM.from_pretrained(PRETRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)


model = SequenceCorrection(len(class_names))
model = model.to(device)

