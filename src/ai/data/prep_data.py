# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv('data/articles.csv', on_bad_lines='skip', sep=';', encoding='utf8')

sentence_data = {
  'correct': [],
  'incorrect': [],
}

for i in range(len(df)):
  sentences = df["Daten"][i].split(".")
  for sentence in sentences:
      sentence = sentence.strip()
      sentence_data["correct"].append(sentence.strip())
      sentence_data["incorrect"].append(sentence.strip().lower().replace(": ", " ").replace(", ", " ").replace(".", " ").replace("\n", ""))

df_sentence = pd.DataFrame(sentence_data)
df_sentence.to_csv('data/sentences.csv', sep=';', index=False, encoding='utf8')   