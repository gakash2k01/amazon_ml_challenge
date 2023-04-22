import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from tqdm import tqdm

# Load the T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small', model_max_length=1024)
model = T5ForConditionalGeneration.from_pretrained('t5-small').to('cuda:0')
BATCH_SIZE = 128 * 4
# Define a function to summarize a batch of texts
def summarize_batch(texts):
  # Tokenize the input texts
  inputs = tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True, truncation=True).to('cuda')
  # Generate the summaries
  summaries = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=50, num_beams=4)
  # Decode the summaries
  decoded_summaries = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
  return decoded_summaries

# Define a function to summarize a dataframe in batches
def summarize_dataframe(df, batch_size=8):
  # Split the dataframe into batches
  batches = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]
  # Summarize each batch and concatenate the results
  summaries = []
  for batch in tqdm(batches):
    texts = list(batch)
    batch_summaries = summarize_batch(texts)
    summaries.extend(batch_summaries)
  return summaries

# Load the input dataframe
print("Reading data.")
df = pd.read_csv('dataset/train.csv')
df = df.dropna(axis = 0)

# Summarize the dataframe in batches
print("Summarizing Title.")
title = summarize_dataframe(df['TITLE'], batch_size=BATCH_SIZE)
print("Summarizing Title.")
df['TITLE'] = title
BULLET_POINTS = summarize_dataframe(df['BULLET_POINTS'], batch_size=BATCH_SIZE)
print("Summarizing Title.")
df['BULLET_POINTS'] = BULLET_POINTS
DESCRIPTION = summarize_dataframe(df['DESCRIPTION'], batch_size=BATCH_SIZE)
df['DESCRIPTION'] = DESCRIPTION
# Add the summaries to the dataframe

# Save the output dataframe
df.to_csv('train_updated.csv', index=False)
