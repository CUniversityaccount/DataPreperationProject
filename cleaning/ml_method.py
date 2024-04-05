from transformers import BertTokenizer, BertForMaskedLM, AdamW
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from cleaning.lstm import LSTMAutoencoder, TextDataset, find_spelling_errors, train_lstm
from cleaning.bert_lora import LoRABert, fine_tune_model, BertModel, CustomDataset
from sklearn.model_selection import train_test_split

def predict_masked_word(df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    model = torch.load('ckpt/lstm_model.pth').to(device)

    misspelled_words = find_spelling_errors(df, tokenizer, model, device)

    misspelled_words = []
    for index, word in misspelled_words:
        misspelled_words.append((index, word))

    return misspelled_words

def mask_spelling_errors(df, misspelled_words):
    for index, word in misspelled_words:
        text = df.at[index, 'reviewText']
        corrected_text = text.replace(word, '[MASK]')
        df.at[index, 'reviewText'] = corrected_text
    return df

def mlm_correct_spelling(df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    bert_model = BertForMaskedLM.from_pretrained('bert-base-cased')
    bert_model.to(device)
    bert_model.eval()

    lora_model = LoRABert.from_pretrained('ckpt/lora_model.pth')
    lora_model.to(device)
    lora_model.eval()

    for index, row in df.iterrows():
        review_text = row['reviewText']
        inputs = tokenizer(
            review_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ) 
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            completed_text = lora_model(input_ids=input_ids, attention_mask=attention_mask)

        df.at[index, 'reviewText'] = completed_text

    return df

if __name__ == "__main__":
    
    df_1 = pd.read_json("data/corrupted_test.json",
                #   compression="gzip", 
                  lines=True)
    df_2 = pd.read_json("data/test.json",
                #   compression="gzip", 
                  lines=True)
    
    train_lstm(df_1)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    model = BertForMaskedLM.from_pretrained('bert-base-cased')

    num_classes = 2
    lora_model = LoRABert(model, num_classes)

    texts = df_1['reviewText'].tolist()
    labels = df_2['label'].tolist()
    max_length = 512
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)

    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(lora_model.parameters(), lr=2e-5)

    epochs = 200

    fine_tune_model(lora_model, train_loader, val_loader, criterion, optimizer, epochs)

    for index, row in df_1.iterrows():
        review_text = row['reviewText']

        inputs = tokenizer(review_text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            completed_text = lora_model(input_ids, attention_mask)
        df_1.at[index, 'reviewText'] = completed_text
    
    df = mlm_correct_spelling(df_1)
    
    pd.to_json(df, "data/clean_psc_test.json", lines=True)
    
    # sentence = "I love to work in ktchen and I like to cok."

    # corrected_sentence = mlm_correct_spelling(sentence)
    # print("Original sentence:", sentence)
    # print("Corrected sentence:", corrected_sentence)
