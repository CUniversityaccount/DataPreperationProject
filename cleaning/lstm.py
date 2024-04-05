import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from tqdm import tqdm

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data[idx])
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return inputs['input_ids'].squeeze(0)

def find_spelling_errors(dataframe, tokenizer, model, device):
    misspelled_words = []
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        text = row['reviewText']
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = inputs.to(device)

        reconstructed = model(inputs).squeeze(0)

        original_ids = inputs['input_ids'].squeeze(0)
        reconstructed_ids = torch.argmax(reconstructed, dim=-1)

        original_tokens = tokenizer.convert_ids_to_tokens(original_ids)
        reconstructed_tokens = tokenizer.convert_ids_to_tokens(reconstructed_ids)

        for original, reconstructed in zip(original_tokens, reconstructed_tokens):
            if original != reconstructed and original != '[PAD]' and reconstructed != '[PAD]':
                misspelled_words.append((index, original))
    return misspelled_words

def train_lstm():
    df = pd.read_json("data/corrupted_train.json",
                #   compression="gzip", 
                  lines=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    input_size = tokenizer.vocab_size
    hidden_size = 128
    num_layers = 2

    model = LSTMAutoencoder(input_size, hidden_size, num_layers).to(device)

    max_length = 1000
    dataset = TextDataset(df['reviewText'], tokenizer, max_length)
    data_loader = DataLoader(dataset, batch_size=1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs in data_loader:
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(data_loader)}")

    torch.save(model.state_dict(), "ckpt/lstm_model.pth")

if __name__ == "__main__":
    df = pd.read_json("data/corrupted_test.json",
                #   compression="gzip", 
                  lines=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    input_size = tokenizer.vocab_size
    hidden_size = 128
    num_layers = 2

    model = LSTMAutoencoder(input_size, hidden_size, num_layers).to(device)

    max_length = 1000
    dataset = TextDataset(df['reviewText'], tokenizer, max_length)
    data_loader = DataLoader(dataset, batch_size=1)

    misspelled_words = find_spelling_errors(df, tokenizer, model, device)

    output_list = []
    for index, word in misspelled_words:
        output_list.append((index, word))