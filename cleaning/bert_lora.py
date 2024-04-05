import torch, tqdm
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.model_selection import train_test_split

class LoRALayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(LoRALayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = nn.Parameter(torch.Tensor(output_size, input_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=1)

    def forward(self, input):
        output = input.matmul(self.weights.t())
        return output

class LoRABert(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(LoRABert, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.lora_layer = LoRALayer(self.bert.config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.lora_layer(pooled_output)
        probabilities = self.softmax(logits)
        return probabilities

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def fine_tune_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = correct / total

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}')
    
    torch.save(model.state_dict(), 'ckpt/lora_model.pth')

    # def find_spelling_errors(dataframe, tokenizer, model, device):
    #     misspelled_words = []
    #     for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
    #         text = row['reviewText']
    #         inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    #         inputs = inputs.to(device)

    #         reconstructed = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])[0]

    #         original_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    #         reconstructed_tokens = tokenizer.convert_ids_to_tokens(torch.argmax(reconstructed, dim=-1)[0])

    #         for i, (original, reconstructed) in enumerate(zip(original_tokens, reconstructed_tokens)):
    #             if original != reconstructed and original != '[PAD]' and reconstructed != '[PAD]':
    #                 misspelled_words.append((index, original, i))

    #     return misspelled_words

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased')
    num_classes = 2  # Example: binary classification
    lora_model = LoRABert(model, num_classes)
    
    df_1 = pd.read_json("data/corrupted_test.json",
                #   compression="gzip", 
                  lines=True)
    df_2 = pd.read_json("data/test.json",
                #   compression="gzip", 
                  lines=True)

    # Fine-tuning dataset
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
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased')
    num_classes = 2
    lora_model = LoRABert(model, num_classes)

    input_text = "Your input text here."
    inputs = tokenizer(input_text, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = lora_model(input_ids, attention_mask)
        predicted_class = torch.argmax(outputs, dim=1).item()
        print("Predicted class:", predicted_class)