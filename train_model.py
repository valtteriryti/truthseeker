import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from torch.optim import AdamW as AdamWPT

label_mapping_5 = {
    'Disagree': 0,
    'Mostly Disagree': 1,
    'NO MAJORITY': 2,
    'Mostly Agree': 3,
    'Agree': 4
}

label_mapping_3 = {
    'Disagree': 0,
    'NO MAJORITY': 1,
    'Agree': 2
}

class TruthSeekerDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def load_data(label_column, label_mapping, frac=0.1):
    dataset_path = 'cleaned_dataset.csv'
    data = pd.read_csv(dataset_path).sample(frac=frac, random_state=42)
    texts = data['statement'].values
    labels = data[label_column].map(label_mapping).astype(int).values
    return texts, labels

def train_model(label_column):
    if label_column == '5_label_majority_answer':
        label_mapping = label_mapping_5
        num_labels = 5
    else:
        label_mapping = label_mapping_3
        num_labels = 3

    texts, labels = load_data(label_column, label_mapping)
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

    train_dataset = TruthSeekerDataset(train_encodings, train_labels)
    val_dataset = TruthSeekerDataset(val_encodings, val_labels)

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    optimizer = AdamWPT(model.parameters(), lr=5e-5)

    device = torch.device('mps')
    model.to(device)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    if len(class_weights) != num_labels:
        class_weights = np.ones(num_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, batch['labels'])
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Batch {batch_idx + 1}/{len(train_loader)} done")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

        model.save_pretrained("model")

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['labels'].cpu().numpy())

        print(classification_report(all_labels, all_preds, target_names=list(label_mapping.keys()), labels=list(label_mapping.values()), zero_division=0))

if __name__ == "__main__":
    train_model('3_label_majority_answer')