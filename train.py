import json 
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words
import torch
from model import Net
from torch.utils.data import Dataset, DataLoader

with open('intents.json', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)  # CrossEntropyLoss

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    #dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# Hiperparámetros
BATCH_SIZE = 6
INPUT_SIZE = len(X_train[0])
HIDDEN_SIZE = 8
OUTPUT_SIZE = len(tags)
LEARNING_RATE = 1e-3
EPOCHS = 1000
WORKERS = 0

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(device)

# loss y optimizador
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backprop y update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1}/{EPOCHS}, loss={loss.item():.4f}')

print(f'final loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": INPUT_SIZE,
    "output_size": OUTPUT_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print("Entrenamiento completo. Archivo guardado como "+ FILE)

