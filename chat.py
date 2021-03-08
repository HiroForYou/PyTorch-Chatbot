import random
import json
import torch
from model import Net
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', encoding='utf-8') as f:
    intents = json.load(f)

BOT_NAME = "Ivi"
FILE = "data.pth"
data = torch.load(FILE)
INPUT_SIZE = data["input_size"]
HIDDEN_SIZE = data["hidden_size"]
OUTPUT_SIZE = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = Net(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(device)
model.load_state_dict(model_state)
model.eval()

def getResponse(inputText):
    sentence = tokenize(inputText)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    
    output = model(X)
    
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    else:
        return f"No los entiendos.."
    


