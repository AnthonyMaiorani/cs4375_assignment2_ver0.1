import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle
import torch.backends.mps

# Set the device to use CPU or GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

unk = '<UNK>'

class RNN(nn.Module):
    def __init__(self, input_dim, h):
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)
    
    def forward(self, input_vector):
        hidden, _ = self.rnn(input_vector)
        predicted_vector = self.W(hidden[-1])
        predicted_vector = self.softmax(predicted_vector)
        return predicted_vector

def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"] - 1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"] - 1)))
    return tra, val

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # Fix random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    # Lists to record training loss and validation accuracy
    training_losses = []
    validation_accuracies = []

    stopping_condition = False
    epoch = 0
    max_epochs = args.epochs  # Limit the number of epochs
    last_validation_accuracy = 0

    while not stopping_condition and epoch < max_epochs:
        random.shuffle(train_data)
        model.train()
        print("Training started for epoch {}".format(epoch + 1))
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0
        start_time = time.time()

        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(str.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding else word_embedding['unk'] for i in input_words]

                # Transform the input into required shape
                vectors = torch.tensor(vectors, dtype=torch.float32).view(len(vectors), 1, -1).to(device)
                output = model(vectors)

                # Get loss
                example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label], dtype=torch.long).to(device))

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.item()
            loss_count += 1
            loss.backward()
            optimizer.step()

        average_epoch_loss = loss_total / loss_count
        training_losses.append(average_epoch_loss)
        training_accuracy = correct / total
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {:.4f}".format(epoch + 1, training_accuracy))
        print("Training loss for epoch {}: {:.4f}".format(epoch + 1, average_epoch_loss))
        print("Training time for this epoch: {:.2f}s".format(time.time() - start_time))

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_start_time = time.time()
        val_loss_total = 0
        val_loss_count = 0
        print("Validation started for epoch {}".format(epoch + 1))

        with torch.no_grad():
            for input_words, gold_label in tqdm(valid_data):
                input_words = " ".join(input_words)
                input_words = input_words.translate(str.maketrans("", "", string.punctuation)).split()
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding else word_embedding['unk'] for i in input_words]

                vectors = torch.tensor(vectors, dtype=torch.float32).view(len(vectors), 1, -1).to(device)
                output = model(vectors)
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1

                # Compute loss
                example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label], dtype=torch.long).to(device))
                val_loss_total += example_loss.item()
                val_loss_count += 1

        val_accuracy = correct / total
        val_average_loss = val_loss_total / val_loss_count
        validation_accuracies.append(val_accuracy)
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {:.4f}".format(epoch + 1, val_accuracy))
        print("Validation loss for epoch {}: {:.4f}".format(epoch + 1, val_average_loss))
        print("Validation time for this epoch: {:.2f}s".format(time.time() - val_start_time))

        if val_accuracy < last_validation_accuracy:
            stopping_condition = True
            print("Training stopped to prevent overfitting.")
            print("Best validation accuracy: {:.4f}".format(last_validation_accuracy))
        else:
            last_validation_accuracy = val_accuracy

        epoch += 1

    # After training, print out the recorded metrics
    print("Training Losses:", training_losses)
    print("Validation Accuracies:", validation_accuracies)
