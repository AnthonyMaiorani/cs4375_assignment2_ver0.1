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
from argparse import ArgumentParser
import torch.backends.mps

# Set the device to use CPU or GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=1)  # Specified dimension for batch processing
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        hidden = self.W1(input_vector)
        hidden = self.activation(hidden)
        predicted_vector = self.W2(hidden)
        predicted_vector = self.softmax(predicted_vector)
        return predicted_vector

# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val


def validate(model, valid_data, batch_size=32):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    start_time = time.time()
    
    # Process in batches
    batches = [valid_data[i:i + batch_size] for i in range(0, len(valid_data), batch_size)]
    
    with torch.no_grad():  # Disable gradient computation
        for batch in tqdm(batches):
            # Batch processing
            vectors = torch.stack([x[0] for x in batch]).to(device)
            labels = torch.tensor([x[1] for x in batch]).to(device)
            
            # Forward pass
            outputs = model(vectors)
            loss = model.compute_Loss(outputs, labels)
            
            # Calculate accuracy
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
    
    accuracy = correct / total
    avg_loss = total_loss / len(batches)
    
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation loss: {avg_loss:.4f}")
    print(f"Validation time: {time.time() - start_time:.2f}s")
    
    return accuracy, avg_loss


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # Load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)

    # Initialize model and move to device
    model = FFNN(input_dim=len(vocab), h=args.hidden_dim)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Lists to record training loss and validation accuracy
    training_losses = []
    validation_accuracies = []

    print("========== Training for {} epochs ==========".format(args.epochs))
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        correct = 0
        total = 0
        start_time = time.time()
        epoch_loss = 0  # To accumulate loss over the epoch

        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data)  # Good practice to shuffle order of training data
        minibatch_size = 16
        N = len(train_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            batch_inputs = []
            batch_labels = []
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                batch_inputs.append(input_vector)
                batch_labels.append(gold_label)
            batch_inputs = torch.stack(batch_inputs).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

            predicted_vectors = model(batch_inputs)
            loss = model.compute_Loss(predicted_vectors, batch_labels)
            epoch_loss += loss.item()

            predicted_labels = torch.argmax(predicted_vectors, dim=1)
            correct += (predicted_labels == batch_labels).sum().item()
            total += batch_labels.size(0)

            loss.backward()
            optimizer.step()
        average_epoch_loss = epoch_loss / (N // minibatch_size)
        training_losses.append(average_epoch_loss)
        training_accuracy = correct / total
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {:.4f}".format(epoch + 1, training_accuracy))
        print("Training loss for epoch {}: {:.4f}".format(epoch + 1, average_epoch_loss))
        print("Training time for this epoch: {:.2f}s".format(time.time() - start_time))

        # Validation
        val_accuracy, val_loss = validate(model, valid_data)

        # Record validation accuracy
        validation_accuracies.append(val_accuracy)

    # After training, print out the recorded metrics
    print("Training Losses:", training_losses)
    print("Validation Accuracies:", validation_accuracies)
