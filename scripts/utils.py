import os
import numpy as np
import pandas as pd
import datetime
import torch
import torch.nn as nn
import itertools
import matplotlib.pyplot as plt


def find_dir(number, path, name):
    for dirname in os.listdir(path):
        splitted = dirname.split("-")
        if splitted[0] != name:
            continue
        if (int(splitted[1]) < number <= int(splitted[2])):
            subpath = os.path.join(path, dirname)
            for subdirname in os.listdir(subpath):
                subsplitted = subdirname.split(name)
                if subsplitted[0] != "":
                    continue
                if int(subsplitted[1]) == number:
                    dest_path = os.path.join(subpath, subdirname)
                    for dest_file in os.listdir(dest_path):
                        if dest_file.split(".")[1] == "csv":
                            yield dest_path, dest_file


def read_data(file_path_X, file_path_y, seq_length):
    X_data = np.array(pd.read_csv(file_path_X, sep="\t", header=None, dtype=np.float32))
    y_data = np.array(pd.read_csv(file_path_y, sep="\t", header=None, dtype=np.int_))
    
    blocks = X_data.shape[0] / seq_length
    
    X_seq = np.array(np.split(X_data, blocks, axis=0))
    
    return X_seq, y_data


# NN models


class RnnClassifier(nn.Module):
    
    def __init__(self, input_dim, n_classes, lstm_hidden_dim=256, fc_hidden_dim=256, n_lstm_layers=2):
        super(RnnClassifier, self).__init__()
        
        self._lstm = nn.RNN(input_size=input_dim,
                             hidden_size=lstm_hidden_dim,
                             num_layers=n_lstm_layers,
                             batch_first=True)
        
        self._fc = nn.Sequential(nn.Linear(lstm_hidden_dim, fc_hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(fc_hidden_dim, n_classes))
        
    def forward(self, x):
        lstm_output, _ = self._lstm.forward(x)
        lstm_output = lstm_output[:, -1, :]
        fc_output = self._fc.forward(lstm_output)
        return fc_output


class LstmClassifier(nn.Module):
    
    def __init__(self, input_dim, n_classes, lstm_hidden_dim=256, fc_hidden_dim=256, n_lstm_layers=2):
        super(LstmClassifier, self).__init__()
        
        self._lstm = nn.LSTM(input_size=input_dim,
                             hidden_size=lstm_hidden_dim,
                             num_layers=n_lstm_layers,
                             batch_first=True)
        
        self._fc = nn.Sequential(nn.Linear(lstm_hidden_dim, fc_hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(fc_hidden_dim, n_classes))
        
    def forward(self, x):
        lstm_output, _ = self._lstm.forward(x)
        lstm_output = lstm_output[:, -1, :]
        fc_output = self._fc.forward(lstm_output)
        return fc_output

class GruClassifier(nn.Module):
    
    def __init__(self, input_dim, n_classes, lstm_hidden_dim=256, fc_hidden_dim=256, n_lstm_layers=2):
        super(GruClassifier, self).__init__()
        
        self._lstm = nn.GRU(input_size=input_dim,
                             hidden_size=lstm_hidden_dim,
                             num_layers=n_lstm_layers,
                             batch_first=True)
        
        self._fc = nn.Sequential(nn.Linear(lstm_hidden_dim, fc_hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(fc_hidden_dim, n_classes))
        
    def forward(self, x):
        lstm_output, _ = self._lstm.forward(x)
        lstm_output = lstm_output[:, -1, :]
        fc_output = self._fc.forward(lstm_output)
        return fc_output


# Training utils


def run_epoch(model, optimizer, criterion, batches, device, phase='train'):
    is_train = phase == 'train'
    if is_train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    n_predictions = 0
    
    correct_predictions = 0

    for X_batch, y_batch in batches:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        with torch.set_grad_enabled(is_train):
            y_pred = model.forward(X_batch)
            loss = criterion.forward(y_pred, y_batch)
    
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * y_batch.shape[0]
        correct_predictions += (torch.argmax(y_pred, dim=1) == y_batch).sum().item()
        n_predictions += y_batch.shape[0]

    epoch_loss = epoch_loss / n_predictions
    epoch_accuracy = correct_predictions / n_predictions

    return epoch_loss, epoch_accuracy


def train_model(model, optimizer, criterion, n_epoch, batch_size, train_dataset, val_dataset, backup_name, device):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = np.inf
    best_val_accuracy = np.inf
    best_epoch = 0
    
    train_batches = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_batches = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    for epoch in range(n_epoch):
        train_loss, train_accuracy = run_epoch(model, optimizer, criterion, train_batches, device, phase='train')
        val_loss, val_accuracy = run_epoch(model, optimizer, criterion, val_batches, device, phase='val')

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), backup_name)
            best_epoch = epoch

        print("Epoch: " + str(epoch))
        print("Train loss: " + str(train_loss) + ", accuracy: " + str(train_accuracy))
        print("Val loss: " + str(val_loss) + ", accuracy: " + str(val_accuracy) + "\n\n")
        
    return train_losses, train_accuracies, val_losses, val_accuracies, best_val_loss, best_val_accuracy, best_epoch

# Run prediction

def get_all_preds(model, batches, device, criterion):
    model.eval()
    all_preds = torch.tensor([]).to(device)
    correct_predictions = 0
    n_predictions = 0
    epoch_loss = 0.0
    inference_time = 0.0


    for X_batch, y_batch in batches:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        with torch.no_grad():
            start_time = datetime.datetime.now()
            y_pred = model.forward(X_batch)
            finish_time = datetime.datetime.now()
            delta = finish_time - start_time
            inference_time += delta.total_seconds()
            loss = criterion.forward(y_pred, y_batch)

        epoch_loss += loss.item() * y_batch.shape[0]
        correct_predictions += (torch.argmax(y_pred, dim=1) == y_batch).sum().item()
        n_predictions += y_batch.shape[0]
        all_preds = torch.cat(
            (all_preds, torch.argmax(y_pred, dim=1))
            ,dim=0
        )
    
    epoch_loss = epoch_loss / n_predictions
    epoch_accuracy = correct_predictions / n_predictions
    
    return all_preds, epoch_loss, epoch_accuracy, inference_time / n_predictions

# Draw plots

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        title = "Normalized confusion matrix"

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')