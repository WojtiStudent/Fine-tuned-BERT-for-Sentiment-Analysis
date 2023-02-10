import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from tqdm import tqdm

from collections import defaultdict


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model = model.train()

    losses = []
    n_examples = 0
    correct_predictions = 0
    for encodings, labels in tqdm(data_loader):
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        labels = labels.to(device)

        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        n_examples += len(labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()

    losses = []
    correct_predictions = 0
    n_examples = 0
    with torch.no_grad():
        for encodings, labels in tqdm(data_loader):
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            n_examples += len(labels)
            losses.append(loss.item())
    
    return correct_predictions.double() / n_examples, np.mean(losses)


def predict(mode, data_loader):
    model = model.eval()

    texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for encodings, labels in tqdm(data_loader):
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            texts.extend(encodings['text'])
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(labels)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return texts, predictions, prediction_probs, real_values


    
def training(model, train_data_loader, val_data_loader, loss_fn, optimizer, scheduler, device, n_epochs):
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1}/{n_epochs}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,    
            loss_fn, 
            optimizer, 
            device, 
            scheduler, 
        )


        print(f'Train Accuracy: {train_acc: .3f}')
        print(f'Train Loss: {train_loss: .5f}')

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn, 
            device
        )

        print(f'Val   Accuracy: {val_acc: .3f}')
        print(f'Val   Loss: {val_loss: .5f}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc
    
    return history




