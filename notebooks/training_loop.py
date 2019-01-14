from datetime import datetime
import torch
import os

def train_model(name, model, dataloaders, criterion, optimizer, device, num_epochs=25):
    best_acc = 0.0
    
    if not os.path.exists('results'):
        os.mkdir('results')
    f = open(f'results/{name}.txt', 'w', buffering=1)


    for epoch in range(num_epochs):
        train_loss = None
        train_accuracy = None

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                train_loss = epoch_loss
                train_accuracy = epoch_acc
            else:
                stats = f'Epoch: {epoch}, TL: {train_loss:.5f}, VL: {epoch_loss:.5f}, '\
                        f'TA: {train_accuracy:.5f}, VA: {epoch_acc:.5f}'
                print(stats)
                f.write(f'{stats}\n')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), f'{name}_{best_acc:.5f}.pth')

def train_model_multi_class(name, model, dataloaders, criterion, optimizer, device, num_epochs=25, calc_accuracy=False):
    best_loss = 100000000.0
    
    if not os.path.exists('results'):
        os.mkdir('results')
    f = open(f'results/{name}.txt', 'w', buffering=1)


    for epoch in range(num_epochs):
        train_loss = None
        train_accuracy = None

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                if epoch % 20 == 0:
                    shape = outputs.shape
                    for i in range(shape[0]):
                        for j in range(shape[1]):
                            if not ((outputs[i][j] >= 0.9 and labels[i][j] >= 0.9) or
                                    (outputs[i][j] < 0.9 and labels[i][j] < 0.9)):
                                break
                        else:
                            running_corrects += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            if phase == 'train':
                train_loss = epoch_loss
                train_accuracy = epoch_acc
            else:
                stats = f'Epoch: {epoch}, TL: {train_loss:.5f}, VL: {epoch_loss:.5f}'
                if epoch % 20 == 0:
                    stats += f', TA: {train_accuracy:.5f}, VA: {epoch_acc:.5f}'
                print(stats)
                f.write(f'{stats}\n')

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), f'{name}_{best_loss:.5f}.pth')
