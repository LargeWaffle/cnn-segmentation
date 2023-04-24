import copy
import time

import torch

from metrics import pixel_accuracy, mIoU


def train_model(model, dataloaders, criterion, optimizer, nb_class, device, epochs=15):
    model = model.to(device)

    since = time.time()

    train_acc_history = []
    train_loss_history = []
    train_score_history = []

    val_acc_history = []
    val_loss_history = []
    val_score_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0
            running_acc = 0
            running_miou = 0

            # Iterate over data.
            for images, masks in dataloaders[phase]:

                images = images.to(device)
                masks = masks.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(images)['out']

                    loss = criterion(outputs, masks.squeeze(1))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                preds = torch.argmax(outputs, dim=1).unsqueeze(1).float()

                # statistics
                running_loss += loss.item() * images.size(0)
                running_acc += pixel_accuracy(preds, masks)
                running_miou += mIoU(preds, masks, nb_class)

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_acc / len(dataloaders[phase]) * 100
            epoch_miou = running_miou / len(dataloaders[phase]) * 100

            print('{} loss: {:.4f} acc: {:.2f}% mIoU {:.2f}%'.format(phase, epoch_loss, epoch_acc, epoch_miou))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
                val_score_history.append(epoch_miou)
            else:
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
                train_score_history.append(epoch_miou)

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    metrics = {
        "acc": {"train": train_acc_history, "val": val_acc_history},
        "loss": {"train": train_loss_history, "val": val_loss_history},
        "score": {"train": train_score_history, "val": val_score_history}
    }

    return model, metrics
