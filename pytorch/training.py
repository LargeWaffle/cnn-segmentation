import time

import numpy as np
import torch
from tqdm import tqdm

from metrics import pixel_accuracy, mIoU


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(model, dataloaders, criterion, optimizer, scheduler, nb_class, device, epochs=15):
    train_loader, val_loader = dataloaders["train"], dataloaders["val"]

    train_losses = []
    test_losses = []

    val_iou = []
    val_acc = []

    train_iou = []
    train_acc = []

    lrs = []
    min_loss = np.inf
    decrease = 1
    not_improve = 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0

        # training loop
        model.train()

        for i, data in enumerate(train_loader):

            # training phase
            image_tiles, mask_tiles = data

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)

            # forward
            output = model(image)['out']
            soft = torch.nn.functional.softmax(output, dim=1)
            preds = torch.argmax(soft, dim=1).unsqueeze(1).float()

            loss = criterion(preds, mask)
            loss.requires_grad = True

            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            with torch.no_grad():
                # evaluation metrics
                iou_score += mIoU(preds, mask, nb_class)
                accuracy += pixel_accuracy(preds, mask)

            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()
        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0

            # validation loop
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    image_tiles, mask_tiles = data

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)

                    # forward
                    output = model(image)['out']
                    soft = torch.nn.functional.softmax(output, dim=1)
                    preds = torch.argmax(soft, dim=1).unsqueeze(1).float()

                    # loss
                    loss = criterion(preds, mask)
                    test_loss += loss.item()

                    # evaluation metrics
                    val_iou_score += mIoU(preds, mask, nb_class)
                    test_accuracy += pixel_accuracy(preds, mask)

            # calculate mean for each batch
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(val_loader))

            if min_loss > (test_loss / len(val_loader)):
                print('loss_decreasing {:.3f} >> {:.3f} '.format(min_loss, (test_loss / len(val_loader))))
                min_loss = (test_loss / len(val_loader))
                decrease += 1
                if decrease % 5 == 0:
                    print('saving model...')
                    torch.save(model, 'dlab_{:.3f}.pt'.format(val_iou_score / len(val_loader)))

            if (test_loss / len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = (test_loss / len(val_loader))
                print(f'Loss has not decreased for {not_improve} time(s)')
                if not_improve == 7:
                    print('Loss has not decreased for 7 times, training stopped')
                    break

            # iou
            val_iou.append(val_iou_score / len(val_loader))
            train_iou.append(iou_score / len(train_loader))
            train_acc.append(accuracy / len(train_loader))
            val_acc.append(test_accuracy / len(val_loader))

            print("Epoch:{}/{}".format(e + 1, epochs),
                  "train_loss: {:.3f}".format(running_loss / len(train_loader)),
                  "val_loss: {:.3f}".format(test_loss / len(val_loader)),
                  "train_mIoU:{:.3f}".format(iou_score / len(train_loader)),
                  "val_mIoU: {:.3f}".format(val_iou_score / len(val_loader)),
                  "train_acc:{:.3f}".format(accuracy / len(train_loader)),
                  "val_acc:{:.3f}".format(test_accuracy / len(val_loader)),
                  "Epoch time: {:.2f}m".format((time.time() - since) / 60))

    history = {'train_loss': train_losses, 'val_loss': test_losses,
               'train_miou': train_iou, 'val_miou': val_iou,
               'train_acc': train_acc, 'val_acc': val_acc, 'lrs': lrs}

    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))

    return model, history
