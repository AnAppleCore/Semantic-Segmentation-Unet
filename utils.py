import os
import cv2
import sys
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from PIL import Image
from dataset import Cityscapes_Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2


class Metrics(object):
    def __init__(self):
        self.val = 0.0
        self.sum = 0.0
        self.cnt = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class Iou(object):
    def __init__(self, classLabels, validClasses, voidClass):
        self.classLabels    = classLabels
        self.validClasses   = validClasses
        self.voidClass      = voidClass
        self.evalClasses    = [c for c in validClasses if c != voidClass]

        self.perImgStats    = []
        self.pixelCnt       = 0
        # confusion matrix 
        self.M              = np.zeros(shape=(len(self.validClasses), len(self.validClasses)), dtype=np.ulonglong)

    def get_iou(self, label):
        ''' Get the Iou score for a specific label (train id)'''
        if label == self.voidClass:
            return float('nan')

        true_pos = np.longlong(self.M[label, label])
        false_neg = np.longlong(self.M[label, :].sum()) - true_pos

        other_labels = [c for c in self.evalClasses if c != label]
        false_pos = np.longlong(self.M[other_labels, label].sum())

        union = true_pos + false_neg + false_pos
        if union == 0:
            return float('nan')
        else:
            return float(true_pos) / union

    def update(self, preds, labels):
        '''update the Iou within each batch step'''
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        for img_id in range(preds.shape[0]):
            pred = preds[img_id, :, :]
            trueth = labels[img_id, :, :]

            W = pred.shape[0]
            H = pred.shape[1]
            res = W*H

            max_val = max(trueth.max(), pred.max()).astype(np.int32)+1
            encoded = (trueth.astype(np.int32)*max_val) + pred
            values, cnts = np.unique(encoded, return_counts=True)
            for value, cnt in zip(values, cnts):
                pred_id = value % max_val
                true_id = int((value-pred_id)/max_val)
                self.M[true_id][pred_id] += cnt

            not_ignored_px = np.in1d(trueth, self.evalClasses, invert=True).reshape(trueth.shape)
            wrong_px = np.logical_and(not_ignored_px, (pred!=trueth))
            not_ignored_px_cnt = np.count_nonzero(not_ignored_px)
            wrong_px_cnt = np.count_nonzero(wrong_px)
            self.perImgStats.append([not_ignored_px_cnt, wrong_px_cnt])
            self.pixelCnt += res

        return

    def get_mean(self):
        '''Calculate the miou within each epoch'''
        iou_list = []

        print('classes\tIoU score')
        for c in self.evalClasses:
            iou = self.get_iou(c)
            iou_list.append(iou)
            print('{:<14}:\t {:>5.3f}'.format(self.classLabels[c], iou))
        iou_sum = 0.0
        valid_cnt = 0
        for iou in iou_list:
            if not np.isnan(iou):
                valid_cnt += 1
                iou_sum += iou
            if valid_cnt == 0:
                return float('nan')
        miou = iou_sum / valid_cnt
        print('Mean IoU:\t {avg:5.3f}'.format(avg=miou))
        return miou


def get_dataloaders(args):

    train_transform = A.Compose([
        A.Resize(height=args.img_size[0], width=args.img_size[1], interpolation=cv2.INTER_NEAREST),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
    ],)

    val_transform = A.Compose([
        A.Resize(height=args.img_size[0], width=args.img_size[1], interpolation=cv2.INTER_NEAREST),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
    ],)

    if args.transform is False:
        train_transform = None
        val_transform = None

    batch_size = args.batch_size
    num_workers = args.num_workers
    pin_memory = args.pin_memory

    train_dataset = Cityscapes_Dataset(root='data', split='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=pin_memory)
    val_dataset = Cityscapes_Dataset(root='data', split='val', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_dataset = Cityscapes_Dataset(root='data', split='test', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader

def train(train_loader, model, loss_func, optimizer, scheduler, epoch, voidClass, args):

    train_loss = Metrics()
    train_acc = Metrics()
    res = args.img_size[0]*args.img_size[1]
    # set the model to train mode
    model.train()
    with torch.set_grad_enabled(True):
        for epoch_step, (inputs, labels, _) in enumerate(train_loader):

            inputs = inputs.float().cuda()
            labels = labels.long().cuda()
            step_size = inputs.size(0) # current batch_size

            optimizer.zero_grad()

            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            loss = loss_func(outputs, labels)

            loss.backward()
            optimizer.step()

            # update starics    
            loss = loss.item()
            correct_pixels = torch.sum(preds == labels.data)
            void_pixels = int((labels == voidClass).sum())
            acc = correct_pixels.double()/(step_size*res - void_pixels)

            train_loss.update(loss, step_size)
            train_acc.update(acc, step_size)

            if epoch_step % 100 == 0 or epoch_step <= 10:
                print('Train epoch: [{}] step [{}/{}] loss: {:.4e} acc: {:.3f}'.format(epoch, epoch_step, len(train_loader), train_loss.avg, train_acc.avg))

        scheduler.step(train_loss.avg)

    return train_loss.avg, train_acc.avg

def validate(val_loader, model, loss_func, epoch, classLabels, validClasses, voidClass, mask_colors, args):

    val_loss = Metrics()
    val_acc = Metrics()
    iou = Iou(classLabels, validClasses, voidClass)
    res = args.img_size[0]*args.img_size[1]
    # set the model to evaluate mode
    model.eval()
    with torch.no_grad():
        for epoch_step, (inputs, labels, filepath) in enumerate(val_loader):

            inputs = inputs.float().cuda()
            labels = labels.long().cuda()
            step_size = inputs.size(0) # current batch_size

            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            loss = loss_func(outputs, labels)

            # update statics
            loss = loss.item()
            correct_pixels = torch.sum(preds == labels.data)
            void_pixels = int((labels == voidClass).sum())
            acc = correct_pixels.double()/(step_size*res - void_pixels)

            val_loss.update(loss, step_size)
            val_acc.update(acc, step_size)
            iou.update(preds, labels)

            # save the visualization result of first batch
            if epoch_step == 0 and mask_colors is not None:
                for i in range(inputs.size(0)):
                    filename = os.path.splitext(os.path.basename(filepath[i]))[0]
                    if epoch == 0:
                        img = vis_img(inputs[i, :, :, :])
                        label = vis_label(labels[i, :, :, :], mask_colors)
                        if len(img.shape) == 3:
                            cv2.imwrite(args.save_path + '/images/{}.png'.format(filename),img[:,:,::-1])
                        else:
                            cv2.imwrite(args.save_path + '/images/{}.png'.format(filename),img)
                        cv2.imwrite(args.save_path + '/images/{}_gt.png'.format(filename),label[:,:,::-1])
                    # save predictions
                    pred = vis_label(preds[i, :, :], mask_colors)
                    cv2.imwrite(args.save_path + '/images/{}_epoch_{}.png'.format(filename,epoch),pred[:,:,::-1])

            if epoch_step % 100 == 0 or epoch_step <= 10:
                print('Valid epoch: [{}] step [{}/{}] loss: {:.4e} acc: {:.3f}'.format(epoch, epoch_step, len(val_loader), val_loss.avg, val_acc.avg))

    return val_loss.avg, val_acc.avg, iou.get_mean()

def predict(loader, model, mask_colors, mode='val', args=None):
    folder = args.save_path
    dataset = Cityscapes_Dataset
    print('------Predicting------')
    model.eval()
    with torch.no_grad():
        for epoch_step, batch in enumerate(loader):
            if len(batch) == 2:
                inputs, filepath = batch
            else:
                inputs, _, filepath = batch
            inputs = inputs.float().cuda()
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)

            # save the visualization result of first batch
            if epoch_step == 0 and mask_colors is not None:
                for i in range(inputs.size(0)):
                    filename = os.path.splitext(os.path.basename(filepath[i]))[0]
                    img = vis_img(inputs[i,:,:,:])
                    img = Image.fromarray(img, 'RGB')
                    img.save(folder + '/results_color_{}/{}_input.png'.format(mode, filename))
                    # Save prediction with color labels
                    pred = preds[i,:,:].cpu()
                    pred_color = vis_label(pred, mask_colors)
                    pred_color = Image.fromarray(pred_color.astype('uint8'))
                    pred_color.save(folder + '/results_color_{}/{}_prediction.png'.format(mode, filename))
                    # Save class id prediction
                    pred_id = dataset.trainid2id[pred]
                    pred_id = Image.fromarray(pred_id)
                    pred_id = pred_id.resize((2048,1024), resample=Image.NEAREST)
                    pred_id.save(folder + '/results_{}/{}.png'.format(mode, filename))

            if epoch_step % 100 == 0 or epoch_step <= 10:
                print('Predict step [{}/{}]'.format(epoch_step, len(loader)))

    print('Predict Complete!')

def vis_img(img):
    img = img.cpu()
    # Convert image data to visual representation
    npimg = (img.numpy()*255).astype('uint8')
    if len(npimg.shape) == 3 and npimg.shape[0] == 3:
        npimg = np.transpose(npimg, (1, 2, 0))
    else:
        npimg = npimg[0,:,:]
    return npimg

def vis_label(label, mask_colors):
    label = label.cpu()
    # Convert label data to visual representation
    label = np.array(label.numpy())
    if label.shape[-1] == 1:
        label = label[:,:,0]

    # Convert train_ids to colors
    label = mask_colors[label]
    return label

def plot_curves(metrics=None, args=None):
    x = np.arange(args.epochs)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ln1 = ax1.plot(x, metrics['train_loss'], color='tab:red')
    ln2 = ax1.plot(x, metrics['val_loss'], color='tab:red', linestyle='dashed')
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    ln3 = ax2.plot(x, metrics['train_acc'], color='tab:blue')
    ln4 = ax2.plot(x, metrics['val_acc'], color='tab:blue', linestyle='dashed')
    ln5 = ax2.plot(x, metrics['miou'], color='tab:green')
    lns = ln1+ln2+ln3+ln4+ln5
    plt.legend(lns, ['Train loss','Validation loss','Train accuracy','Validation accuracy','mIoU'])
    plt.tight_layout()
    plt.savefig(args.save_path + '/learning_curve.png', bbox_inches='tight')

def test():
    dataset = Cityscapes_Dataset
    M = Metrics()
    iou = Iou(dataset.classLabels, dataset.validClasses, dataset.voidClass)

if __name__ == '__main__':
    test()