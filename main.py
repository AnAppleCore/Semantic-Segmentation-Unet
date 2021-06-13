import os
import time
import argparse
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from model import UNET
from dataset import Cityscapes_Dataset
from utils import get_dataloaders, train, validate, predict, plot_curves

def get_args():
    parser = argparse.ArgumentParser('Hyperparameters setting')
    parser.add_argument('-e', '--epochs', type = int, default = 20)
    parser.add_argument('-b', '--batch_size', type = int, default = 1)
    parser.add_argument('-r', '--learning_rate', type = float, default = 1e-2)
    parser.add_argument('-m', '--momentum', type = float, default = 0.9)
    parser.add_argument('-w', '--weight_decay', type = float, default = 1e-4)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--pin_memory', type = bool, default = True)
    parser.add_argument('--save_path', type = str, default = './save')
    parser.add_argument('--weights', type = str, default = None)
    parser.add_argument('-p', '--predict', action = 'store_true')
    parser.add_argument('--transform', action = 'store_true')
    parser.add_argument('--img_size', nargs='+', type=int, default=[1024, 2048])
    args = parser.parse_args()
    print('Arguments:', args)
    # Create directory to store images results
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path + '/images')
    if not os.path.isdir(args.save_path + '/results_color_val'):
        os.makedirs(args.save_path + '/results_color_val')
        os.makedirs(args.save_path + '/results_color_test')
    # Create results directory
    if not os.path.isdir(args.save_path + '/results_val'):
        os.makedirs(args.save_path + '/results_val')
    if not os.path.isdir(args.save_path + '/results_test'):
        os.makedirs(args.save_path + '/results_test')
    return args

def main():
    args = get_args()

    # data loading
    dataset = Cityscapes_Dataset
    train_loader, val_loader, test_loader = get_dataloaders(args)

    # model initialization
    model = UNET(in_channels=3, out_channels=len(dataset.validClasses))
    model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # parameters and metrics initialization
    best_miou = 0.0
    start_epoch = 0
    metrics = {'train_loss' : [],
               'val_loss' : [],
               'train_acc' : [],
               'val_acc' : [],
               'miou' : []}
    best_path = args.save_path + '/best_model.pth.tar'

    # load checkpoint
    if args.weights is not None:
        print('Load checkpoint from {}.'.format(args.weights))
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_miou = checkpoint['best_miou']
        start_epoch = checkpoint['epoch']+1
        metrics = checkpoint['metrics']

    # push model to GPU
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
    else:
        print('No GPU avaliable.')
        return

    # predict
    if args.predict:
        # load the best model
        print('Load best model from {}.'.format(best_path))
        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        predict(test_loader, model, dataset.mask_colors, mode='test', args=args)
        return

    # generating log
    with open(args.save_path + '/log_epoch.csv', 'a') as epoch_log:
        epoch_log.write('epoch, train_loss, val loss, tran_acc, val acc, miou\n')

    start = time.time()
    for epoch in range(start_epoch, args.epochs):

        # train
        print('------Training------')
        train_loss, train_acc = train(train_loader, model, loss_func, optimizer, scheduler, epoch, dataset.voidClass, args)

        # update metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        print('Train epoch {} complete! train loss: {:.4f}, acc: {:.4f}'.format(epoch,train_loss,train_acc))

        # validate
        print('------Validating------')
        val_loss, val_acc, miou = validate(val_loader, model, loss_func, epoch, dataset.classLabels, dataset.validClasses, dataset.voidClass, dataset.mask_colors, args)

        # update metrics
        metrics['val_acc'].append(val_acc)
        metrics['val_loss'].append(val_loss)
        metrics['miou'].append(miou)
        print('Val epoch {} complete! val loss: {:.4f}, acc: {:.4f}, miou {:.4f}'.format(epoch,val_loss,val_acc, miou))

        # write log
        with open(args.save_path + '/log_epoch.csv', 'a') as epoch_log:
            epoch_log.write('{}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n'.format( epoch, train_loss, val_loss, train_acc, val_acc, miou))

        # save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_miou': best_miou,
            'metrics': metrics,
        }, args.save_path + '/checkpoint.pth.tar')

        # save best model
        if miou > best_miou:
            print('mIoU improved from {:.4f} to {:.4f}.'.format(best_miou, miou))
            best_miou = best_miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, best_path)
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    plot_curves(metrics, args)

    # output results
    checkpoint = torch.load(best_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print('Loaded best model weights (epoch {}) from {}'.format(checkpoint['epoch'], best_path))
    predict(test_loader, model, dataset.mask_colors, mode='test', args=args)
    predict(val_loader, model, dataset.mask_colors, mode='val', args=args)


if __name__ == '__main__':
    main()
    # args = get_args()