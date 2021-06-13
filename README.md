# Semantic Segmentation

Implemented the Unet model to do semantic segmentation

## dataset: Cityscapes

### Download packages

Download packaes listed in the offical website and unzip them to directory ```data```. 

        ├── data
        │   ├── gtCoarse
        │   │   ├── train
        │   │   ├── train_extra
        │   │   └── val
        │   ├── gtFine
        │   │   ├── test
        │   │   ├── train
        │   │   └── val
        │   └── leftImg8bit
        │       ├── test
        │       ├── train
        │       ├── train_extra
        │       └── val

## Model utilization

### Model loading 

Type the following command to continue trainning

        python main.py --weights /save/checkpoint.pth.tar

### Predicting

Type the following command to predict on the test set (it will automatically fetch the best model weights)

        python main.py --predict


### Result

Since the network is relatively large, I have tried to simplify it a little bit so that we can train it faster. See the model details in ```model.py``` file. And we can find the results in ```log_epoch.csv```. Here's the best resuls.

        epoch, train loss, val loss, train acc, val acc, miou
        1, 0.68217, 0.72684, 1.23130, 1.23180, 0.22013

## Reference 

Thanks to github repository https://github.com/hoya012/semantic-segmentation-tutorial-pytorch, I used its interface and training tricks
Thanks to github repository https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet, I have studied its network architecture