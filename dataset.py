import os
import numpy as np
from PIL import Image
import torch
from torchvision.datasets import Cityscapes
import torchvision.transforms.functional as TF

class Cityscapes_Dataset(Cityscapes):

    voidClass = 19

    # convert the id to train id
    id2trainid = np.array([label.train_id for label in Cityscapes.classes if label.train_id >= 0], dtype = 'uint8')
    id2trainid[np.where(id2trainid == 255)] = voidClass # 255 is void

    # convert train id to mask color
    mask_colors = [list(label.color) for label in Cityscapes.classes if label.train_id >= 0 and label.train_id <= 19]
    mask_colors.append([0, 0, 0])
    mask_colors = np.array(mask_colors)

    # convert train id to ids
    trainid2id = np.zeros((256), dtype = 'uint8')
    for label in Cityscapes.classes:
        if label.train_id >= 0 and label.train_id < 255:
            trainid2id[label.train_id] = label.id

    # list of valid class ids
    validClasses = np.unique([label.train_id for label in Cityscapes.classes if label.id >= 0])
    validClasses[np.where(validClasses == 255)] = voidClass
    validClasses = list(validClasses)

    # list of class names
    classLabels = [label.name for label in Cityscapes.classes if not (label.ignore_in_eval or label.id < 0)]
    classLabels.append('void')

    def __init__(self, root='data', split='train', transform=None):
        super(Cityscapes, self).__init__(root)
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, 'gtFine', split)

        self.split = split
        self.images = []
        self.targets = []

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                if split != 'test':
                    target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], 'gtFine_labelIds.png')
                    self.targets.append(os.path.join(target_dir, target_name))

    def __getitem__(self, index):
        filepath = self.images[index]
        image = Image.open(filepath).convert('RGB')

        if self.split != 'test':
            # train, val
            target = Image.open(self.targets[index])
            if self.transform is not None:
                image, target = self.transform(image, mask = target)

            image = TF.to_tensor(image)
            target = self.id2trainid[target] # using trainid
            target = torch.from_numpy(target)
            return image, target, filepath

        elif self.transform is not None:
            # actually transform is always None during test
            image = self.transform(image)

        image = TF.to_tensor(image)
        return image, filepath


def test():
    dataset = Cityscapes_Dataset(root='data', split='train', transform = None)
    L = len(dataset)
    print('Dataset length: ', L, '\tvalid classes: ', len(dataset.validClasses))
    image, target, filepath = dataset.__getitem__(L-1)
    image = TF.to_pil_image(image)
    print('test jpg path:', filepath)
    image.save('dataset_test.jpg')

if __name__ == '__main__':
    test()