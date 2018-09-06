import numpy as np
import pickle
import PIL.Image as Image
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt


class MiniImagenet(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, item):
        image = Image.fromarray(self.images[item, :, :, :])
        label = self.labels[item]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def concat_train_val(train_file, val_file):
    train_pkl, val_pkl = [pickle.load(open(file, 'rb')) for file in [train_file, val_file]]
    image_data = np.concatenate((train_pkl['image_data'], val_pkl['image_data']))
    label = np.array([i//600 for i in range(image_data.shape[0])])

    train_ind = [i % 6 != 0 for i in range(image_data.shape[0])]
    val_ind = [i % 6 == 0 for i in range(image_data.shape[0])]

    train_data = image_data[train_ind, :, :, :]
    train_label = label[train_ind]

    val_data = image_data[val_ind, :, :, :]
    val_label = label[val_ind]

    train_trans = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4714, 0.4529, 0.4087], std=[0.2832, 0.2743, 0.2895])]
    )
    val_trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4714, 0.4529, 0.4087], std=[0.2832, 0.2743, 0.2895])]
    )

    train_dataset = MiniImagenet(train_data, train_label, train_trans)
    val_dataset = MiniImagenet(val_data, val_label, val_trans)
    return {'train': train_dataset, 'val': val_dataset}


def imshow(inp, title=None):
    inp = inp.numpy().transpose(1, 2, 0)
    mean = [0.4714, 0.4529, 0.4087]
    std = [0.2832, 0.2743, 0.2895]
    inp = inp * std + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


if __name__ == '__main__':
    datasets = concat_train_val('/home/wentao/data/mini-Imagenet/mini-imagenet-cache-train.pkl',
                                '/home/wentao/data/mini-Imagenet/mini-imagenet-cache-val.pkl')
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=4, shuffle=False, num_workers=4)
                   for x in ['train', 'val']}
    i = 0
    for images, labels in dataloaders['train']:
        if i % 125 == 0:
            images = torchvision.utils.make_grid(images)
            title = [str(x.item()) for x in labels]
            imshow(images, title)
            pass
        i += 1
