import pickle
import PIL.Image as Image
import torchvision
import torch.utils.data


class MiniImagenet(torch.utils.data.Dataset):
    def __init__(self, pkl_dir, transform=None):
        self.pkl = pickle.load(open(pkl_dir, 'rb'))
        self.transform = transform

    def __len__(self):
        return self.pkl('image_data').shape[0]

    def __getitem__(self, idx):
        image = Image.fromarray(self.pkl['image_data'][idx, :, :, :])
        label = idx // 600
        if self.transform is not None:
            image = self.transform(image)
        return {'image': image, 'label': label}


trans = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.4704, 0.4488, 0.4014], std=[0.2843, 0.2753, 0.2903])]
)
train_dataset = MiniImagenet('/home/wentao/data/mini-Imagenet/mini-imagenet-cache-train.pkl', trans)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=32)

trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.4704, 0.4488, 0.4014], std=[0.2843, 0.2753, 0.2903])]
)
val_dataset = MiniImagenet('/home/wentao/data/mini-Imagenet/mini-imagenet-cache-val.pkl', trans)
val_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=32)