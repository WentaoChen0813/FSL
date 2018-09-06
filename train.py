import torchvision
import torch
import torch.utils.data
import read_mydata
import copy


datasets = read_mydata.concat_train_val('/home/wentao/data/mini-Imagenet/mini-imagenet-cache-train.pkl',
                                              '/home/wentao/data/mini-Imagenet/mini-imagenet-cache-val.pkl')
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=4, shuffle=True, num_workers=4)
               for x in ['train', 'val']}

resnet18 = torchvision.models.resnet18()
# resnet18.avgpool = torch.nn.AvgPool2d(2, stride=2, padding=0)
resnet18.fc = torch.nn.Linear(in_features=512, out_features=80)

criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(resnet18.parameters(), lr=0.05)
lr_step = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.1)
device = torch.device('cuda:0')


def train(model, dataloaders, criterion, optim, lr_step, epoch_num, device):
    model = model.to(device)
    model_best = copy.deepcopy(model.state_dict())
    acc_best = 0.0
    epoch_best = 0
    for epoch in range(epoch_num):
        print('-' * 10)
        print('epoch: {}'.format(epoch+1))

        for phase in ['train', 'val']:
            loss_epoch = 0.0
            acc_epoch = 0.0
            num_epoch = 0.0
            if phase == 'train':
                model.train()
                lr_step.step()
            else:
                model.eval()

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                loss = criterion(output, labels)

                _, preds = torch.max(output, 1)
                acc_epoch += torch.sum(preds == labels).item()
                loss_epoch += loss.item() * inputs.size(0)
                num_epoch += inputs.size(0)

                if phase == 'train':
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

            loss_epoch = loss_epoch / num_epoch
            acc_epoch = acc_epoch / num_epoch
            print('{}: loss = {:4f}, acc = {:4f}'.format(phase, loss_epoch, acc_epoch))

            if phase == 'val' and acc_epoch > acc_best:
                acc_best = acc_epoch
                epoch_best = epoch
                model_best = copy.deepcopy(model.state_dict())

    print('best val acc = {:4f} at epoch {:d}'.format(acc_best, epoch_best+1))
    model.load_state_dict(model_best)
    return model


model = train(resnet18, dataloaders, criterion, optim, lr_step, 100, device)
torch.save(model.state_dict(), 'model/resnet18_class80.pkl')
