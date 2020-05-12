from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from code.utils.RandomErasing import RandomErasing
from code.utils.RandomSampler import RandomSampler
from PIL import Image
import os
import re


class Data():
    def __init__(self):
        train_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.trainset = Cartoon(train_transform, 'train', '\iCartoon')  # path可以设到超参里
        self.testset = Cartoon(test_transform, 'test', '\iCartoon')
        self.queryset = Cartoon(test_transform, 'query', '\iCartoon')

        self.train_loader = dataloader.DataLoader(self.trainset,
                                                  sampler=RandomSampler(self.trainset, batch_id=opt.batchid,
                                                                        batch_image=opt.batchimage),
                                                  batch_size=opt.batchid * opt.batchimage, num_workers=8,
                                                  pin_memory=True)
        self.test_loader = dataloader.DataLoader(self.testset, batch_size=opt.batchtest, num_workers=8, pin_memory=True)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=opt.batchtest, num_workers=8,
                                                  pin_memory=True)

class Cartoon(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):
        self.transform = transform
        if dtype == 'train':
            data_path += '/....'
            fh = open(data_path+'/....txt', 'r')
        elif dtype == 'test':
            data_path += '/....'
            fh = open(data_path + '/....txt', 'r')
        else:
            data_path += '/....'
            fh = open(data_path + '/...txt', 'r')

        imgs = []
        for line in fh:
            words = line.split() #以空格将每行数据分成多列
            imgs.append(data_path + '/' + words[0], words[0].split(_)[3])  # 图片路径
        self.imgs = imgs

    def __getitem__(self, index):
        path,label = self.imgs[index]
        img = Image.open(path).covert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

