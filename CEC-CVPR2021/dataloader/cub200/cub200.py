import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.resnet import resnet18


class CUB200(Dataset):

    def __init__(self, root='./', train=True,
                 index_path=None, index=None, base_sess=None):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self._pre_operate(self.root)

        if train:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                #transforms.CenterCrop(224),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomFlipLeftRight(), # additional augment
                #transforms.RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # additional augment
                #transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)], p=0.25),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            # self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def text_read(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self, list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def _pre_operate(self, root):
        image_file = os.path.join(root, 'CUB_200_2011/images.txt')
        split_file = os.path.join(root, 'CUB_200_2011/train_test_split.txt')
        class_file = os.path.join(root, 'CUB_200_2011/image_class_labels.txt')
        id2image = self.list2dict(self.text_read(image_file))
        id2train = self.list2dict(self.text_read(split_file))  # 1: train images; 0: test iamges
        id2class = self.list2dict(self.text_read(class_file))
        train_idx = []
        test_idx = []
        for k in sorted(id2train.keys()):
            if id2train[k] == '1':
                train_idx.append(k)
            else:
                test_idx.append(k)

        self.data = []
        self.targets = []
        self.data2label = {}
        if self.train:
            for k in train_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

        else:
            for k in test_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

    def SelectfromTxt(self, data2label, index_path):
        index = open(index_path).read().splitlines()
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.root, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets



def loadTrained_CEC():
  
  #checkpoint_path = '/net/patriot/scratch/tahmad/FSCIL_datasets/CEC_models/CUB200/session0_max_acc7552_cos.pth'
  checkpoint_path = '/scratch/tahmad/FSCIL_datasets/CEC_models/CUB200/session0_max_acc7552_cos.pth'
  model = resnet18(num_classes = 200)
  
  
  
  state_dict1 = torch.load(checkpoint_path)
  state_dict = state_dict1['params']
  #print(state_dict)
  
  for k, v in list(state_dict.items()):
      print(k, v.shape)
  for k, v in list(model.state_dict().items()):
      print(k, v.shape)
  
  if checkpoint_path != None:
    print("loading checkpoint")
    assert os.path.isfile(checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    state_dict1 = torch.load(checkpoint_path)
    state_dict_model = state_dict1['params']
    from collections import OrderedDict
    new_state_dict_model = OrderedDict()
    for k, v in state_dict_model.items():
      if 'module.encoder.' == k[:15]: 
        name = k[15:] # remove `module.`
        new_state_dict_model[name] = v
      else:
        new_state_dict_model[k] = v
    model.load_state_dict(new_state_dict_model, strict=False)
  
  return model    
    


if __name__ == '__main__':
    """
    txt_path = "../../data/index_list/cub200/session_1.txt"
    # class_index = open(txt_path).read().splitlines()
    base_class = 100
    class_index = np.arange(base_class)
    #dataroot = '~/dataloader/data'
    #dataroot = '/scratch/tahmad/FSCIL_datasets/CUB_200'
    dataroot = '/net/patriot/scratch/tahmad/FSCIL_datasets/CUB_200'
    
    batch_size_base = 400
    trainset = CUB200(root=dataroot, train=False,  index=class_index,
                      base_sess=True)
    cls = np.unique(trainset.targets)
    print(trainset.__len__())
    
    
    trainset = CUB200(root=dataroot, train=True,  index=class_index,
                      base_sess=True)
    cls = np.unique(trainset.targets)
    print(trainset.__len__())
    
    """
    
    txt_path = "../../data/index_list/cub200/session_11.txt"
    dataroot = '/scratch/tahmad/FSCIL_datasets/CUB_200'
    #dataroot = '/net/patriot/scratch/tahmad/FSCIL_datasets/CUB_200'
    
    base_class = 110
    class_index = np.arange(base_class)
    
    trainset = CUB200(root=dataroot, train=False, index=class_index)
    cls = np.unique(trainset.targets)
    print(trainset.__len__())
    
    trainset = CUB200(root=dataroot, train=True, index_path=txt_path)
    cls = np.unique(trainset.targets)
    print(trainset.__len__())
    
    
    
    #model = loadTrained_CEC()
    #print(model)
    
    
    #trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
    #                                          pin_memory=True)

    # txt_path = "../../data/index_list/cifar100/session_2.txt"
    # # class_index = open(txt_path).read().splitlines()
    # class_index = np.arange(base_class)
    # trainset = CIFAR100(root=dataroot, train=True, download=True, transform=None, index=class_index,
    #                     base_sess=True)
    # cls = np.unique(trainset.targets)
    # trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
    #                                           pin_memory=True)
