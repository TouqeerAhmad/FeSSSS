import argparse
import os
import numpy as np
import h5py
import pathlib
import multiprocessing as mp
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

from cub200 import CUB200 

n_cpu = int(os.cpu_count()*0.8)
batch_size = 256
#N_feature = 2048

def deep_get(dict_obj, key):
    d = dict_obj
    for k in key.split(":s:"):
        if type(d) == torch.nn.modules.container.Sequential or type(d) == torchvision.models.resnet.Bottleneck:
            d = d.__dict__['_modules']
        d = d[k]
    return d

class Model_Operations():
    def __init__(self, model, layer_names):
        self.model = model
        self.outputs = []
        self.layer_names = layer_names
        for layer_name in layer_names:
            deep_get(self.model.__dict__['_modules'], layer_name).register_forward_hook(self.feature_hook)

    def feature_hook(self, module, input, output):
        self.outputs.append(output)

    def __call__(self, x):
        self.outputs = []
        _ = self.model(x)
        return dict(zip(self.layer_names,self.outputs))


def extract_features(N, N_feature, val_loader, modelObj):
    print('N: ', N)
    
    feature_val = np.empty([N, N_feature+1])
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader, 0):
            print(n)
            images = images.cuda()
            target = target.cuda()
            layer_outputs = modelObj(images)
            for layer in layer_outputs:
                print(layer)
                print(layer_outputs[layer].shape)
                
                feature_val[n:(n+images.size(0)),0] = target.cpu().data.numpy()
                feature_val[n:(n+images.size(0)),1:] = layer_outputs[layer][:,:,0,0].cpu().data.numpy()
                n = n+images.size(0)
                print(target.cpu().data.numpy())
    
    return feature_val


def getTrainSet(session_ID):
    
    session_file_names = ['session_1.txt', 'session_2.txt', 'session_3.txt', 'session_4.txt', 'session_5.txt', 'session_6.txt', 'session_7.txt',
                            'session_8.txt', 'session_9.txt', 'session_10.txt', 'session_11.txt']
    txt_path = "../../data/index_list/cub200/" + session_file_names[session_ID-1]
    
    # class_index = open(txt_path).read().splitlines()
    num_classes_per_session = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    base_class = num_classes_per_session[session_ID-1]
    class_index = np.arange(base_class)
    #dataroot = '~/dataloader/data'
    #dataroot = '/net/patriot/scratch/tahmad/FSCIL_datasets/CUB_200'
    dataroot = '/scratch/tahmad/FSCIL_datasets/CUB_200'
    batch_size_base = 400
    if (session_ID-1) == 0:
        trainset = CUB200(root=dataroot, train=True,  index=class_index, base_sess=True) # for base session
    else:
        trainset = CUB200(root=dataroot, train=True, index_path=txt_path)
    cls = np.unique(trainset.targets)
    print(trainset.__len__())
    
    return trainset


def getValSet(session_ID):
    
    session_file_names = ['session_1.txt', 'session_2.txt', 'session_3.txt', 'session_4.txt', 'session_5.txt', 'session_6.txt', 'session_7.txt',
                            'session_8.txt', 'session_9.txt', 'session_10.txt', 'session_11.txt']
    txt_path = "../../data/index_list/cub200/" + session_file_names[session_ID-1]
    
    # class_index = open(txt_path).read().splitlines()
    num_classes_per_session = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    base_class = num_classes_per_session[session_ID-1]
    class_index = np.arange(base_class)
    #dataroot = '~/dataloader/data'
    #dataroot = '/net/patriot/scratch/tahmad/FSCIL_datasets/CUB_200'
    dataroot = '/scratch/tahmad/FSCIL_datasets/CUB_200'
    batch_size_base = 400
    if (session_ID-1) == 0:
        trainset = CUB200(root=dataroot, train=False,  index=class_index, base_sess=True) # for base session
    else:
        trainset = CUB200(root=dataroot, train=False,  index=class_index, base_sess=False)
    cls = np.unique(trainset.targets)
    print(trainset.__len__())
    
    return trainset


def load_CEC_stat_dict(checkpoint_path):
    
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
        
    return new_state_dict_model

def main(args):
    # cudnn.benchmark = True
    # Data loading code
    pytorch_models = sorted(name for name in models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(models.__dict__[name]))
    if args.arch in pytorch_models:
        model = models.__dict__[args.arch](pretrained=True)
        #model = models.__dict__[args.arch](pretrained=False)

    if args.weights is not None:
        if args.special_net == 'cec':
            state_dict = load_CEC_stat_dict(args.weights)
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"\n\n\nmsg {msg}")
        else:    
            #state_dict = torch.load(args.weights, map_location="cpu")['state_dict']
            state_dict = torch.load(args.weights, map_location="cpu")
            if 'state_dict' in state_dict:
                print('In the If condition ...')
                state_dict = state_dict['state_dict']
            
            """
            print('\n############################################## ')
            print('Printing state_dict of pretrained model ... ')
            for k, v in list(model.state_dict().items()):
                print(k, v.shape)
            
            print('\n############################################## ')
            print('Printing state_dict of loaded model ... ')
            for k, v in list(state_dict.items()):
                print(k, v.shape)
            """
            
            if args.saved_with_data_parallel:
                #print('\n############################################## ')
                #print('Processing saved with data parallel ...')
                new_state_dict = {}
                for k in state_dict:
                    #print(k)
                    new_state_dict[k.replace('module.', '')] = state_dict[k]
                state_dict = new_state_dict
            
            """    
            print('\n############################################## ')
            print('Printing state_dict after removing module. ')
            for k, v in list(state_dict.items()):
                print(k, v.shape)
            """
    
            if args.ignore_fc:
                del state_dict['fc.weight']
                del state_dict['fc.bias']
            
            if args.for_moco:
                # specific to MoCoV2
                for k in list(state_dict.keys()):
                    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                        print('In the MocoV2 loop ...')
                        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    del state_dict[k]
            
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"\n\n\nmsg {msg}")

        #if len(msg.missing_keys)>0 or len(msg.unexpected_keys)>0:
        #    temp = input("\nPlease confirm to continue or press Ctrl+C to exit\n")

    #print(f"\n\n######### Model Architecture for {args.arch} ##############")
    #print(model)
    #print(f"######### Model Architecture for {args.arch} ##############\n\n")

    model.eval()
    model = model.to('cuda')
    modelObj = Model_Operations(model, args.layer_names)
    
    if args.split == 'train':
        print('Train Split')
        train_dataset = getTrainSet(args.session_ID)
    else:
        print('Val Split')
        train_dataset = getValSet(args.session_ID)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu)
    feature_train = extract_features(train_dataset.__len__(), args.N_feature, train_loader, modelObj)
    np.save(args.feature_file_name, feature_train)
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="This script extracts features from a specific layer for a pytorch model")
    parser.add_argument("--arch",
                        default='resnet18',
                        help="The architecture from which to extract layers. "
                             "Can be a model architecture already available in torchvision or a saved pytorch model.")
    parser.add_argument("--layer_names",
                        nargs="+",
                        help="Layer names to extract",
                        default=["fc"])
    parser.add_argument("--weights", help="network weights",
                        default=None, required=False)
    parser.add_argument("--batch-size", help="Number of samples per forward pass", default=256, type=int)
    parser.add_argument("--saved-with-data-parallel", help="If you saved your model with data parallel set this flag",
                        default=False, action="store_true")
    parser.add_argument("--feature_file_name",
                        default='/net/patriot/scratch/tahmad/FSCIL_datasets/CUB_200_Mocov2_features/features_CUB_train_01.npy',
                        help="Feature file name.")
    parser.add_argument("--split",
                        default='train',
                        help="Name of Train or Val split")
    parser.add_argument("--special_net",
                        default=None,
                        help="Name of special net i.e. cec -- to skip other processing")
    parser.add_argument("--N_feature",
                        default=512,
                        type=int,
                        help="Feature dimension of avgpool")
    parser.add_argument("--ignore-fc", help="""
                                            Ignore FC layer useful only if number of classes in the loaded network 
                                            is different from the standard network architecture and the layer being 
                                            extracted is not the fc layer.
                                            """,
                        default=False, action="store_true")
    
    
    parser.add_argument("--session_ID",
                        default=1,
                        type=int,
                        help="Session for which to save data")
                        
    
    parser.add_argument("--for_moco", help=""" set True if getting moco features""",
                        default=False, action="store_true")
                        
                        
    args = parser.parse_args()
    if args.ignore_fc:
        assert 'fc' not in args.layer_names, \
                    f"OOPS! You have been stopped from doing something you might repent :P"

    main(args)
    
