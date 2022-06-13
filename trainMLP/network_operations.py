import os
import torch
import torch.nn as nn
import numpy as np
from vast import losses
import torch.utils.data as data_util
from torch.utils.tensorboard import SummaryWriter
from vast.tools import logger as vastlogger
torch.manual_seed(0)

logger = vastlogger.get_logger()

class MLP(nn.Module):
    def __init__(self, input_feature_size=2048, num_classes=50):
        super(MLP, self).__init__()
        
        #self.fc2 = nn.Linear(in_features=input_feature_size, out_features=num_classes, bias=True)
        
        self.fc1 = nn.Linear(in_features=input_feature_size, out_features=input_feature_size//2, bias=True)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=num_classes, bias=True)
        
        #self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=input_feature_size//4, bias=True)
        #self.fc3 = nn.Linear(in_features=self.fc2.out_features, out_features=num_classes, bias=True)
        
        #self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=input_feature_size//4, bias=True)
        #self.fc3 = nn.Linear(in_features=self.fc2.out_features, out_features=input_feature_size//8, bias=True)
        #self.fc4 = nn.Linear(in_features=self.fc3.out_features, out_features=num_classes, bias=True)

    def forward(self, x):
        #x = self.fc2(x)
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        #x = self.fc3(x)
        #x = self.fc4(x)
        
        return x

class network():
    def __init__(self, num_classes, input_feature_size, output_dir = None):
        self.net = MLP(num_classes=num_classes, input_feature_size=input_feature_size)
        self.net = self.net.cuda()
        self.cls_names = []
        self.input_feature_size = input_feature_size
        self.output_dir = output_dir
        self.bestValAcc = 0.0
        self.bestValAccEpoch = -1

    def prep_training_data(self, classes_in_consideration, training_tensor_x, training_tensor_label):
        
        training_tensor_x = training_tensor_x.type(torch.FloatTensor).cuda()
        training_tensor_label = np.array(training_tensor_label)
        
        training_tensor_y=torch.zeros(training_tensor_label.shape[0]).type(torch.LongTensor).cuda()
        sample_weights = torch.ones(training_tensor_label.shape[0]).cuda()
        sample_weights = sample_weights*100.
        
        for cls_no,cls in enumerate(classes_in_consideration):
            training_tensor_y[training_tensor_label==cls_no]=cls_no
            freq = sample_weights[training_tensor_label==cls_no].shape[0]
            print(cls_no, cls, freq)
            sample_weights[training_tensor_label==cls_no] /= freq
            
        logger.debug(f"Training dataset size {list(training_tensor_x.shape)} "
                     f"labels size {list(training_tensor_y.shape)} "
                     f"sample weights size {list(sample_weights.shape)}")
        self.dataset = data_util.TensorDataset(training_tensor_x, training_tensor_y, sample_weights)
        self.cls_names = classes_in_consideration

    def training(self, classes_in_consideration, training_tensor_x, training_tensor_label, epochs=150, lr=0.01, batch_size=256):
        print(self.net)
        weights = self.net.state_dict()
        #print(weights)
        self.prep_training_data(classes_in_consideration, training_tensor_x, training_tensor_label)
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        loader = data_util.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        no_of_print_statements = min(10,epochs)
        printing_interval = epochs//no_of_print_statements
        summary_writer = None
        if self.output_dir is not None:
            summary_writer = SummaryWriter(f"{self.output_dir}/MLP_training_logs")
        for epoch in range(epochs):
            loss_history=[]
            train_accuracy = torch.zeros(2, dtype=int)
            for x, y, s in loader:
                optimizer.zero_grad()
                output = self.net(x)
                loss = loss_fn(output, y)
                #print(loss)
                tempvalues, tempindices = torch.max(output, 1)
                
                train_accuracy += losses.accuracy(output, y)
                loss_history.extend(loss.tolist())
                loss *= s
                loss.mean().backward()
                optimizer.step()
                
            to_print=f"Epoch {epoch:03d}/{epochs:03d} \t"\
                     f"train-loss: {np.mean(loss_history):1.5f}  \t"\
                     f"accuracy: {float(train_accuracy[0]) / float(train_accuracy[1]):9.5f}"
            if summary_writer is not None:
                summary_writer.add_scalar(f"{len(self.cls_names)}/loss",
                                          np.mean(loss_history), epoch)
                summary_writer.add_scalar(f"{len(self.cls_names)}/accuracy",
                                          float(train_accuracy[0])/float(train_accuracy[1]), epoch)
            if epoch%printing_interval==0:
                logger.info(to_print)
            else:
                logger.debug(to_print)
                
                
    def training_and_eval(self, classes_in_consideration, training_tensor_x, training_tensor_label, val_tensor_x, val_tensor_label, epochs=150, lr=0.01, batch_size=256, modelOutputDirName=None):
        print(self.net)
        weights = self.net.state_dict()
        #print(weights)
        self.prep_training_data(classes_in_consideration, training_tensor_x, training_tensor_label)
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        loader = data_util.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        no_of_print_statements = min(10,epochs)
        printing_interval = epochs//no_of_print_statements
        summary_writer = None
        if self.output_dir is not None:
            summary_writer = SummaryWriter(f"{self.output_dir}/MLP_training_logs")
        for epoch in range(epochs):
            loss_history=[]
            train_accuracy = torch.zeros(2, dtype=int)
            for x, y, s in loader:
                optimizer.zero_grad()
                output = self.net(x)
                loss = loss_fn(output, y)
                #print(loss)
                tempvalues, tempindices = torch.max(output, 1)
                
                train_accuracy += losses.accuracy(output, y)
                loss_history.extend(loss.tolist())
                loss *= s
                loss.mean().backward()
                optimizer.step()
                
            to_print=f"Epoch {epoch:03d}/{epochs:03d} \t"\
                     f"train-loss: {np.mean(loss_history):1.5f}  \t"\
                     f"accuracy: {float(train_accuracy[0]) / float(train_accuracy[1]):9.5f}"
            if summary_writer is not None:
                summary_writer.add_scalar(f"{len(self.cls_names)}/loss",
                                          np.mean(loss_history), epoch)
                summary_writer.add_scalar(f"{len(self.cls_names)}/accuracy",
                                          float(train_accuracy[0])/float(train_accuracy[1]), epoch)
            if epoch%printing_interval==0:
                logger.info(to_print)
            else:
                logger.debug(to_print)
          
            result = self.inference(val_tensor_x)
            acc = self.computeValAccuray(epoch, result, val_tensor_label)
            if acc > self.bestValAcc:
                self.save_model_v2(epoch, acc, modelOutputDirName)
                self.bestValAcc = acc
                self.bestValAccEpoch = epoch
                
        print('The best model saved was with Val Accuracy: ', self.bestValAcc)
        print('At Epoch #: ', self.bestValAccEpoch)
          
    
    def computeValAccuray(self, epoch, result, val_tensor_label):
      
      values, indices = torch.max(result, 1)
      predictions = indices.cpu().numpy()
      sm_scores = values.cpu().numpy()
      labels_val = val_tensor_label.cpu().numpy()
      
      correctCount = 0
      
      for k in range(len(predictions)):
        if predictions[k] == labels_val[k]:
          correctCount += 1
      
      acc = 100.0 * correctCount / len(predictions)
      print('############################################### Epoch # : ', epoch)
      print('# of samples: ', len(predictions), '# predicted correctly: ', correctCount, '% predicted correctly: ', acc)
      
      return acc
        

    def inference(self, validation_data):
        results = {}
        with torch.no_grad():
            logits = self.net(validation_data.type(torch.FloatTensor).cuda()).cpu()
            results = torch.nn.functional.softmax(logits, dim=1)
        logger.info(f"Inference done on {len(results)} classes")
        return results
        
    
    def save_model(self, epoch, modelOutputDirName):
        state = { 'epoch': epoch, 'state_dict': self.net.state_dict()}
        torch.save(state, f'{modelOutputDirName}/mlp_epoch_{epoch}.pth.tar')
    
    def save_model_v2(self, epoch, acc, modelOutputDirName):
        state = { 'epoch': epoch, 'state_dict': self.net.state_dict()}
        if epoch < 10:
          epoch1 = '000' + str(epoch)
        elif epoch >= 10 and epoch < 100:
          epoch1 = '00' + str(epoch)
        elif epoch >= 100 and epoch < 1000:
          epoch1 = '0' + str(epoch)
        else:
          epoch1 = str(epoch)
        torch.save(state, f'{modelOutputDirName}/mlp_epoch_{epoch1}_val_acc_{acc}.pth.tar')
        
    def load_model(self, checkpoint_path):
        if checkpoint_path != None:
            print("loading checkpoint")
            assert os.path.isfile(checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            if 'epoch' in checkpoint.keys():
              state_dict_model = checkpoint['state_dict']
            else:
              state_dict_model = checkpoint
            from collections import OrderedDict
            new_state_dict_model = OrderedDict()
            for k, v in state_dict_model.items():
                if 'module.' == k[:7]: 
                    name = k[7:] # remove `module.`
                    new_state_dict_model[name] = v
                else:
                    new_state_dict_model[k] = v
            self.net.load_state_dict(new_state_dict_model)
            #self.net.cuda()
            
    def modify_net(self, new_num_classes):
        torch.manual_seed(0)
        new_net = MLP(num_classes=new_num_classes,
                      input_feature_size=self.input_feature_size).cuda()
        weights = self.net.state_dict()
        to_add = torch.rand(new_net.fc2.out_features - self.net.fc2.out_features, weights['fc2.weight'].shape[1]).cuda()
        weights['fc2.weight'] = torch.cat((weights['fc2.weight'], to_add))
        to_add = torch.rand(new_net.fc2.out_features - self.net.fc2.out_features).cuda()
        weights['fc2.bias'] = torch.cat((weights['fc2.bias'], to_add))
        new_net.load_state_dict(weights)
        self.net = new_net
        
        
    def print_model(self):
        print(self.net)
        #weights = self.net.state_dict()
        #print(weights)
