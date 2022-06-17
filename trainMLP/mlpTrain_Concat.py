import argparse
import os
import sys
import vast
import torch
from vast.tools import logger as vastlogger
import network_operations as network_operations 
import numpy as np
import time

np.random.seed(0)
torch.manual_seed(0)

################################################################################################
batch_size = 256


## Things to change 

#dataSetInUse = 'CUB' #'cifar100' # #'miniimagenet' #
#NUMBER_OF_FEATURE_FILES = 60
#gnerateData = False
#useVectorVariance = False 

#featureBeingUsed = 'concat' #'selfsupervised_only' # #'supervised_only' # #'supervised_only' #  # # # # #  # #

# cifar100
#dirSelfSupervisedFeatures = 'cifar100_deepclusterv2_ResNet50_features_imagenet/'
#dirSupervisedFeatures = 'cifar100_CEC_ResNet20_features_batch0/'


# miniimagenet
#dirSelfSupervisedFeatures = 'miniimagenet_deepclusterv2_ResNet50_features_imagenet/'
#dirSelfSupervisedFeatures = 'miniimagenet_deepclusterv2_ResNet50_features_imagenet_smaller/'
#dirSupervisedFeatures = 'miniimagenet_CEC_ResNet18_features_batch0/'

# CUB
#dirSelfSupervisedFeatures = 'CUB_200_deepclusterv2_ResNet50_features_imagenet/'
#dirSelfSupervisedFeatures = 'CUB_200_Mocov2_ResNet50_features_imagenet_800ep/'
#dirSelfSupervisedFeatures = 'CUB_200_Mocov2_ResNet50_features_openimages/'
#dirSelfSupervisedFeatures = 'CUB_200_swav_ResNet50_features_imagenet/'
#dirSelfSupervisedFeatures = 'CUB_200_selav2_ResNet50_features_imagenet/'

#dirSupervisedFeatures = 'CUB_200_CEC_ResNet18_features_batch0/'
#dirSupervisedFeatures = 'CUB_200_supervised_ResNet18_features_batch0/'
################################################################################################



def getData(dirName, numFiles, dataSetInUse, root_path):
  print('\n ################################ Loading Feature Files ################################ ')
  trainFile1 = root_path + dirName + '/features_' + dataSetInUse + '_train_01.npy'
  trainFile2 = root_path + dirName + '/features_' + dataSetInUse + '_train_02.npy'
  
  trainFeatureWithLabels1 = np.load(trainFile1)
  trainFeatureWithLabels2 = np.load(trainFile2)
  featuresTemp = np.vstack((trainFeatureWithLabels1,trainFeatureWithLabels2))
  
  for k in range(3, numFiles+1):
    if k < 10:
      trainFile = root_path + dirName + '/features_' + dataSetInUse + '_train_0' + str(k) + '.npy'
    else:
      trainFile = root_path + dirName + '/features_' + dataSetInUse + '_train_' + str(k) + '.npy'
      
    trainFeatureWithLabels = np.load(trainFile)
    featuresTemp = np.vstack((featuresTemp,trainFeatureWithLabels))
  
  valFile1 = root_path + dirName + '/features_' + dataSetInUse + '_val_central.npy'
  valFeatureWithLabels1 = np.load(valFile1)
  
  print('Shape of Train Features ....')
  print(featuresTemp.shape)
  
  print('Shape of Val Features ....')
  print(valFeatureWithLabels1.shape)
  
  return featuresTemp, valFeatureWithLabels1
  
  
def normalize_and_concat_features(features1, features2, featureBeingUsed):
  
  sz1 = features1.shape
  sz2 = features2.shape
  
  features1_norm = np.linalg.norm(features1, axis=1)
  features2_norm = np.linalg.norm(features2, axis=1)
  
  features1_norm_mat = np.repeat(features1_norm, sz1[1], axis=0)
  features2_norm_mat = np.repeat(features2_norm, sz2[1], axis=0)
  
  features1_norm_mat_reshaped = np.reshape(features1_norm_mat, sz1)
  features2_norm_mat_reshaped = np.reshape(features2_norm_mat, sz2)
  
  features1 /= features1_norm_mat_reshaped
  features2 /= features2_norm_mat_reshaped
  
  if featureBeingUsed == 'concat':
    features = np.hstack((features1,features2)) # concatenated features
  elif featureBeingUsed == 'selfsupervised_only':
    features = features1 # self-supervised only
  else:
    features = features2 # supervised only
  
  return features
  

def run_base_session(dataSetInUse, NUMBER_OF_FEATURE_FILES, epochs, dirSelfSupervisedFeatures, dirSupervisedFeatures, featureBeingUsed, root_path):
  print('###############################################################################################')
  print('###############################################################################################')
  print('###############################################################################################')
  print('############################### session_01 #################################')
  print('###############################################################################################')
  print('###############################################################################################')
  print('###############################################################################################')
    
  
  start = time.time()
  
  dirName = dirSelfSupervisedFeatures + 'session_01'
  featuresTemp1, valFeatureWithLabels1 = getData(dirName, NUMBER_OF_FEATURE_FILES, dataSetInUse, root_path)
  
  dirName = dirSupervisedFeatures + 'session_01'
  featuresTemp2, valFeatureWithLabels2 = getData(dirName, NUMBER_OF_FEATURE_FILES, dataSetInUse, root_path)
  
  features1 = featuresTemp1[:,1:]
  features2 = featuresTemp2[:,1:]
  
  features = normalize_and_concat_features(features1, features2, featureBeingUsed)
  labels = featuresTemp1[:,0]
  
  print('Shape of train features concatenated ...')
  print(features.shape)
  
  
  features_val1 = valFeatureWithLabels1[:,1:]
  features_val2 = valFeatureWithLabels2[:,1:]
  
  
  features_val = normalize_and_concat_features(features_val1, features_val2, featureBeingUsed)
  labels_val = valFeatureWithLabels1[:,0]
  
  print('Shape of val features concatenated ...')
  print(features_val.shape)
  
  if dataSetInUse == 'CUB':  
    num_classes = 100
  else:
    num_classes = 60
  
  mainDirName = '/home/tahmad/work/FeSSSS/trainMLP/' + dataSetInUse + '/'
  os.mkdir(mainDirName)
  logDirName = mainDirName  + dataSetInUse + '_tempDir1' 
  modelOutputDirName = mainDirName + dataSetInUse +'_trained_models_session_01'
  os.mkdir(modelOutputDirName)
  
  logger = vastlogger.setup_logger(level=0, output=logDirName)
  net_obj = network_operations.network(num_classes=num_classes, input_feature_size=features.shape[1], output_dir = logDirName)
  
  #epochs = 500 #1000 #400 #1000
  lr = 0.1
  
  logger.info(f"Running MLP training with LR {lr} for {epochs} epochs")
  classes_in_consideration = torch.from_numpy(np.arange(num_classes))
  
  features = torch.from_numpy(features)
  labels = torch.from_numpy(labels)
  
  features_val = torch.from_numpy(features_val)
  labels_val = torch.from_numpy(labels_val)
  
  net_obj.training_and_eval(classes_in_consideration, features, labels, features_val, labels_val, epochs, lr, batch_size, modelOutputDirName)
  
  end = time.time()
  print('Time took to train: ', end - start)
  
  # save the network 
  net_obj.save_model(epochs, modelOutputDirName)
  
  result = net_obj.inference(features_val)
  top1_accuracy = printValAccuracy(result, labels_val)
  
  return top1_accuracy
  
  

def getPrototypesForSession(sessionName, dataSetInUse, NUMBER_OF_FEATURE_FILES, useVectorVariance, dirSelfSupervisedFeatures, dirSupervisedFeatures, featureBeingUsed, root_path):
  
  # reading self-supervised features
  dirName = dirSelfSupervisedFeatures + sessionName
  featuresTemp1, valFeatureWithLabels1 = getData(dirName, NUMBER_OF_FEATURE_FILES, dataSetInUse, root_path)
  
  # reading supervised features
  dirName = dirSupervisedFeatures + sessionName
  featuresTemp2, valFeatureWithLabels2 = getData(dirName, NUMBER_OF_FEATURE_FILES, dataSetInUse, root_path)
  
  # concatentaed train features
  features1 = featuresTemp1[:,1:]
  features2 = featuresTemp2[:,1:]
  features = normalize_and_concat_features(features1, features2, featureBeingUsed)
  labels = featuresTemp1[:,0]
  
  if dataSetInUse == 'CUB':
    if sessionName == 'session_01':
      num_classes = 100
    else:
      num_classes = 10
  else:
    if sessionName == 'session_01':
      num_classes = 60
    else:
      num_classes = 5
  
  # compute prototypes
  numInstancesPerClass = int(labels.shape[0]/num_classes)
  prototypes = np.zeros(shape=(num_classes, features.shape[1]), dtype=np.float32)
  exmplesPerClass = np.zeros(num_classes)
  prototypeLabels = np.zeros(num_classes)
  examplesInClass = np.zeros(shape=(num_classes, numInstancesPerClass, features.shape[1]))
  smallestClassIndex = np.min(labels)
  classVariances = np.zeros(shape=(num_classes, features.shape[1]))
  
  for k in range(features.shape[0]):
    rowIndex = int(labels[k] - smallestClassIndex)
    prototypes[rowIndex,:] += features[k,:]
    examplesInClass[rowIndex,int(exmplesPerClass[rowIndex]),:] = features[k,:]
    exmplesPerClass[rowIndex] += 1
    
  print('exmplesPerClass: ', exmplesPerClass)
  if useVectorVariance:
    for k in range(num_classes):
      classVariances[k,:] = np.var(examplesInClass[k,:,:], axis=0)
  else:
    for k in range(num_classes):
      classVariances[k,:] = np.var(examplesInClass[k,:,:]) * np.ones(features.shape[1])
  
  for k in range(num_classes):
    prototypes[k] /= exmplesPerClass[k]
    prototypeLabels[k] = int(k + smallestClassIndex)
  
  return prototypes, prototypeLabels, classVariances, numInstancesPerClass
  


def generateMutivariateData(prototypes, prototypeLabels, classVariances, numInstancesPerClass):
  numClasses = prototypes.shape[0]
  numFeatures = prototypes.shape[1]
  generatedData = np.zeros(shape=(numInstancesPerClass*numClasses,numFeatures))
  generatedLabels = np.zeros(numInstancesPerClass*numClasses)
  
  
  for k in range(numClasses):
    x = np.random.multivariate_normal(prototypes[k,:],  np.eye(numFeatures)*classVariances[k,:].T, numInstancesPerClass)
    generatedLabels[k*numInstancesPerClass:(k+1)*numInstancesPerClass] = prototypeLabels[k]*np.ones(numInstancesPerClass)
    generatedData[k*numInstancesPerClass:(k+1)*numInstancesPerClass,:] = x
  
  return generatedData, generatedLabels


def main(args):
  top1_accuracy = run_base_session(args.dataSetInUse, args.NUMBER_OF_FEATURE_FILES, args.base_epochs, args.dirSelfSupervisedFeatures, args.dirSupervisedFeatures, args.featureBeingUsed, args.root_path)
  #top1_accuracy = 0.0
  
  if args.dataSetInUse == 'CUB':
    sessionNames = ['session_01', 'session_02', 'session_03', 'session_04', 'session_05', 'session_06', 'session_07', 'session_08', 'session_09', 'session_10', 'session_11']
    num_classesAll = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    numOfSessionInThisDataSet = 11
    numIncrementalClasses = 10
  else:
    sessionNames = ['session_01', 'session_02', 'session_03', 'session_04', 'session_05', 'session_06', 'session_07', 'session_08', 'session_09']
    num_classesAll = [60, 65, 70, 75, 80, 85, 90, 95, 100]
    numOfSessionInThisDataSet = 9
    numIncrementalClasses = 5
    
      
  
  Top1_Performance = []
  Top1_Performance.append(top1_accuracy)
  
  for sessIndex in range(1,numOfSessionInThisDataSet):
    
    print('###############################################################################################')
    print('###############################################################################################')
    print('###############################################################################################')
    print('###############################' + sessionNames[sessIndex] + '#################################')
    print('###############################################################################################')
    print('###############################################################################################')
    print('###############################################################################################')
    
    
    sessionName = sessionNames[sessIndex-1]
    if sessIndex == 1:
      prototypes, prototypeLabels, classVariances, numInstancesPerClass = getPrototypesForSession(sessionName, args.dataSetInUse, args.NUMBER_OF_FEATURE_FILES, args.useVectorVariance, args.dirSelfSupervisedFeatures, args.dirSupervisedFeatures, args.featureBeingUsed, args.root_path)
      #np.save('./prototypes_base_session/prototypes.npy', prototypes)
      #np.save('./prototypes_base_session/prototypeLabels.npy', prototypeLabels)
      #np.save('./prototypes_base_session/classVariances.npy', classVariances)
      #np.save('./prototypes_base_session/numInstancesPerClass.npy', numInstancesPerClass)
    else:
      prototypes_curr, prototypeLabels_curr, classVariances_curr, numInstancesPerClass = getPrototypesForSession(sessionName, args.dataSetInUse, args.NUMBER_OF_FEATURE_FILES, args.useVectorVariance, args.dirSelfSupervisedFeatures, args.dirSupervisedFeatures, args.featureBeingUsed, args.root_path)
      prototypes = np.vstack((prototypes,prototypes_curr))
      prototypeLabels = np.hstack((prototypeLabels, prototypeLabels_curr))
      classVariances = np.vstack((classVariances, classVariances_curr))
    
    #exit()
    
    N_feature = prototypes.shape[1]
    num_classes = num_classesAll[sessIndex-1]
    
    # initialize MLP with session_01 trained model
    mainDirName = '/home/tahmad/work/FeSSSS/trainMLP/' + args.dataSetInUse + '/'
    output_dir = mainDirName + args.dataSetInUse + '_tempDir'  + str(sessIndex+1)
    logger = vastlogger.setup_logger(level=0, output=output_dir)
    net_obj = network_operations.network(num_classes=num_classes, input_feature_size=N_feature, output_dir = output_dir)
    if sessIndex == 1:
      checkpoint_path = mainDirName + args.dataSetInUse + '_trained_models_' + sessionName +'/mlp_epoch_' + str(args.base_epochs) + '.pth.tar'
    else:
      checkpoint_path = mainDirName + args.dataSetInUse + '_trained_models_' + sessionName +'/mlp_epoch_500.pth.tar'
  
  
    net_obj.load_model(checkpoint_path)
    print('Trained Model ...')
    print(net_obj.print_model())
  
    ##
    # extend the MLP by adding 10 new nodes
    num_classes_new = num_classesAll[sessIndex]
    net_obj.modify_net(num_classes_new)
    print('New Modified Model ...')
    print(net_obj.print_model())
    
    # getting self-supervsied and supervised features
    dirName = args.dirSelfSupervisedFeatures + sessionNames[sessIndex]
    featuresTemp1, valFeatureWithLabels1 = getData(dirName, args.NUMBER_OF_FEATURE_FILES, args.dataSetInUse, args.root_path)
    dirName = args.dirSupervisedFeatures  + sessionNames[sessIndex]
    featuresTemp2, valFeatureWithLabels2 = getData(dirName, args.NUMBER_OF_FEATURE_FILES, args.dataSetInUse, args.root_path)
    
  
    # train features -- session_02
    features1 = featuresTemp1[:,1:]
    features2 = featuresTemp2[:,1:]
    features = normalize_and_concat_features(features1, features2, args.featureBeingUsed)
    labels = featuresTemp1[:,0]
    numInstancesPerClass = int(labels.shape[0] / numIncrementalClasses)
    
    if (args.gnerateData):
      print('Add Generated Data ...')
      generatedData, generatedLabels = generateMutivariateData(prototypes, prototypeLabels, classVariances, numInstancesPerClass)
      features = np.vstack((features,generatedData))
      labels = np.hstack((labels,generatedLabels))
    else:
      # adding prototypes
      print('Add Centroids ...')
      features = np.vstack((features,prototypes))
      labels = np.hstack((labels,prototypeLabels))
  
    #exit()    
      
    print('Shape of train features concatenated ...')
    print(features.shape)
    features = torch.from_numpy(features)
    labels = torch.from_numpy(labels)
    
    # val features -- session_02
    features_val1 = valFeatureWithLabels1[:,1:]
    features_val2 = valFeatureWithLabels2[:,1:]
    features_val = normalize_and_concat_features(features_val1, features_val2, args.featureBeingUsed)
    labels_val = valFeatureWithLabels1[:,0]
    print('Shape of val features concatenated ...')
    print(features_val.shape)
    features_val = torch.from_numpy(features_val)
    labels_val = torch.from_numpy(labels_val)
    
    # inference for session_02 data before training
    print('inference for data before training')
    result = net_obj.inference(features_val)
    top1_accuracy = printValAccuracy(result, labels_val)
    
    # Training modified MLP using train data in session_02
    start = time.time()
    modelOutputDirName = mainDirName + args.dataSetInUse + '_trained_models_' + sessionNames[sessIndex]
    os.mkdir(modelOutputDirName)
    epochs = 500 #400 #1000
    lr = 0.001
    
    logger.info(f"Running MLP training with LR {lr} for {epochs} epochs")
    classes_in_consideration = torch.from_numpy(np.arange(num_classes_new))
    net_obj.training_and_eval(classes_in_consideration, features, labels, features_val, labels_val, epochs, lr, batch_size, modelOutputDirName)
    end = time.time()
    print('Time took to train: ', end - start)
    
    # save the network 
    net_obj.save_model(epochs, modelOutputDirName)
    
    # inference for session_02 data after training
    print('inference for data after training')
    result = net_obj.inference(features_val)
    top1_accuracy = printValAccuracy(result, labels_val)
    Top1_Performance.append(top1_accuracy)
    
    
  print('###############################################################################################')
  print('################## Top-1 Accuracy of All Incremental ##########################################')
  for k in range(0,numOfSessionInThisDataSet):
    print('session # ' + str(k) + ': ', Top1_Performance[k])
    

def printValAccuracy(result, labels_val):
  
  values, indices = torch.max(result, 1)
  predictions = indices.cpu().numpy()
  sm_scores = values.cpu().numpy()
  
  correctCount = 0
  
  for k in range(len(predictions)):
    if predictions[k] == labels_val[k]:
      correctCount += 1
  
  print('# of samples: ', len(predictions))
  print('# of samples predicted correctly: ', correctCount)
  print('% of samples predicted correctly: ', 100.0 * correctCount / len(predictions))
  
  top1_accuracy = 100.0 * correctCount / len(predictions)
  
  return top1_accuracy 
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="This script performs training of MLP using concatenated featurs for FSCIL.")
  parser.add_argument("--dataSetInUse",
                        default='CUB',
                        help="The dataset for which one wants to run FSCIL, options are 'CUB', 'cifar100', and 'miniimagenet'.")
  parser.add_argument("--featureBeingUsed",
                        default='concat',
                        help="Which features you want to use, options are 'concat', 'selfsupervised_only', and 'supervised_only'.")
  parser.add_argument("--NUMBER_OF_FEATURE_FILES",
                        default=60,
                        type=int,
                        help="Number of augmented files features per image.")
  parser.add_argument("--gnerateData", help=" set True if want to use generated data",
                        default=False, action="store_true")
  parser.add_argument("--useVectorVariance", help=" set True if want to use vector for variance instead of scalar",
                        default=False, action="store_true")
  parser.add_argument("--base_epochs",
                        default=500,
                        type=int,
                        help="number of epochs for base session")
  parser.add_argument("--root_path",
                        default='/scratch/tahmad/FSCIL_datasets/',
                        help="Name of root directory where features are held.")
  parser.add_argument("--dirSelfSupervisedFeatures",
                        default='CUB_200_deepclusterv2_ResNet50_features_imagenet/',
                        help="Name of directory where self-supervised features are held.")
  parser.add_argument("--dirSupervisedFeatures",
                        default='CUB_200_CEC_ResNet18_features_batch0/',
                        help="Name of directory where supervised features are held.")
  
  args = parser.parse_args()                      
  
  main(args)
  
