# FeSSSS

Implementation for our CVPR workshop paper listed below:  

**[Few-Shot Class Incremental Learning Leveraging Self-Supervised Features](https://drive.google.com/file/d/1NMiVTmSzoa8s2isFVhmAqHg8ChZXhQMo/view?usp=sharing)**, [CVPR-Workshops 2022.](https://cvpr2022.thecvf.com/)

Authors: [Touqeer Ahmad](https://sites.google.com/site/touqeerahmadsite/Touqeer?authuser=0), [Akshay Raj Dhamija](https://akshay-raj-dhamija.github.io/), Steve Cruz, Ryan Rabinowitz, Chunchun Li, [Mohsen Jafarzadeh](http://www.mohsen-jafarzadeh.com/index.php), and [Terrance E. Boult](https://vast.uccs.edu/~tboult/) 

The paper is focused on enhanced performance of FSCIL by concatenating self-supervised and supervised features respectively learned from a disjoint unlabeled data set and data from the base-session of the FSCIL setting. Both types of features are extracted and pre-saved as the first step.

Mainly our code builds upon the CVPR 2021 paper on FSCIL called [Continually Evolved Classifiers](https://github.com/icoz69/CEC-CVPR2021). For ease we provide our code integrated into CEC repo where we have made changes to their original code, for licensing of CEC, please consult authors' original repo.  


## Self-Supervised Feature Extraction
To extract the self-supervised features, run the feature extractor ```FeatureExtraction_MocoV2.py``` in respective data set directory by providing the path to the pre-trained self-supervised model and other required arguments. For example for cub200, the above file is located in CEC_CVPR2021/dataloader/cub200/ directory. A bash file (```run_feature_extractor_mocov2_rest_of_sessions.sh```) demonstrates running feature extractor for all incremental sessions and saving self-supervised features for 60 random crops per each training sample and one central crop per each validation sample.   

## Self-Supervised Pre-Trained Models
We have used the following self-supervised models respectively for cub200, mini_imagenet, and cifar100 datasets. To avoid overlap between mini_imagenet and imagenet, OpenImages based self-supervised features learned through MoCo-v2 are used instead.  
### CUB200: 
  1. ILSVRC-2012 based ResNet50 trained via DeepCluster-v2
  2. ILSVRC-2012 based ResNet50 trained via MoCo-v2
  3. ILSVRC-2012 based ResNet50 trained via SeLa-v2
  4. ILSVRC-2012 based ResNet50 trained via SwAV

### mini_ImageNet:
  1. OpenImages-v6 based ResNet50 trained via MoCo-v2
  3. ILSVRC-2012 based ResNet50 trained via DeepCluster-v2 (for ablation study)
  4. ILSVRC-2012 based ResNet50 trained via MoCo-v2 (for ablation study)
  5. ILSVRC-2012 based ResNet50 trained via SeLa-v2 (for ablation study)
  6. ILSVRC-2012 based ResNet50 trained via SwAV (for ablation study)

### CIFAR100: 
  1. ILSVRC-2012 based ResNet50 trained via DeepCluster-v2
  2. ILSVRC-2012 based ResNet50 trained via MoCo-v2
  3. ILSVRC-2012 based ResNet50 trained via SeLa-v2
  4. ILSVRC-2012 based ResNet50 trained via SwAV

The pre-trained self-supervised models are provided from the following [Google drive link]().   


