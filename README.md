# FeSSSS

Implementation for our CVPR workshop paper listed below:  

**[Few-Shot Class Incremental Learning Leveraging Self-Supervised Features](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/html/Ahmad_Few-Shot_Class_Incremental_Learning_Leveraging_Self-Supervised_Features_CVPRW_2022_paper.html)**, [CVPR-Workshops 2022.](https://cvpr2022.thecvf.com/)

Authors: [Touqeer Ahmad](https://sites.google.com/site/touqeerahmadsite/Touqeer?authuser=0), [Akshay Raj Dhamija](https://akshay-raj-dhamija.github.io/), [Steve Cruz](https://scholar.google.com/citations?user=_zl-yoMAAAAJ&hl=en), [Ryan Rabinowitz](https://scholar.google.com/citations?hl=en&user=w-3eXsMAAAAJ), [Chunchun Li](https://scholar.google.com/citations?user=xPJiRT0AAAAJ&hl=en), [Mohsen Jafarzadeh](http://www.mohsen-jafarzadeh.com/index.php), and [Terrance E. Boult](https://vast.uccs.edu/~tboult/) 

The paper is focused on enhanced performance of FSCIL by concatenating self-supervised and supervised features respectively learned from a disjoint unlabeled data set and data from the base-session of the FSCIL setting. Both types of features are extracted and pre-saved as the first step.

Mainly our code builds upon the CVPR 2021 paper on FSCIL called [Continually Evolved Classifiers](https://github.com/icoz69/CEC-CVPR2021). For ease we provide our code integrated into CEC repo where we have made changes to their original code, for licensing of CEC, please consult CEC-authors' original repo.  

## Requirements
Code has been tested in [conda](https://docs.conda.io/en/latest/) virtual env and uses the following: 
- Python = 3.8.5
- [PyTorch = 1.8.1+cu111](https://pytorch.org)


## Self-Supervised Feature Extraction
To extract the self-supervised features, run the feature extractor ```FeatureExtraction_MocoV2.py``` in respective data set directory by providing the path to the pre-trained self-supervised model and other required arguments. For example for cub200, the above file is located in CEC_CVPR2021/dataloader/cub200/ directory. A bash file (```run_feature_extractor_mocov2_rest_of_sessions.sh```) demonstrates running feature extractor for all incremental sessions and saving self-supervised features for 60 random crops per each training sample and one central crop per each validation sample.   

## Self-Supervised Pre-Trained Models
We have used the following self-supervised models respectively for cub200, mini_imagenet, and cifar100 datasets. To avoid overlap between mini_imagenet and imagenet, OpenImages based self-supervised features learned through MoCo-v2 are used instead. The ImageNet based self-supervised features are only used for an ablation of mini_imagenet.   
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

The pre-trained self-supervised models are available from the following [Google drive link](https://drive.google.com/drive/folders/16H2kYIBpNcPCCM7MaT6LjzgEOPv6zY16?usp=sharing).   

## Supervised Feature Extraction
To extract the supervised features learned using the base-session data of FSCIL, run the feature extractor ```FeatureExtraction_MocoV2.py``` in respective data set directory by providing the path to the pre-trained supervised model and other required arguments. While any supervised model can be trained using the base-session data, we have specifically used the CEC-based learned models.

### Checkpoints for CEC-based Supervised Models
The supervised models can be accessed via the Google drive link provided on CEC's [repo](https://github.com/icoz69/CEC-CVPR2021) or our self-archived version from the Google drive link shared above. 


## Training the Two Layer Perceptron
Once the self-supervised and supervised feature are extracted, the multi (two) layer perceptron can be trained using the code provided in trainMLP directory, specifically ```mlpTrain_Concat.py```. A bash file  (```run_script.sh```) demonstrates training the MLP using concatenated features by providing the directories containing respective feature files, and other required arguments. MLP training uses the logger from [VAST repo](https://github.com/Vastlab/vast), please install it following the instructions. Once the training is complete, the log file would be saved in the trainMLP folder due to comman-line redirection as shown in ```run_script.sh```. Average incremental accuracy is reported for all incremental sessions at the end of the log file.           


### BibTeX
If you find our work helpful, please cite the following:

```
@InProceedings{Ahmad_2022_CVPR,
    author    = {Ahmad, Touqeer and Dhamija, Akshay Raj and Cruz, Steve and Rabinowitz, Ryan and Li, Chunchun and Jafarzadeh, Mohsen and Boult, Terrance E.},
    title     = {Few-Shot Class Incremental Learning Leveraging Self-Supervised Features},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {3900-3910}
}
``` 
