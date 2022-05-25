# Adaptive_Dual_Attention_Module
My implementation of the paper [Wu, Huisi, et al. "Automated skin lesion segmentation via an adaptive dual attention module." IEEE Transactions on Medical Imaging 40.1 (2020): 357-370.]


### Project Modules
> * `adamnet.py` net architecture definitions including ADAM and several ADAMNets.
> * `dataset.py` ISIC-2017 dataset preprocessing and relavent augmentation.
> * `indicators.py` model performance indicator definitions including DCS, JSI, and etc.
> * `loss_fn.py` DiceCE loss function.
> * `pretrained_models.py` URLs of pretrained ResNet models.
> * `train.py` ADAMNet instance creation, ADAMNet training and testing.
> * `utils.py` ResNet creation helper.