# cnn-segmentation
CNN based Deep Learning project for image segmentation.
Heavily relies on transfer learning.
 
## Running the scripts

You can run : 
- training for finetuning
- training for feature extraction
- inference

...depending on the values of the booleans *train*, and *ft*

See PyTorch's definitions of these terms [link](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)

A DeepLab MobileNet example is already there, you can load it by setting *model_choice* to 'dlab_large'

The expected file structure if you use the MS COCO Dataset locally is :
 
.  
├── data  
│   ├── images  
│   │   ├── train  
│   │   ├── val  
│   │   └── test  
│   └── annotations  
└── ...  

## Running the GUI

You can use it without the MS COCO dataset if you set the 'coco' parameter to False

Kaggle notebook used for training : [link](https://www.kaggle.com/code/thomassirvent/semantic-segmentation-with-pytorch)
