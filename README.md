# cnn-segmentation
CNN based Deep Learning project for image segmentation.
Heavily relies on transfer learning.

## Training
The available model has been trained with Kaggle, to take adavantage of the GPU. [Kaggle notebook](https://www.kaggle.com/code/thomassirvent/semantic-segmentation-with-pytorch)
 
## Running the scripts

You can run the *main.py* script in order to : 
- train for finetuning
- train for feature extraction
- run inferences

...depending on the values of the booleans **train**, and **ft**

See PyTorch's definitions of these terms. [Pytorch's documentation](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)

A DeepLab MobileNet example is already there, you can load it by setting **model_choice** to 'dlab_large'

The expected file structure if you use the MS COCO Dataset locally is :
 
In the data file : 
- images  
    - train  
    - val  
    - test  
- annotations 

## Running the GUI

You can run the *gui.py* script
You can use it without the MS COCO dataset if you set the **coco** parameter to False

Kaggle notebook used for training : [link](https://www.kaggle.com/code/thomassirvent/semantic-segmentation-with-pytorch)
