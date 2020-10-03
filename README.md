* Repository github url : https://github.com/jai-mr/
* Assignment Repository : https://github.com/jai-mr/Assignment-10
* Submitted by : Jaideep Rangnekar
* Registered email id : jaideepmr@gmail.com

## Assignment: 

1. Pick your last code
2. Add CutOut to your code. It should come from your transformations (albumentations)
3. Use this repo: https://github.com/davidtvs/pytorch-lr-finder
4. Move LR Finder code to your modules
5. Implement LR Finder (for SGD, not for ADAM)
6. Implement ReduceLROnPlatea: https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
7. Find best LR to train your model
8. Use SDG with Momentum
9. Train for 50 Epochs. 
10. Show Training and Test Accuracy curves
11. Target 88% Accuracy.
12. Run GradCAM on the any 25 misclassified images. Make sure you mention what is the prediction and what was the ground truth label.
13. Submit

## Details for Assignment Executed
### * [Jupyter Notebook File reference executed in Colab](https://github.com/jai-mr/Assignment-10/blob/main/10_CodeFinal.ipynb)

### * [Data View](https://github.com/jai-mr/Assignment-10/blob/main/images/DataView.png)

### * [Mis-Classified Images](https://github.com/jai-mr/Assignment-10/blob/main/images/MisClassified.png)

### * [GradCam-Misclassified images wrt Predicted(wrong) class](https://github.com/jai-mr/Assignment-10/blob/main/images/GradCam-Misclassified%20images%20wrt%20Predicted(wrong)%20class.png)

### * [GradCam-Misclassified images wrt Actual(correct) class](https://github.com/jai-mr/Assignment-10/blob/main/images/GradCam-Misclassified%20images%20wrt%20Actual(correct)%20class.png)

### * [LR Finder](https://github.com/jai-mr/Assignment-10/blob/main/images/LR%20Finder.png)

### * [Changed LR](https://github.com/jai-mr/Assignment-10/blob/main/images/ChangeLR.png)

### * [Training/Test - Loss & Accuracy Curve](https://github.com/jai-mr/Assignment-9/blob/master/images/TrainTestLossAcc.png)

### * [Test vs Train Accuracy](https://github.com/jai-mr/Assignment-10/blob/main/images/TestvsTrain.png)

### * Test Accuracy : 92.51%

### * Class wise accuracies
* Accuracy of plane : 93 %
* Accuracy of   car : 97 %
* Accuracy of  bird : 89 %
* Accuracy of   cat : 83 %
* Accuracy of  deer : 92 %
* Accuracy of   dog : 88 %
* Accuracy of  frog : 95 %
* Accuracy of horse : 93 %
* Accuracy of  ship : 95 %
* Accuracy of truck : 95 %


## **ReduceLROnPlateau:**

CLASS torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)[SOURCE]

* Reduce learning rate when a metric has stopped improving. 
* Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. 
* This scheduler reads a metrics quantity and if no improvement is seen for a ‘patience’ number of epochs, the learning rate is reduced.

### Parameters:	
* optimizer (Optimizer) – Wrapped optimizer.
* mode (str) – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
* factor (float) – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
* patience (int) – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
* verbose (bool) – If True, prints a message to stdout for each update. Default: False.
* threshold (float) – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
* threshold_mode (str) – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
* cooldown (int) – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
min_lr (float or list) – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
* eps (float) – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.

### Detailed parameters

* Model: min and max models, taking min as an example. When the optimized indicator is not falling, change the learning rate. Generally use min mode, when using, first declare the class, *then scheduler.step (test_acc), the brackets are the indicators generally use the loss of the validation set.
* Factor:new_lr = lr * factor, default 0.1
* Patience: When several epochs are unchanged, the learning rate is changed. The default is 10
* Verbose: whether to print out the information

## Albumentation
Albumentation is a fast image augmentation library and easy to use with other libraries as a wrapper. The package is written on NumPy, OpenCV, and imgaug. What makes this library different is the number of data augmentation techniques that are available. While most of the augmentation libraries include techniques like cropping, flipping, rotating and scaling, albumentation provides a range of very extensive image augmentation techniques like contrast, blur and channel shuffle. Here is the range of augmentations that can be performed. 

The real power of albumentation is in pipelining different transformations for the image at once. 

* CLAHE : Contrast Limited Adaptive Histogram Equalization to equalize images
* Cutout : takes out a part of the image that is not very important for classification.
* Random rotate : rotates the image by a certain degree
* Blur : that reduces the intensity of pixels to appear blur
* Optical distortion : This distorts certain elements of the image.
* ShiftScaleRotate : Allows you to scale and rotate the image by certain angles. 


[Image Transformations available in Albumentations-1](https://github.com/jai-mr/Assignment-9/blob/master/images/albumentation.png?raw=true)

[Image Transformations available in Albumentations-2](https://github.com/jai-mr/Assignment-9/blob/master/images/albumentation2.png)

References:

[Learning Rate Scheduling](https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/lr_scheduling/)

[ReduceLROnPlateau learning rate adjustment](https://www.programmersought.com/article/5488495227/)

[Albumentations](https://github.com/albumentations-team/albumentations)
