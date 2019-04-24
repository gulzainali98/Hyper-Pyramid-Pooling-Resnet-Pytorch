# Hyper-Pyramid-Pooling-Resnet-Pytorch
This repository include implementation of hyper pyramid pooling(HPP) mentioned in the paper **"Horizontal Pyramid Matching for Person Re-identification"** written by **Yang Fu1, Yunchao Wei1∗, Yuqian Zhou1, Honghui Shi1, Gao Huang3, Xinchao Wang4, Zhiqiang Yao5, Thomas Huang1**

# Accuracy
While training RESNET-50 was able to achieve a maximum of 96.569% accuracy on CIFAR-10 dataset after 40 Epochs. I believe this will produce better accuracy with further training because model was still converging. 

# Implementation Details
I implemented a function for Hyper Pyramid pooling which takes the feature layer with dimensions 2048x4x4 and slices it into total of three bins at 2 scales. At first scale bin size is 4x4 while at second scale bin size is 2x4. Average pooling and max pooling is applied on all of these bins reducing these to 2048x1x1 dimensions. This is then fed into convolutional layer which reduces the dimensions to 256 from 2048. All of the results are classified, concatenated before returning to the training function.

For loss function i simply calculate loss on chunks of returned output from model and the result of each chunk is added to form final loss. Loss function used is cross entropy as mentioned in the paper. 

Feel free to reuse the code as you like. CHEERS! :)

# Application of HPP:
HPP is primarily used for person re-identification task in the paper. However, i believe this can be utilized at any place where considerable information is given in different chunks of image and the full image is also needed in order to make final judgement. One such example is facial gesture recognition. In facial gesture recognition information is provided in cheeks, forehead and areas around lips but all of this information needs to be considered globally as well to find what kind of expression is being shown. 
