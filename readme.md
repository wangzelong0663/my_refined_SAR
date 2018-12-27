# Refining the Simulated SAR Images Based on the Deep Fully Convolutional Networks
This is a Tensorflow implementation of turing the SAR simulated images into real images.
We use the model like the pix2pix, some of my code is from [pix2pix github](https://github.com/phillipi/pix2pix).


![image](https://github.com/wangzelong0663/my_refined_SAR/raw/master/image/1.png)<br>

## About the Simulate tool
The [SARbake](http://www2.compute.dtu.dk/~dmal/project.html) tool is used to generate the image. And the Simulated data is from this [link](https://data.mendeley.com/datasets/jxhsg8tj7g/2/files/34d52e09-4f9b-41b3-899f-29fa272c2d8e). The real data is from MSTAR.

## U_net and GAN,L1,L2 loss
Thr U-net model just like the figure below<br>
![image](https://github.com/wangzelong0663/my_refined_SAR/raw/master/image/2.png)<br>

## How to use
```python
python pix2pix_new.py --checkpoint_dir ./checkpoint_dir_train17_L1_GAN --is_L2 False --gan_weight 1
```
the parameter of is_L2 (False) stands for L1 and True for L2 loss. As for gan_weight, 0 stands for donot use gan, 1 for using gan. And please donot forget  to modify the checkpoint_dir for different mode.

The classfication model is using the CNN blow:
![image](https://github.com/wangzelong0663/my_refined_SAR/raw/master/image/3.png)<br>

## the result
After running the 1000 epoch.
![image](https://github.com/wangzelong0663/my_refined_SAR/raw/master/image/4.png)<br>
from the left to right: the simulated data, the real data, L1, L1+gan, L2,L2+GAN.

## the data will upload to baiduyun or somewhere soon!
