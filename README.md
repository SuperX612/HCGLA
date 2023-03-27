# HCGLA

[English](README.md) | [中文](README_zh.md)

<br/>

## 1. Introduction

This is an official implementation of our paper:*[Using Highly Compressed Gradients in Federated Learning for Data Reconstruction Attacks](https://ieeexplore.ieee.org/document/10003066)*. We present a novel data leakage attack algorithm against highly compressed gradients.

<br/>

## 2. Requirements

To install the requirements, run:

```shell
pip install -r requirements.txt
```

<br/>

## 3. Examples

| ![example_batchsize1](readmeimg/example_batchsize1.png)      |   ![example_batchsize4](readmeimg/example_batchsize4.png)    |
| :----------------------------------------------------------- | :----------------------------------------------------------: |
| (a) Visualization of HCGLA (*Init-Generation*) on  popular datasets at a 0.1% compression rate with batchsize=1. | (b) Visualization of HCGLA (*Init-Generation*) on  popular facial dataset at a 0.1% compression rate with batchsize=4. |

<br/>

## 4. How to use

### 4.1 Prepare dataset and models we trained:

You can download the dataset used in our paper by the following table. Then place these datasets in the corresponding locations refering  [ProjectStructure.txt](ProjectStructure.txt).

| Dataset name |                        Download link                         |
| :----------: | :----------------------------------------------------------: |
|    CelebA    | [CelebA Dataset (cuhk.edu.hk)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
|     LFW      | [LFW Face Database : Main (umass.edu)](http://vis-www.cs.umass.edu/lfw/) |
|   Pubface    | link：https://pan.baidu.com/s/1UYBx_Wd37ngo8iwfKoranQ , password：ahlb |
| GoogleImage  | link：https://pan.baidu.com/s/157bfS8JkquwIgjop0GgHEQ , password：eia1 |
|   CIFAR10    | [CIFAR-10 and CIFAR-100 datasets (toronto.edu)](http://www.cs.toronto.edu/~kriz/cifar.html) |
|   CIFAR100   | [CIFAR-10 and CIFAR-100 datasets (toronto.edu)](http://www.cs.toronto.edu/~kriz/cifar.html) |
|   ImageNet   |      [ImageNet (image-net.org)](https://image-net.org/)      |
|    MNIST     | link：https://pan.baidu.com/s/1YinvpHh1wxfN-LxRJ5bLOA , password：j4ew |
|    FMNIST    | link：https://pan.baidu.com/s/18itHnRISvdJE1SL2gbto8g , password：2nlz |

You can download the models we trained in our experiments by the following table. Then place these models in the corresponding locations refering [ProjectStructure.txt](ProjectStructure.txt).

|   Model name    |                        Download link                         |
| :-------------: | :----------------------------------------------------------: |
| Denoising model | link：https://pan.baidu.com/s/1EhFDwx8Z4Y3pLySPTz1iLg , password：vrbs |
| generator model | link：https://pan.baidu.com/s/1ffmEis1uffoYB69BN_k2pA , password：2p5e |

### 4.2 Train denoising model:

You need to demonstrate the data reconstruction attack enough times to collect enough noisy images from the reconstruction process and then train a denoising model with the original images and these collected noisy images.

Step 1: To connect enough noising images by running:

```shell
python ConnectNoisyimg.py
```

Step 2: To train denoising model by running:

```shell
python Train_DnCNN.py
```

Step 3: To test denoising model by running:

```shell
python Test_DnCNN.py
```

Then you can see the trained model under the `./models/DenoisingModel/DnCNN` folder.  You can also download the denoising model we trained by the link：https://pan.baidu.com/s/1EhFDwx8Z4Y3pLySPTz1iLg and its password：vrbs.

### 4.3 Train and test generator in *Init-Generation*:

When batchsize=1, You can train and test a generator by referring to the  [Gradinv.html](https://pan.baidu.com/s/1p1qzDWuVk_Emvt26Ru_erQ?pwd=k89m) file. When batchsize!=1, you can train a generator by running `python Train_batchsize_generator.py`, Then you can test a generator by running `python Test_batchsize_generator.py`. You can download the generators by the link：https://pan.baidu.com/s/1ffmEis1uffoYB69BN_k2pA , password：2p5e. Due to the limited computing power of the lab equipment, we did not train the multi-batch, ResNet generator, but you can also train the ResNet, multi-batch attack scenario generator by yourself if you need.

### 4.4 Launch HCGLA and Recover data:

After prepare all the datasets and models you can launch HCGLA. If you want to launch HCGLA with batchsize = 1, run: `python Reconstruct_batchsize1.py`. If batchsize != 1, run: `python Reconstruct_minibatch.py`

### 4.5 Notes:

`You can modify the parameter settings to suit your needs`

<br/>

## 5. Citation

If you find our work useful in your research, please consider citing:

```latex
@ARTICLE{10003066,
  author={Yang, Haomiao and Ge, Mengyu and Xiang, Kunlan and Li, Jingwei},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Using Highly Compressed Gradients in Federated Learning for Data Reconstruction Attacks}, 
  year={2023},
  volume={18},
  number={},
  pages={818-830},
  doi={10.1109/TIFS.2022.3227761}}
```

<br/>

## 6. Contact

If you have any questions, please contact me via email kunlan_xiang@163.com
