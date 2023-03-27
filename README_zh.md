# HCGLA

[English](README.md) | [中文](README_zh.md)

## 1. 简介

这个项目是论文*[Using Highly Compressed Gradients in Federated Learning for Data Reconstruction Attacks](https://ieeexplore.ieee.org/document/10003066)*的实验部分。 该项目主要实现了HCGLA，一种新的抗梯度高压缩的数据重建攻击方法。

## 2. 环境配置

请运行以下命令以安装复现该项目所需要的库:

```
pip install -r requirements.txt
```

## 3. 攻击样例

| ![example_batchsize1](readmeimg/example_batchsize1.png)      |   ![example_batchsize4](readmeimg/example_batchsize4.png)    |
| :----------------------------------------------------------- | :----------------------------------------------------------: |
| (a) HCGLA (*Init-Generation*)在梯度压缩率为0.1%且被攻击数据batch大小为1的场景下攻击经典数据集的可视化结果。 | (b) Visualization of HCGLA (*Init-Generation*) on  popular facial dataset at a 0.1% compression rate with batchsize=4. |

## 4. How to use

- **Prepare dataset and models we trained:**

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

- **Train denoising model:**

You need to demonstrate the data reconstruction attack enough times to collect enough noisy images from the reconstruction process and then train a denoising model with the original images and these collected noisy images.
Step 1: To connect enough noising images by running:

```cmd
python ConnectNoisyimg.py
```

Step 2: To train denoising model by running:

```cmd
python Train_DnCNN.py
```

Step 3: To test denoising model by running:

```cmd
python Test_DnCNN.py
```

Then you can see the trained model under the `./models/DenoisingModel/DnCNN` folder.  You can also download the denoising model we trained by the link：https://pan.baidu.com/s/1EhFDwx8Z4Y3pLySPTz1iLg and its password：vrbs.

- **Train and test generator in *Init-Generation*:**

When batchsize=1, You can train and test a generator by referring to the  [Gradinv.html](https://pan.baidu.com/s/1p1qzDWuVk_Emvt26Ru_erQ?pwd=k89m) file. When batchsize!=1, you can train a generator by running `python Train_batchsize_generator.py`, Then you can test a generator by running `python Test_batchsize_generator.py`. You can download the generators by the link：https://pan.baidu.com/s/1ffmEis1uffoYB69BN_k2pA , password：2p5e. Due to the limited computing power of the lab equipment, we did not train the multi-batch, ResNet generator, but you can also train the ResNet, multi-batch attack scenario generator by yourself if you need.

- **Train HCGLA and Recover data:**

After prepare all the datasets and models you can launch HCGLA.
If you want to launch HCGLA with batchsize = 1, run:

```
python Reconstruct_batchsize1.py
```

batchsize != 1, run:

```
Reconstruct_minibatch.py
```

- **Notes:**

`You can modify the parameter settings to suit your needs`

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

## 6. Contact

If you have any questions, please contact me via email kunlan_xiang@163.com