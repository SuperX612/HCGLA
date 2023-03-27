# HCGLA

[English](README.md) | [中文](README_zh.md)

<br/>

## 1. 简介

这个项目是论文[Using Highly Compressed Gradients in Federated Learning for Data Reconstruction Attacks](https://ieeexplore.ieee.org/document/10003066) 的实验部分。 该项目主要实现了HCGLA，一种新的抗梯度高压缩的数据重建攻击方法。

<br/>

## 2. 环境配置

请运行以下命令以安装复现该项目所需要的库:
```
pip install -r requirements.txt
```

<br/>

## 3. 攻击样例

| ![example_batchsize1](readmeimg/example_batchsize1.png)      |   ![example_batchsize4](readmeimg/example_batchsize4.png)    |
| :----------------------------------------------------------- | :----------------------------------------------------------: |
| (a) HCGLA (*Init-Generation*)在梯度压缩率为0.1%且被攻击数据batch大小为1的场景下攻击经典数据集的可视化结果。 | (b) HCGLA (*Init-Generation*)在梯度压缩率为0.1%且被攻击数据batch大小为4的场景下攻击人脸数据集CelebA的可视化结果。 |

<br/>

## 4. 使用此项目

### 4.1 准备数据集和我们训练的模型:

通过下表中的链接可以下载我们论文中用到的数据集。然后参照项目结构文件 [ProjectStructure.txt](ProjectStructure.txt) 把数据集放在相应的位置。

| Dataset name |                        Download link                         |
| :----------: | :----------------------------------------------------------: |
|    CelebA    | [CelebA Dataset (cuhk.edu.hk)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
|     LFW      | [LFW Face Database : Main (umass.edu)](http://vis-www.cs.umass.edu/lfw/) |
|   Pubface    | 链接：https://pan.baidu.com/s/1UYBx_Wd37ngo8iwfKoranQ , 密码：ahlb |
| GoogleImage  | 链接：https://pan.baidu.com/s/157bfS8JkquwIgjop0GgHEQ , 密码：eia1 |
|   CIFAR10    | [CIFAR-10 and CIFAR-100 datasets (toronto.edu)](http://www.cs.toronto.edu/~kriz/cifar.html) |
|   CIFAR100   | [CIFAR-10 and CIFAR-100 datasets (toronto.edu)](http://www.cs.toronto.edu/~kriz/cifar.html) |
|   ImageNet   |      [ImageNet (image-net.org)](https://image-net.org/)      |
|    MNIST     | 链接：https://pan.baidu.com/s/1YinvpHh1wxfN-LxRJ5bLOA , 密码：j4ew |
|    FMNIST    | 链接：https://pan.baidu.com/s/18itHnRISvdJE1SL2gbto8g , 密码：2nlz |

通过下表中的链接可以下载我们训练好的模型，您也可以通过4.2、4.3、4.4的步骤用自己的数据集训练模型。然后参照项目结构文件 [ProjectStructure.txt](ProjectStructure.txt) 把模型放在相应的位置。

|   Model name    |                        Download link                         |
| :-------------: | :----------------------------------------------------------: |
| Denoising model | 链接：https://pan.baidu.com/s/1EhFDwx8Z4Y3pLySPTz1iLg , 密码：vrbs |
| Generator model | 链接：https://pan.baidu.com/s/1ffmEis1uffoYB69BN_k2pA , 密码：2p5e |

### 4.2 训练去噪模型:

你需要重复足够次数的数据重建攻击，以便从重建过程中收集足够多的噪声图像，然后用原始图像和这些收集的噪声图像训练去噪模型。

步骤 1，请运行以下命令以收集足够多数据重建攻击过程中产生的图像:

```shell
python ConnectNoisyimg.py
```

步骤 2，请运行以下命令训练适用于数据重建攻击的去噪模型:

```shell
python Train_DnCNN.py
```

步骤 3，请运行以下命令测试上一步训练得到去噪模型:

```shell
python Test_DnCNN.py
```

然后你可以在`./models/DenoisingModel/DnCNN`文件夹下看到训练好的模型。 你也可以通过以下链接下载我们训练的去噪模型：https://pan.baidu.com/s/1EhFDwx8Z4Y3pLySPTz1iLg，其密码为：vrbs。

### 4.3 训练生成初始化中的生成器:

当batchsize=1时，你可以通过参考[Gradinv.html](https://pan.baidu.com/s/1p1qzDWuVk_Emvt26Ru_erQ?pwd=k89m)文件来训练和测试一个生成器。当batchsize！=1时，你可以通过运行`python Train_batchsize_generator.py`来训练一个生成器，然后你可以通过运行`python Test_batchsize_generator.py`来测试一个生成器。你可以通过以下链接下载生成器：https://pan.baidu.com/s/1ffmEis1uffoYB69BN_k2pA ，密码：2p5e。由于实验室设备的计算能力有限，我们没有训练多批次、ResNet生成器，但如果你需要，你也可以自己训练ResNet、多批次攻击场景生成器。

### 4.4 发起HCGLA重建数据:

在准备好所有的数据集和模型之后，你可以发起HCGLA。如果你想在 batchsize = 1的情况下发起HCGLA，请运行： `python Reconstruct_batchsize1.py`。如果你想攻击在batchsize !=1的情况下发起HCGLA，运行： `python Reconstruct_minibatch.py`。

### 4.5 注意:

`你可以修改参数设置以适应你的需要`

<br/>

## 5. 引用

如果你觉得我们的工作对你的研究有用，请考虑引用:

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

## 6. 联系

如果你有任何问题，请通过电子邮件与我联系 kunlan_xiang@163.com。