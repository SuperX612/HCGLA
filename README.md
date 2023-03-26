# HCGLA

## 1. Introduction

- This is an official implementation of the following paper:*[Using Highly Compressed Gradients in Federated Learning for Data Reconstruction Attacks](https://ieeexplore.ieee.org/document/10003066)*.

- We present a novel data leakage attack algorithm against highly compressed gradients.
## 2. Requirements

- If your device is `GPU` please running:

```
pip install -r requirements_GPU.txt
```
- If your device is `CPU` please running:

```
pip install -r requirements_CPU.txt
```

## 3. Examples

| ![example_batchsize1](readmeimg/example_batchsize1.png)      |   ![example_batchsize4](readmeimg/example_batchsize4.png)    |
| :----------------------------------------------------------- | :----------------------------------------------------------: |
| (a) Visualization of HCGLA (*Init-Generation*) on  popular datasets at a 0.1% compression rate with batchsize=1. | (b) Visualization of HCGLA (*Init-Generation*) on  popular facial dataset at a 0.1% compression rate with batchsize=4. |

## 4. How to use
- **Prepare dataset:**

You can download the dataset used in our paper by the following table.

| Dataset name |                        Download link                         |
| :----------: | :----------------------------------------------------------: |
|    CelebA    | [CelebA Dataset (cuhk.edu.hk)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
|     LFW      | [LFW Face Database : Main (umass.edu)](http://vis-www.cs.umass.edu/lfw/) |
|   Pubface    | 链接：https://pan.baidu.com/s/1UYBx_Wd37ngo8iwfKoranQ <br/>提取码：ahlb |
| GoogleImage  | 链接：https://pan.baidu.com/s/157bfS8JkquwIgjop0GgHEQ <br/>提取码：eia1 |
|   CIFAR10    | [CIFAR-10 and CIFAR-100 datasets (toronto.edu)](http://www.cs.toronto.edu/~kriz/cifar.html) |
|   CIFAR100   | [CIFAR-10 and CIFAR-100 datasets (toronto.edu)](http://www.cs.toronto.edu/~kriz/cifar.html) |
|   ImageNet   |      [ImageNet (image-net.org)](https://image-net.org/)      |
|    MNIST     | 链接：https://pan.baidu.com/s/1YinvpHh1wxfN-LxRJ5bLOA <br/>提取码：j4ew |
|    FMNIST    | 链接：https://pan.baidu.com/s/18itHnRISvdJE1SL2gbto8g <br/>提取码：2nlz |

- **Train denoising model:**

You need to demonstrate the data reconstruction attack enough times to collect enough noisy images from the reconstruction process and then train a denoising model with the original images and these collected noisy images.

To connect enough noising images by running:

```
python ConnectNoisyimg.py
```

To train denoising model by running:

```cmd
python Train_DnCNN.py
```

To test denoising model by running:

```cmd
python Test_DnCNN.py
```

Then you can see the trained model under the `./models/DenoisingModel/DnCNN` folder. 

==You can modify the parameter settings to suit your needs==

- **Train and test generator in *Init-Generation*:**

When batchsize=1, You can train and test a generator by referring to the  [Gradinv.html](Gradinv.html)  file.

When batchsize!=1, you can train a generator by running `python Train_batchsize_generator.py`, Then you can test a generator by running `python Test_batchsize_generator.py`.

Due to the limited computing power of the lab equipment, we did not train the multi-batch, ResNet generator, but you can also train the ResNet, multi-batch attack scenario generator by yourself if you need.

- **Train HCGLA and Recover data, run:**

- **Notes:**



## 5. Colab

## 6. Citation

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

## 7. Contact

If you have any questions, please contact me via email kunlan_xiang@163.com

