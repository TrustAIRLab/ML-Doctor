# ML-Doctor

[![arXiv](https://img.shields.io/badge/arxiv-2102.02551-b31b1b)](https://arxiv.org/abs/2102.02551)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

This is the code for our USENIX Security 22 paper [ML-Doctor: Holistic Risk Assessment of Inference Attacks Against Machine Learning Models](https://www.usenix.org/conference/usenixsecurity22/presentation/liu-yugeng).
In general, ML-Doctor is a modular framework geared to evaluate the four inference attacks, i.e., Membership Inference (MemInf), Model Inversion (ModInv), Attribute Inference (AttrInv), and Model Stealing (ModSteal). 

## Build Datasets
We prefer the users could provide the dataloader by themselves. But we show the demo dataloader in the code. Due to the size of the dataset, we won't upload it to github.

For UTKFace, we have two folders downloaded from [official website](https://susanqq.github.io/UTKFace/) in the UTKFace folder. The first is the "processed" folder which contains three landmark_list files(also can be downloaded from the official website). It is used to get the image name in a fast way because all the features of the images can be achieved from the file names. The second folder is the "raw" folder which contains all the aligned and cropped images. 

For CelebA dataset, we have one folder and three files in the "celeba" folder. For the "img_celeba" folder, it contains all the images downloaded from the [official website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and we align and crop them by ourselves. The others are three files used to get the attributes or file names, named "identity_CelebA.txt", "list_attr_celeba.txt", and "list_eval_partition.txt". The crop center is \[89, 121\] but it is ok if the users wouldn't like to crop it because we have resize function in the transforms so that it will not affect the input shapes.

For FMNIST and STL10, PyTorch has offered the datasets and they can be easily employed.

## Prepare
Users should install Python3 and PyTorch at first. We recommend using conda to install it based on the official documents.

Or directly run

```
conda env create -f environment.yaml
conda activate ML-Doctor
```

## Evaluate

```python demo.py --attack_type C --dataset T --mode R --model L```

<table><tbody>
<!-- TABLE BODY -->
<tr>
<td align="center">Attack Type</td>
<td align="center">0</td>
<td align="center">1</td>
<td align="center">2</td>
<td align="center">3</td>
</tr>
<tr>
<td align="center">Name</td>
<td align="center">MemInf</td>
<td align="center">ModInv</td>
<td align="center">AttrInf</td>
<td align="center">ModSteal</td>
</tr>
</tbody></table>

For dataset name, there are 4 datasets in the code, namely CelebA, FMNIST (Fashion-MNIST), STL10, and UTKFace.
For the models, we provide standard AlexNet, ResNet, and VGG, from PyTorch, and a CNN model (containing 2 convolutional layers and 2 fully connected layers).

### For MemInf
We have four modes in this function
<table><tbody>
<!-- TABLE BODY -->
<tr>
<td align="center">Mode</td>
<td align="center">0</td>
<td align="center">1</td>
<td align="center">2</td>
<td align="center">3</td>
</tr>
<tr>
<td align="center">Name</td>
<td align="center">BlackBox Shadow</td>
<td align="center">BlackBox Partial</td>
<td align="center">WhiteBox Partial</td>
<td align="center">WhiteBox Shadow</td>
</tr>
</tbody></table>

#### Buliding attack dataset
When using mode 0 and mode 3, i.e. with shadow models, users should choose [```get_attack_dataset_with_shadow```](./doctor/meminf.py#L552) function.
For the others (mode 1 and mode 2), it should be [```get_attack_dataset_without_shadow```](./doctor/meminf.py#L526) function.

#### Choosing attack model
When using mode 0, ```attack_model``` should be [```ShadowAttackModel```](./utils/define_models.py#L15), while [```PartialAttackModel```](./utils/define_models.py#L56) is  ```attack_model``` for mode 1 in blackbox.
For whitebox (mode 2 and mode 3), users need to change ```attack_model``` to [```WhiteBoxAttackModel```](./utils/define_models.py#L97).
Users can also define attack models by themselves so we didn't fix the models here.

Note: we have the same [```ShadowAttackModel```](./utils/define_models.py#L15) and [```PartialAttackModel```](./utils/define_models.py#L56) in the code.

### For ModInv
For Secret Revealer method, users should use a pre-trained model as the evaluation model. In our paper we choose a ResNet18 model and name it as ```{Dataset}_eval.pth``` with the same path as the target model.

### For AttrInf and ModSteal
There are two modes in general, i.e. Partial and Shadow.

For more details, you can check our paper.


## Citation
Please cite this paper in your publications if it helps your research:

    @inproceedings{LWHSZBCFZ22,
    author = {Yugeng Liu and Rui Wen and Xinlei He and Ahmed Salem and Zhikun Zhang and Michael Backes and Emiliano De Cristofaro and Mario Fritz and Yang Zhang},
    title = {{ML-Doctor: Holistic Risk Assessment of Inference Attacks Against Machine Learning Models}},
    booktitle = {{USENIX Security Symposium (USENIX Security)}},
    pages = {4525-4542},
    publisher = {USENIX},
    year = {2022}
    }



## License

ML-Doctor is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail at zhang[AT]cispa.de. We will send the detail agreement to you.
