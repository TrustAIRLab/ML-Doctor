import os
import torch
import pandas
import torchvision
import torch.nn as nn
import PIL.Image as Image
import torchvision.transforms as transforms

from functools import partial
from torchvision.models.resnet import ResNet, BasicBlock
from typing import Any, Callable, List, Optional, Union, Tuple

class CNN(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*6*6, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class UTKFaceDataset(torch.utils.data.Dataset):
    """
    UTKFace Dataloader.

    Args:
        root: the file path of UTKFace.

        attr: the attributes of the images, including race and gender.

        transform and target_transform are the augmentations of pytorch.
    """
    def __init__(self, root, attr: Union[List[str], str] = "gender", transform=None, target_transform=None)-> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.processed_path = os.path.join(self.root, 'UTKFace/processed/')
        self.files = os.listdir(self.processed_path)
        if isinstance(attr, list):
            self.attr = attr
        else:
            self.attr = [attr]

        self.lines = []
        for txt_file in self.files:
            txt_file_path = os.path.join(self.processed_path, txt_file)
            with open(txt_file_path, 'r') as f:
                assert f is not None
                for i in f:
                    image_name = i.split('jpg ')[0]
                    attrs = image_name.split('_')
                    if len(attrs) < 4 or int(attrs[2]) >= 4  or '' in attrs:
                        continue
                    self.lines.append(image_name+'jpg')


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index:int)-> Tuple[Any, Any]:
        attrs = self.lines[index].split('_')

        age = int(attrs[0])
        gender = int(attrs[1])
        race = int(attrs[2])

        image_path = os.path.join(self.root, 'UTKFace/raw/', self.lines[index]+'.chip.jpg').rstrip()

        image = Image.open(image_path).convert('RGB')

        target: Any = []
        for t in self.attr:
            if t == "age":
                target.append(age)
            elif t == "gender":
                target.append(gender)
            elif t == "race":
                target.append(race)
            
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform:
            image = self.transform(image)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return image, target

class CelebA(torch.utils.data.Dataset):
    base_folder = "celeba"

    """
    CelebA Dataloader.

    Args:
        root: the file path of UTKFace.

        attr_list: the attributes of CelebA dataset, including heavyMakeup, MouthSlightlyOpen, Smiling, Male, and Young.
            We choose these labels because they are camparely fair in the dataset.

        target_type: AttrInf needs two attributes, so target_tpye should be a list. Other attacks only need one.
        
        transform and target_transform are the augmentations of pytorch.
    """

    def __init__(
            self,
            root: str,
            attr_list: str,
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        self.root = root
        self.transform = transform
        self.target_transform =target_transform
        self.attr_list = attr_list

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None)

        self.filename = splits[mask].index.values
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = Image.open(os.path.join(self.root, self.base_folder, "img_celeba", self.filename[index]))

        target: Any = []
        for t, nums in zip(self.target_type, self.attr_list):
            if t == "attr":
                final_attr = 0
                for i in range(len(nums)):
                    final_attr += 2 ** i * self.attr[index][nums[i]]
                target.append(final_attr)
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

def prepare_dataset(dataset, attr, root, model_name):
    num_classes, dataset, target_model, shadow_model = get_model_dataset(dataset, attr, root, model_name)
    length = len(dataset)
    each_length = length//4
    target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(dataset, [each_length, each_length, each_length, each_length, len(dataset)-(each_length*4)])
    
    return num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model


def get_model_dataset(dataset_name, attr, root, model_name):
    if dataset_name.lower() == "utkface":
        if isinstance(attr, list):
            num_classes = []
            for a in attr:
                if a == "age":
                    num_classes.append(117)
                elif a == "gender":
                    num_classes.append(2)
                elif a == "race":
                    num_classes.append(4)
                else:
                    raise ValueError("Target type \"{}\" is not recognized.".format(a))
        else:
            if attr == "age":
                num_classes = 117
            elif attr == "gender":
                num_classes = 2
            elif attr == "race":
                num_classes = 4
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(attr))

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = UTKFaceDataset(root=root, attr=attr, transform=transform)
        input_channel = 3
        
    elif dataset_name.lower() == "celeba":
        if isinstance(attr, list):
            for a in attr:
                if a != "attr":
                    raise ValueError("Target type \"{}\" is not recognized.".format(a))

                num_classes = [8, 4]
                # heavyMakeup MouthSlightlyOpen Smiling, Male Young
                attr_list = [[18, 21, 31], [20, 39]]
        else:
            if attr == "attr":
                num_classes = 8
                attr_list = [[18, 21, 31]]
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(attr))

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = CelebA(root=root, attr_list=attr_list, target_type=attr, transform=transform)
        input_channel = 3

    elif dataset_name.lower() == "stl10":
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = torchvision.datasets.STL10(
                root=root, split='train', transform=transform, download=True)
            
        test_set = torchvision.datasets.STL10(
                root=root, split='test', transform=transform, download=True)

        dataset = train_set + test_set
        input_channel = 3

    elif dataset_name.lower() == "fmnist":
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_set = torchvision.datasets.FashionMNIST(
                root=root, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(
                root=root, train=False, download=True, transform=transform)

        dataset = train_set + test_set
        input_channel = 1

    if isinstance(num_classes, int):
        if model_name.lower() == 'alexnet':
            target_model = torchvision.models.alexnet(num_classes=num_classes)
            shadow_model = torchvision.models.alexnet(num_classes=num_classes)

            if dataset_name.lower() == "fmnist":
                target_model.features[0] = nn.Conv2d(input_channel, 64, kernel_size=11, stride=4, padding=2)
                shadow_model.features[0] = nn.Conv2d(input_channel, 64, kernel_size=11, stride=4, padding=2)

        elif model_name.lower() == 'resnet18':
            target_model = torchvision.models.resnet18(num_classes=num_classes)
            shadow_model = torchvision.models.resnet18(num_classes=num_classes)

            if dataset_name.lower() == "fmnist":
                target_model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
                shadow_model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        elif model_name.lower() == 'resnet50':
            target_model = torchvision.models.resnet50(num_classes=num_classes)
            shadow_model = torchvision.models.resnet50(num_classes=num_classes)

            if dataset_name.lower() == "fmnist":
                target_model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
                shadow_model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        elif model_name.lower() == 'vgg19':
            target_model = torchvision.models.vgg19_bn(num_classes=num_classes)
            shadow_model = torchvision.models.vgg19_bn(num_classes=num_classes)

            if dataset_name.lower() == "fmnist":
                target_model.features[0] = nn.Conv2d(input_channel, 64, kernel_size=3, padding=1)
                shadow_model.features[0] = nn.Conv2d(input_channel, 64, kernel_size=3, padding=1)

        elif model_name.lower() == 'cnn':
            target_model = CNN(input_channel=input_channel, num_classes=num_classes)
            shadow_model = CNN(input_channel=input_channel, num_classes=num_classes)
        
    else:
        if model_name.lower() == 'alexnet':
            target_model = torchvision.models.alexnet(num_classes=num_classes[0])
            shadow_model = torchvision.models.alexnet(num_classes=num_classes[0])

            if dataset_name.lower() == "fmnist":
                target_model.features[0] = nn.Conv2d(input_channel, 64, kernel_size=11, stride=4, padding=2)
                shadow_model.features[0] = nn.Conv2d(input_channel, 64, kernel_size=11, stride=4, padding=2)

        elif model_name.lower() == 'resnet18':
            target_model = torchvision.models.resnet18(num_classes=num_classes[0])
            shadow_model = torchvision.models.resnet18(num_classes=num_classes[0])

            if dataset_name.lower() == "fmnist":
                target_model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
                shadow_model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        elif model_name.lower() == 'vgg19':
            target_model = torchvision.models.vgg19_bn(num_classes=num_classes[0])
            shadow_model = torchvision.models.vgg19_bn(num_classes=num_classes[0])

            if dataset_name.lower() == "fmnist":
                target_model.features[0] = nn.Conv2d(input_channel, 64, kernel_size=3, padding=1)
                shadow_model.features[0] = nn.Conv2d(input_channel, 64, kernel_size=3, padding=1)

        elif model_name.lower() == 'cnn':
            target_model = CNN(input_channel=input_channel, num_classes=num_classes[0])
            shadow_model = CNN(input_channel=input_channel, num_classes=num_classes[0])


    return num_classes, dataset, target_model, shadow_model