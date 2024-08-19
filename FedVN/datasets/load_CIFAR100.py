from torchvision.datasets import CIFAR100
import numpy as np
from PIL import Image
from  torchvision.transforms import transforms


class iCIFAR100(CIFAR100):
    def __init__(self,root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=False):
        super(iCIFAR100,self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)

        self.target_test_transform=target_test_transform
        self.test_transform=test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def getTestData(self, classes):
        datas,labels=[],[]
        
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TestData, self.TestLabels=self.concatenate(datas,labels)
        self.TrainData, self.TrainLabels = [], [] 

    def getTrainData(self, classes):
        datas,labels=[],[]

        for label in classes:
            data=self.data[np.array(self.targets)==label]
            datas.append(data)
            labels.append(np.full((data.shape[0]),label))
        self.TrainData, self.TrainLabels=self.concatenate(datas,labels)

    def getTrainItem(self,index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]

        if self.transform:
            img=self.transform(img)

        if self.target_transform:
            target=self.target_transform(target)

        return index,img,target

    def getTestItem(self,index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]

        if self.test_transform:
            img=self.test_transform(img)

        if self.target_test_transform:
            target=self.target_test_transform(target)

        return index, img, target

    def __getitem__(self, index):
        if list(self.TrainData)!=[]:
            return self.getTrainItem(index)
        elif list(self.TestData)!=[]:
            return self.getTestItem(index)


    def __len__(self):
        if list(self.TrainData)!=[]:
            return len(self.TrainData)
        elif list(self.TestData)!=[]:
            return len(self.TestData)

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]
img_size = 224
train_transform = transforms.Compose([transforms.RandomCrop((img_size, img_size), padding=4),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ColorJitter(brightness=0.24705882352941178),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
test_transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])



# train_dataset = iCIFAR100('data', test_transform=train_transform,  download=True)
# test_datasets = iCIFAR100('data', transform=test_transform, train=False, download=True)