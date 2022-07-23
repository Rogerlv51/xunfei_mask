
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
import  torch.nn as nn
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

def make_label():   # 创建一个对应图片label的dataframe并把对应的图片地址写进去
    originpath = os.getcwd()    # 返回根地址
    traindir = os.listdir("./")[7]
    label0 = "mask_weared_incorrect"
    label1 = "with_mask"
    label2 = "without_mask"
    path0 = originpath + "\\" + traindir + "\\" + label0
    path1 = originpath + "\\" + traindir + "\\" + label1
    path2 = originpath + "\\" + traindir + "\\" + label2
    incorrect_namelist = os.listdir(path0)
    mask_namelist = os.listdir(path1)
    without_namelist = os.listdir(path2)
    data1 = {"image": incorrect_namelist, "label": label0}
    data1 = pd.DataFrame(data=data1)
    data2 = {"image": mask_namelist, "label": label1}
    data2 = pd.DataFrame(data=data2)
    data3 = {"image": without_namelist, "label": label2}
    data3 = pd.DataFrame(data=data3)
    data = pd.concat([data1, data2, data3], axis=0)     # 上下合并
    data["img_path"] = originpath + "\\" + traindir + "\\" + data["label"] + "\\" + data["image"]
    return data

# 对3个种类进行编码，不然算不了loss嘛，还是得转成数字类型
label_to_number = {"mask_weared_incorrect" : 0,
                   "with_mask" : 1,
                   "without_mask" : 2,}

# 定义基本参数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
torch.manual_seed(1024)
epoch = 25
batchsize = 64
lr = 1e-4
weight_decay = 1e-3
valid_size = 0.05

# 这段transforms的操作是参考李沐老师的猫狗分类比赛教程
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomChoice([
        transforms.Pad(padding=10),
        transforms.CenterCrop(480),
        transforms.RandomRotation(20),
        transforms.CenterCrop((576,432)),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1, 
            saturation=0.1,
            hue=0.1
        )
    ]),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
data = make_label()
df = shuffle(data, random_state=1024)
# 划分训练集和验证集
train_df, valid_df = train_test_split(df, test_size=valid_size)
print(f'train len: {len(train_df)}, test len: {len(valid_df)}')

class MyDataset(Dataset):
    def __init__(self, data, transforms, labelToNum):
        super().__init__()
        self.df = data
        self.transform = transforms
        self.label_num = labelToNum
        self.df = self.df.values.tolist()
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df[index]
        image = Image.open(row[2])
        image = self.transform(image)
        label = self.label_num[row[1]]
        return image, label

# 装载数据集
train_dataset = MyDataset(train_df, train_transform, label_to_number)
valid_dataset = MyDataset(valid_df, test_transform, label_to_number)
train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batchsize, shuffle=True)

# 用个resnet50预训练，试下效果
model = models.resnet50(pretrained=True)

model.fc = nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(model.fc.in_features, 3)
)
nn.init.xavier_uniform_(model.fc[1].weight)     # 随机初始化最后一层权重（如果不是微调而是从头训练resnet的话，把这行注释掉）
model = model.to(device)
print(model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, 
                            momentum=0.9, weight_decay=weight_decay)

def train(model, criterion, optimizer, train_dataloader, test_dataloader):

    total_train_loss = 0
    total_test_loss = 0
    
    model.train()
    with tqdm(train_dataloader, unit='batch', leave=False) as pbar:
        pbar.set_description(f'training')
        for images, idxs in pbar:
            images = images.to(device, non_blocking=True)
            idxs = idxs.to(device, non_blocking=True)
            output = model(images)

            loss = criterion(output, idxs)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    model.eval()
    with tqdm(test_dataloader, unit='batch', leave=False) as pbar:
        pbar.set_description(f'testing')
        for images, idxs in pbar:
            images = images.to(device, non_blocking=True)
            idxs = idxs.to(device, non_blocking=True)

            output = model(images)
            loss = criterion(output, idxs)
            total_test_loss += loss.item()

    train_acc = total_train_loss / len(train_dataset)
    test_acc = total_test_loss / len(valid_dataset)
    print(f'Train loss: {train_acc:.4f} Test loss: {test_acc:.4f} ')


for i in range(epoch):
    print(f"Epoch {i+1}/{epoch}")
    train(model, criterion, optimizer, train_dataloader, valid_dataloader)   #从头训练

model.eval()
image_ids, labels = [], []
filenames = os.listdir(os.getcwd()+"\\"+os.listdir("./")[6])
for filename in filenames:
    image = Image.open(os.getcwd()+"\\"+os.listdir("./")[6]+filename)
    image = test_transform(image)
    image = image.unsqueeze(0).to(device)
    image_ids.append(filename)
    labels.append(label_to_number[model(image).argmax().item()])

submission = pd.DataFrame({
    'image_id': image_ids,
    'label': labels,
})
submission.to_csv("./output.csv", index=False, header=True)