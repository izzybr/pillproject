import os
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pickle
import pandas as pd
import numpy as np
import pretrainedmodels as pm
#from torchvision import io

model = pm.__dict__["resnet50"](pretrained='imagenet')
##This is never called
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 5)
        self.fc1 = nn.Linear(256, 50)
        self.dropout = nn.Dropout(0.10) #*a little dropout is probably a good idea
        self.fc_final = nn.Linear(50, 14)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc_final(x)
        return x
model.pth4419160136
test_path = '/home/ngs/pillproject/converted_pngs/converted_pngs'
#model_path = '/home/ngs/pillproject/PILL_COLOR_TRAIN/color_train_records/model.pth8136445306'
model_path = '/home/ngs/pillproject/PILL_COLOR_TRAIN/color_train_records/model.pth9166614547'
model_path = '/home/ngs/pillproject/PILL_COLOR_TRAIN/color_train_records/model.pth4419160136'
#####
# Almost 80% accurate between regular pills and multicolored pills
model_path = '/home/ngs/pillproject/PILL_COLOR_TRAIN/color_train_records/model.pth6862594411'
#######
model_path = '/home/izzy/projects/vision/color_20221121_135456_0.4407175558254695_/data.pkl' # color
shape_path = '/home/izzy/projects/vision/shape_20221122_202125_0.34041054538136445_'

file = open(model_path, 'rb')
model = pickle.load(file)

def load_model(model, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #ptimizer.load_state_dict[checkpoint['optimizer_state_dict']]
    return model

images = [file for file in os.listdir(test_path) if file.endswith('.PNG')]

valid_transform = T.Compose([
    #transforms.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

#model = pm.__dict__["resnet50"](pretrained='imagenet')

model.avg_pool = nn.AdaptiveAvgPool2d(1)

model.last_linear = nn.Sequential(
    nn.BatchNorm1d(2048),
    nn.Dropout(p=0.20),
    nn.Linear(in_features=2048, out_features=2048),
    nn.ReLU(),
    #nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1),
    #nn.Dropout(p=0.5),
    nn.Linear(in_features=2048, out_features=12),
)

ct = 0
for child in model.children():
    ct += 1
    if ct < 9:
        for param in child.parameters():
            param.requires_grad = False

train_dataset = datasets.ImageFolder(
    root=test_path,
    transform=valid_transform
)

#torch.cuda.set_device(1)

def load_model(model, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #ptimizer.load_state_dict[checkpoint['optimizer_state_dict']]
    return model

def checkone(im, model):
    im = im.unsqueeze(0)
    with torch.no_grad():
        model.eval()
        output = model(im)
    return output

# only here to create the dataset - which is turn is only done to get idx_to_class
train_path = '/home/ngs/pillproject/augmented/train/Color/'

dataset = datasets.ImageFolder(
    root=train_path,
    transform=valid_transform
)
x = dataset.class_to_idx

idx_to_class = {v: k for k, v in x.items()}

torch_native_out, smp, actual = [], [], []

for image in images[0:1000]:
    file = f'{test_path}/{image}'
    print(file)
    im = cv2.imread(file)
    im = valid_transform(im)
    output = checkone(im, model)
    ol = output.tolist()
    ol = ol[0]
    op = idx_to_class[ol.index(np.max(ol))]
    sm = output.softmax(dim = 1)
    sl = sm.tolist()
    sl = sl[0]
    sp = idx_to_class[sl.index(np.max(sl))]
    max = output.softmax(dim = 1)
    idx = df.loc[df['Name'] == image].index[0]
    truth = df.iloc[idx, 11]
    smp.append(sp)
    torch_native_out.append(op)
    actual.append(truth)
    
fd = pd.DataFrame(list(zip(torch_native_out, smp, actual)), columns=['torch_max', 'softmax_max', 'actual'])
np.count_nonzero(fd['softmax_max'] == fd['actual'])


partial_match = [x for x,y in list(zip(smp, actual)) if x in y]