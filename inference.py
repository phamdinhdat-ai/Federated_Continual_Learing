
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import time
import torch 
import torch.nn as nn
import torchvision
from torchvision import transforms

from torch.utils.data import TensorDataset, DataLoader
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
import tqdm

classes = {0: 'Down',
 1: 'DownLEFT',
 2: 'DownRIGHT',
 3: 'Left',
 4: 'LeftDOWN',
 5: 'LeftUP',
 6: 'Right',
 7: 'RightDOWN',
 8: 'RightUP',
 9: 'Up',
 10: 'UpLEFT',
 11: 'UpRIGHT'}


def slice_window(data, labels, sequence_length=20, overlap=0.1):
    step  = int(sequence_length - overlap * sequence_length)
    X_local = []
    y_local = []
    for start in range(0, data.shape[0] - sequence_length, step):
        end = start + sequence_length
        X_local.append(data[start:end])
        y_local.append(labels[end-1])
    return np.array(X_local), np.array(y_local)

x = []
y_p = []
y_t = []
def draw_graph(i, y, y_pred):
    x.append(i)
    y_p.append(y_pred)
    y_t.append(y)
    plt.cla()
    plt.plot(x, [y_t, y_p])
  
#load model 
model_buffer15 = torch.load("./workdir/lstm_model_buffer15.pt")
data_transform = transforms.Compose([ transforms.ToTensor()])

#load data

testset = np.load('./datasets/new_data_static/test_static.npy')
data = testset[:,1:4]
labels = testset[:,-1]

#process data
data_sq, label_sq = slice_window(data=data, labels=labels)
eval_dataset = dataset_transform(x=data_sq, y=label_sq, transform=transforms_match['breathing'])
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=True, num_workers=0,
                                       drop_last=True)#to tensor data 

# inference 
for i,(X, y) in enumerate(eval_loader):
    start = time.time()
    y_pred = model_buffer15(X)
    indices = torch.argmax(y_pred, dim=1)
    end = time.time()
    print("Time Excercute: ", end-start)
    print("================================")
    print(f"Grouth Truth:{classes[int(y)]} | Predicted: {classes[int(indices)]}")
    plt.scatter(i, int(y), color = "red", marker="+")
    plt.scatter(i, int(indices), color="green")
    plt.pause(0.5)
    plt.title("Real Time Prediction")
    plt.xlabel("interval")
    plt.ylabel("classes")
    plt.legend(["Grounth Truth", "Prediction"],bbox_to_anchor =(0.75, 1.15))



plt.show()
# generating random data values
# x = np.linspace(1, 1000, 5000)
# y = np.random.randint(1, 1000, 5000)
 
# # enable interactive mode
# plt.ion()
 
# # creating subplot and figure
# fig = plt.figure()
# ax = fig.add_subplot(111)
# line1, = ax.plot(x, y)
 
# # setting labels
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Updating plot...")
 
# # looping
# for _ in range(50):
   
#     # updating the value of x and y
#     line1.set_xdata(x*_)
#     line1.set_ydata(y)
 
#     # re-drawing the figure
#     fig.canvas.draw()
     
#     # to flush the GUI events
#     fig.canvas.flush_events()
#     time.sleep(0.1)