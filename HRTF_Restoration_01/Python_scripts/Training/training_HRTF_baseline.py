import matplotlib.pyplot as plt
import pandas as pd
import datetime
import math
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from visdom import Visdom
viz = Visdom()
assert viz.check_connection()

from mpl_toolkits.mplot3d import Axes3D
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.animation import FuncAnimation

torch.set_printoptions(precision=5)

print(f"Time: {datetime.datetime.now()}")

data_len = 55000   # set the number of data that is going to be used in training
hold_size = 5000   # set the number of hold out data

file_loc = 'C:/Users/.../Downloads/HRTF_Restoration_01/Training_data/Time_aligned/'   # training data folder location

# import trining data
print('Import data (x):')
sh_hrtf = pd.read_csv(file_loc +
                      'SH_HRTFs_1st_order_512_sparse_in_oct_3_hold/SHed_hrtf_dB.txt', header=None)
sh_hrtf = sh_hrtf.append(pd.read_csv(file_loc +
                                     'SH_HRTFs_1st_order_512_sparse_in_oct_3_ari_ele_-30/SHed_hrtf_dB.txt', header=None),
                         ignore_index=True, sort=False)
sh_hrtf = sh_hrtf.append(pd.read_csv(file_loc +
                                     'SH_HRTFs_1st_order_512_sparse_in_riec_oct_3_ele_-30_50/SHed_hrtf_dB.txt', header=None),
                         ignore_index=True, sort=False)
sh_hrtf = sh_hrtf.append(pd.read_csv(file_loc +
                                     'SH_HRTFs_1st_order_512_sparse_in_ITA_oct_3/SHed_hrtf_dB.txt', header=None),
                         ignore_index=True, sort=False)
sh_hrtf = sh_hrtf.append(pd.read_csv(file_loc +
                                     'SH_HRTFs_1st_order_512_sparse_in_ITA_oct_4/SHed_hrtf_dB.txt', header=None),
                         ignore_index=True, sort=False)
sh_hrtf = sh_hrtf.append(pd.read_csv(file_loc +
                                     'SH_HRTFs_1st_order_512_sparse_in_oct_4_hold/SHed_hrtf_dB.txt', header=None),
                         ignore_index=True, sort=False)
sh_hrtf = sh_hrtf.append(pd.read_csv(file_loc +
                                     'SH_HRTFs_1st_order_512_sparse_in_oct_4_ari_ele_-30/SHed_hrtf_dB.txt', header=None),
                         ignore_index=True, sort=False)


print(sh_hrtf.head(5))
print(sh_hrtf.shape)

# shuffle samples and remove some samples (save memory)
np.random.seed(52)
shuffle_array = np.random.permutation(len(sh_hrtf))
print('Shuffle data (x1):')
sh_hrtf = sh_hrtf.iloc[shuffle_array[0:data_len]]
print(sh_hrtf.head(5))
print(sh_hrtf.shape)


print('Import data (y):')
hrtf = pd.read_csv(file_loc +
                   'SH_HRTFs_1st_order_512_sparse_in_oct_3_hold/hrtf_dB.txt', header=None)
hrtf = hrtf.append(pd.read_csv(file_loc +
                                  'SH_HRTFs_1st_order_512_sparse_in_oct_3_ari_ele_-30/hrtf_dB.txt', header=None),
                      ignore_index=True, sort=False)
hrtf = hrtf.append(pd.read_csv(file_loc +
                                  'SH_HRTFs_1st_order_512_sparse_in_riec_oct_3_ele_-30_50/hrtf_dB.txt', header=None),
                      ignore_index=True, sort=False)
hrtf = hrtf.append(pd.read_csv(file_loc +
                                  'SH_HRTFs_1st_order_512_sparse_in_ITA_oct_3/hrtf_dB.txt', header=None),
                      ignore_index=True, sort=False)
hrtf = hrtf.append(pd.read_csv(file_loc +
                                  'SH_HRTFs_1st_order_512_sparse_in_ITA_oct_4/hrtf_dB.txt', header=None),
                      ignore_index=True, sort=False)
hrtf = hrtf.append(pd.read_csv(file_loc +
                                  'SH_HRTFs_1st_order_512_sparse_in_oct_4_hold/hrtf_dB.txt', header=None),
                      ignore_index=True, sort=False)
hrtf = hrtf.append(pd.read_csv(file_loc +
                                  'SH_HRTFs_1st_order_512_sparse_in_oct_4_ari_ele_-30/hrtf_dB.txt', header=None),
                      ignore_index=True, sort=False)


print(hrtf.head(5))
print(hrtf.shape)

print('Shuffle data (y):')
hrtf = hrtf.iloc[shuffle_array[0:data_len]]
print(hrtf.head(5))
print(hrtf.shape)

# slicing data
print('Slicing data:')
train_percent = 0.8
# remain_size = hrir.shape[0] - hold_size
remain_size = hrtf.shape[0] - hold_size
print('hold data size = ' + str(hold_size))
print('remain data size = ' + str(remain_size))

train_size = int(train_percent * remain_size)
test_size = remain_size - train_size
print('train data size = ' + str(train_size))
print('test data size = ' + str(test_size))

print('---------------')
print('x1 data:')
# x1_hold = torch.tensor(hrir.iloc[0:hold_size].values, dtype=torch.float)
# x1_remain = torch.tensor(hrir.iloc[hold_size:].values, dtype=torch.float)
x1_hold = sh_hrtf.iloc[0:hold_size].values
x1_remain = sh_hrtf.iloc[hold_size:].values
x1_hold = x1_hold.reshape(np.size(x1_hold,0), 1, 2, -1)
x1_remain = x1_remain.reshape(np.size(x1_remain,0), 1, 2, -1)
x1_train = x1_remain[0:train_size].astype('float')
x1_val = x1_remain[train_size:].astype('float')
print('shape of x1_hold, x1_train and x1_val:')
print(x1_hold.shape)
print(x1_train.shape)
print(x1_val.shape)


print('---------------')
print('y data:')
# y_hold = torch.tensor(azi.iloc[0:hold_size].values, dtype=torch.float)
# y_remain = torch.tensor(azi.iloc[hold_size:].values, dtype=torch.float)
y_hold = hrtf.iloc[0:hold_size].values
y_remain = hrtf.iloc[hold_size:].values
y_train = y_remain[0:train_size].astype('float')
y_val = y_remain[train_size:].astype('float')
print('shape of y_hold, y_train and y_val:')
print(y_hold.shape)
print(y_train.shape)
print(y_val.shape)

print('===============')

# standardise data
means_1 = x1_train.mean(axis=0)
stds_1 = x1_train.std(axis=0)
x1_train = (x1_train - means_1) / stds_1
x1_val = (x1_val - means_1) / stds_1

# normalise data
x_train_min = x1_train.min()
x_train_max = x1_train.max()
# x1_train = (x1_train - x_train_min) / (x_train_max - x_train_min)
# x1_test = (x1_test - x_train_min) / (x_train_max - x_train_min)


# import test data
print('Import test data (x):')
sh_hrtf_test = pd.read_csv(file_loc + 'SH_HRTFs_1st_order_512_sparse_in_sub_20_oct_3/SHed_hrtf_dB.txt')
print(sh_hrtf_test.head(5))
print(sh_hrtf_test.shape)
print('Import test data (y):')
hrtf_test = pd.read_csv(file_loc + 'SH_HRTFs_1st_order_512_sparse_in_sub_20_oct_3/hrtf_dB.txt')
print(hrtf_test.head(5))
print(hrtf_test.shape)

print('x data:')
x1_test = sh_hrtf_test.values.astype('float')
x1_test = x1_test.reshape(np.size(x1_test,0), 1, 2, -1)
print('shape of x1_hold:')
print(x1_test.shape)


print('---------------')
print('y data (azimuth only):')
y_test = hrtf_test.values.astype('float')
print('shape of y_hold:')
print(y_test.shape)

print('===============')

x1_test = (x1_test - means_1) / stds_1

print('Import test data (x):')
sh_hrtf_test_bern = pd.read_csv(file_loc + 'SH_HRTFs_1st_order_512_sparse_in_bern_oct_3/SHed_hrtf_dB.txt')
print(sh_hrtf_test_bern.head(5))
print(sh_hrtf_test_bern.shape)
print('Import test data (y):')
hrtf_test_bern = pd.read_csv(file_loc + 'SH_HRTFs_1st_order_512_sparse_in_bern_oct_3/hrtf_dB.txt')
print(hrtf_test_bern.head(5))
print(hrtf_test_bern.shape)

print('x data:')
x1_test_bern = sh_hrtf_test_bern.values.astype('float')
x1_test_bern = x1_test_bern.reshape(np.size(x1_test_bern,0), 1, 2, -1)
print('shape of x1_hold:')
print(x1_test_bern.shape)

print('---------------')
print('y data (azimuth only):')
y_test_bern = hrtf_test_bern.values.astype('float')
print('shape of y_hold:')
print(y_test_bern.shape)

print('===============')

x1_test_bern = (x1_test_bern - means_1) / stds_1

# function to load data
class Load_Dataset(Dataset):

    def __init__(self, input_hrtf, output_hrtf):
        self.input_hrtf = torch.tensor(input_hrtf, dtype=torch.float)
        self.output_hrtf = torch.tensor(output_hrtf, dtype=torch.float)

    def __len__(self):
        return len(self.output_hrtf)

    def __getitem__(self, index):
        return self.input_hrtf[index], self.output_hrtf[index]


# load data
trainset = Load_Dataset(x1_train, y_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)

valset = Load_Dataset(x1_val, y_val)
valloader = torch.utils.data.DataLoader(valset, batch_size=8, shuffle=True)

testset = Load_Dataset(x1_test, y_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True)

testset_bern = Load_Dataset(x1_test_bern, y_test_bern)
testloader_bern = torch.utils.data.DataLoader(testset_bern, batch_size=8, shuffle=True)

dataiter = iter(trainloader)
train_input, train_label = dataiter.next()

val_dataiter = iter(valloader)
val_input, val_label = dataiter.next()

test_dataiter = iter(testloader)
test_input, test_label = dataiter.next()

test_dataiter_bern = iter(testloader)
test_bern_input, test_bern_label = dataiter.next()

# setup model
conv1_node = 8
conv2_node = 16
conv3_node = 32
conv4_node = 64
conv5_node = 128

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv1_node, kernel_size=(1, 5))
        self.conv1_bn = nn.BatchNorm2d(conv1_node)
        self.conv2 = nn.Conv2d(in_channels=conv1_node, out_channels=conv2_node, kernel_size=(1, 5))
        self.conv2_bn = nn.BatchNorm2d(conv2_node)
        self.conv3 = nn.Conv2d(in_channels=conv2_node, out_channels=conv3_node, kernel_size=(1, 5))
        self.conv3_bn = nn.BatchNorm2d(conv3_node)
        self.conv4 = nn.Conv2d(in_channels=conv3_node, out_channels=conv4_node, kernel_size=(1, 5))
        self.conv4_bn = nn.BatchNorm2d(conv4_node)
        self.conv5 = nn.Conv2d(in_channels=conv4_node, out_channels=conv5_node, kernel_size=(2, 5))
        self.conv5_bn = nn.BatchNorm2d(conv5_node)

        self.t_conv1 = nn.ConvTranspose2d(in_channels=conv5_node, out_channels=conv4_node, kernel_size=(1, 5))
        self.conv8_bn = nn.BatchNorm2d(conv4_node)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=conv4_node, out_channels=conv3_node, kernel_size=(1, 5))
        self.conv9_bn = nn.BatchNorm2d(conv3_node)
        self.t_conv3 = nn.ConvTranspose2d(in_channels=conv3_node, out_channels=conv2_node, kernel_size=(1, 5))
        self.conv10_bn = nn.BatchNorm2d(conv2_node)
        self.t_conv4 = nn.ConvTranspose2d(in_channels=conv2_node, out_channels=conv1_node, kernel_size=(1, 5))
        self.conv11_bn = nn.BatchNorm2d(conv1_node)
        self.t_conv5 = nn.ConvTranspose2d(in_channels=conv1_node, out_channels=1, kernel_size=(1, 5))

        self.hrtf1 = nn.Linear(512, 512)
        self.hrtf2 = nn.Linear(512, 1024)
        self.hrtf3 = nn.Linear(1024, 512)
        self.hrtf4 = nn.Linear(512, 256)
        self.hrtf5 = nn.Linear(256, 256)

        self.merge = nn.Linear(512, 256)

        # Dropout module with 0.2 drop probability
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x1 = F.relu(self.conv1_bn(self.conv1(x)))
        x1 = F.relu(self.conv2_bn(self.conv2(x1)))
        x1 = F.relu(self.conv3_bn(self.conv3(x1)))
        x1 = F.relu(self.conv4_bn(self.conv4(x1)))
        x1 = F.relu(self.conv5_bn(self.conv5(x1)))
        x1 = F.relu(self.t_conv1(x1))
        x1 = F.relu(self.t_conv2(x1))
        x1 = F.relu(self.t_conv3(x1))
        x1 = F.relu(self.t_conv4(x1))
        x1 = self.t_conv5(x1)
        x1 = torch.reshape(x1, (-1, 256))

        x2 = torch.reshape(x, (-1, 512))
        x2 = F.leaky_relu_(self.hrtf1(x2), 0.01)
        x2 = F.leaky_relu_(self.hrtf2(x2), 0.01)
        x2 = F.leaky_relu_(self.hrtf3(x2), 0.01)
        x2 = F.leaky_relu_(self.hrtf4(x2), 0.01)
        x2 = F.leaky_relu_(self.hrtf5(x2), 0.01)

        ## NN with dropout
        # x2 = torch.reshape(x, (-1, 512))
        # x2 = F.leaky_relu_(self.hrtf1(x2), 0.01)
        # x2 = F.leaky_relu_(self.dropout(self.hrtf2(x2)), 0.01)
        # x2 = F.leaky_relu_(self.dropout(self.hrtf3(x2)), 0.01)
        # x2 = F.leaky_relu_(self.dropout(self.hrtf4(x2)), 0.01)
        # x2 = F.leaky_relu_(self.hrtf5(x2), 0.01)

        x3 = torch.cat((x1, x2), dim=1)
        x = self.merge(x3)
        return x
left_net = Net()
right_net = Net()
print(left_net)
print(right_net)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
left_net.to(device)
right_net.to(device)

criterion = nn.SmoothL1Loss()
mse_loss = nn.MSELoss()
left_optimizer = optim.Adam(left_net.parameters(), lr=0.000001, betas=(0.9, 0.999))
right_optimizer = optim.Adam(right_net.parameters(), lr=0.000001, betas=(0.9, 0.999))
# # weight decay
# left_optimizer = optim.Adam(left_net.parameters(), lr=0.000001, betas=(0.9, 0.999), weight_decay=0.001)
# right_optimizer = optim.Adam(right_net.parameters(), lr=0.000001, betas=(0.9, 0.999), weight_decay=0.001)

epochs = 500 # set number of epoch

left_train_losses, left_val_losses, left_test_losses, left_test_bern_losses = [], [], [], []
right_train_losses, right_val_losses, right_test_losses, right_test_bern_losses = [], [], [], []

# training
for e in range(epochs):

    left_running_loss = 0.0
    right_running_loss = 0.0
    left_running_mse = 0.0
    right_running_mse = 0.0
    for train_input, train_label in trainloader:

        train_input, train_label = train_input.to(device), train_label.to(device)

        left_optimizer.zero_grad()
        left_output = left_net(train_input)
        # left_loss = criterion(left_output, train_label[:,0:256])
        left_loss = criterion(left_output, train_label[:, 0:256])
        left_loss.backward()
        left_optimizer.step()

        right_optimizer.zero_grad()
        right_output = right_net(train_input)
        # right_loss = criterion(right_output, train_label[:,256:512])
        right_loss = criterion(right_output, train_label[:, 256:512])
        right_loss.backward()
        right_optimizer.step()

        left_running_loss += left_loss.item()
        right_running_loss += right_loss.item()

        left_running_mse += mse_loss(left_output, train_label[:, 0:256])
        right_running_mse += mse_loss(right_output, train_label[:, 256:512])

    else: # test
        print(f"Epoch: {e}, " f"Time: {datetime.datetime.now()}")
        print(f"Left training loss: {left_running_loss / len(trainloader)}")
        print(f"Right training loss: {right_running_loss / len(trainloader)}")

        left_val_loss = 0.0
        right_val_loss = 0.0
        left_test_loss = 0.0
        right_test_loss = 0.0
        left_test_bern_loss = 0.0
        right_test_bern_loss = 0.0

        left_val_mse = 0.0
        right_val_mse = 0.0
        left_test_mse = 0.0
        right_test_mse = 0.0
        left_test_bern_mse = 0.0
        right_test_bern_mse = 0.0

        with torch.no_grad():
            left_net.eval()
            right_net.eval()
            for val_input, val_label in valloader:
                val_input, val_label = val_input.to(device), val_label.to(device)
                left_output = left_net(val_input)
                right_output = right_net(val_input)
                left_val_loss += criterion(left_output, val_label[:, 0:256])
                right_val_loss += criterion(right_output, val_label[:, 256:512])
                left_val_mse += mse_loss(left_output, val_label[:, 0:256])
                right_val_mse += mse_loss(right_output, val_label[:, 256:512])

            print(f"Left validate loss: {left_val_loss / len(valloader)}")
            print(f"Right validate loss: {right_val_loss / len(valloader)}")

            org_input = (val_input.cpu().numpy() * stds_1) + means_1
            viz.line(Y=np.stack([left_output.cpu()[0, :], val_label.cpu()[0, 0:256], org_input[0, 0, 0, 0:256]]).T,
                     opts=dict(markers=False, legend=['predict', 'target', 'input'], xtype='log',
                               title='hrtf left (val)'),
                     env='main', win='hrtf_left_val')
            viz.line(Y=np.stack([right_output.cpu()[0, :], val_label.cpu()[0, 256:512], org_input[0, 0, 1, 0:256]]).T,
                     opts=dict(markers=False, legend=['predict', 'target', 'input'], xtype='log',
                               title='hrtf right (val)'),
                     env='main', win='hrtf_right_val')

            for test_input, test_label in testloader:
                test_input, test_label = test_input.to(device), test_label.to(device)
                left_output = left_net(test_input)
                right_output = right_net(test_input)
                left_test_loss += criterion(left_output, test_label[:, 0:256])
                right_test_loss += criterion(right_output, test_label[:, 256:512])
                left_test_mse += mse_loss(left_output, test_label[:, 0:256])
                right_test_mse += mse_loss(right_output, test_label[:, 256:512])

            print(f"Left test loss (Test): {left_test_loss / len(testloader)}")
            print(f"Right test loss (Test): {right_test_loss / len(testloader)}")

            org_input = (test_input.cpu().numpy() * stds_1) + means_1
            viz.line(Y=np.stack([left_output.cpu()[0, :], test_label.cpu()[0, 0:256], org_input[0, 0, 0, 0:256]]).T,
                     opts=dict(markers=False, legend=['predict', 'target', 'input'], xtype='log',
                               title='hrtf left (test)'),
                     env='main', win='hrtf_left_test')
            viz.line(Y=np.stack([right_output.cpu()[0, :], test_label.cpu()[0, 256:512], org_input[0, 0, 1, 0:256]]).T,
                     opts=dict(markers=False, legend=['predict', 'target', 'input'], xtype='log',
                               title='hrtf right (test)'),
                     env='main', win='hrtf_right_test')

            for test_bern_input, test_bern_label in testloader_bern:
                test_bern_input, test_bern_label = test_bern_input.to(device), test_bern_label.to(device)
                left_bern_output = left_net(test_bern_input)
                right_bern_output = right_net(test_bern_input)
                left_test_bern_loss += criterion(left_bern_output, test_bern_label[:, 0:256])
                right_test_bern_loss += criterion(right_bern_output, test_bern_label[:, 256:512])
                left_test_bern_mse += mse_loss(left_bern_output, test_bern_label[:, 0:256])
                right_test_bern_mse += mse_loss(right_bern_output, test_bern_label[:, 256:512])

            print(f"Left test loss (bern): {left_test_bern_loss / len(testloader_bern)}")
            print(f"Right test loss (bern): {right_test_bern_loss / len(testloader_bern)}")

            org_input = (test_bern_input.cpu().numpy() * stds_1) + means_1
            viz.line(Y=np.stack([left_bern_output.cpu()[0, :], test_bern_label.cpu()[0, 0:256], org_input[0, 0, 0, 0:256]]).T,
                     opts=dict(markers=False, legend=['predict', 'target', 'input'], xtype='log',
                               title='hrtf left (Bernschutz)'),
                     env='main', win='hrtf_left_bern')
            viz.line(Y=np.stack([right_bern_output.cpu()[0, :], test_bern_label.cpu()[0, 256:512], org_input[0, 0, 1, 0:256]]).T,
                     opts=dict(markers=False, legend=['predict', 'target', 'input'], xtype='log',
                               title='hrtf right (Bernschutz)'),
                     env='main', win='hrtf_right_bern')

        print('---------------------------------------------------')
        print(f"Left training mse: {left_running_mse / len(trainloader)}")
        print(f"Right training mse: {right_running_mse / len(trainloader)}")
        print(f"Left validate mse: {left_val_mse / len(valloader)}")
        print(f"Right validate mse: {right_val_mse / len(valloader)}")
        print(f"Left test mse (Test): {left_test_mse / len(testloader)}")
        print(f"Right test mse (Test): {right_test_mse / len(testloader)}")
        print(f"Left test MSE (bern): {left_test_bern_mse / len(testloader_bern)}")
        print(f"Right test MSE (bern): {right_test_bern_mse / len(testloader_bern)}")

        print(f"==================================================")

        left_train_losses.append(left_running_loss / len(trainloader))
        left_val_losses.append(left_val_loss / len(valloader))
        left_test_losses.append(left_test_loss / len(testloader))
        left_test_bern_losses.append(left_test_bern_loss / len(testloader_bern))
        right_train_losses.append(right_running_loss / len(trainloader))
        right_val_losses.append(right_val_loss / len(valloader))
        right_test_losses.append(right_test_loss / len(testloader))
        right_test_bern_losses.append(right_test_bern_loss / len(testloader_bern))

        # left_train_losses_mse.append(left_running_mse / len(trainloader))
        # left_val_losses_mse.append(left_val_mse / len(valloader))
        # left_test_losses_mse.append(left_test_mse / len(testloader))
        # left_test_bern_losses_mse.append(left_test_bern_mse / len(testloader_bern))
        # right_train_losses_mse.append(right_running_mse / len(trainloader))
        # right_val_losses_mse.append(right_val_mse / len(valloader))
        # right_test_losses_mse.append(right_test_mse / len(testloader))
        # right_test_bern_losses_mse.append(right_test_bern_mse / len(testloader_bern))

        viz.line(X=np.column_stack((e, e, e, e)), Y=np.column_stack((left_running_loss / len(trainloader),
                                                               left_val_loss.cpu() / len(valloader),
                                                               left_test_loss.cpu() / len(testloader),
                                                               left_test_bern_loss.cpu() / len(testloader_bern))),
                 opts=dict(markers=False, legend=['train', 'val', 'test(SADIE 2)', 'test(Bernschutz)'], title='left loss'),
                 env='main', win='left_loss', update='append')
        viz.line(X=np.column_stack((e, e, e, e)), Y=np.column_stack((right_running_loss / len(trainloader),
                                                               right_val_loss.cpu() / len(valloader),
                                                               right_test_loss.cpu() / len(testloader),
                                                               right_test_bern_loss.cpu() / len(testloader_bern))),
                 opts=dict(markers=False, legend=['train', 'val', 'test(SADIE 2)', 'test(Bernschutz)'], title='right loss'),
                 env='main', win='right_loss', update='append')

        viz.line(X=np.column_stack((e, e, e, e)), Y=np.column_stack((left_running_mse.cpu().detach().numpy() / len(trainloader),
                                                               left_val_mse.cpu() / len(valloader),
                                                               left_test_mse.cpu() / len(testloader),
                                                               left_test_bern_mse.cpu() / len(testloader_bern))),
                 opts=dict(markers=False, legend=['train', 'val', 'test(SADIE 2)', 'test(Bernschutz)'], title='left loss (mse)'),
                 env='main', win='left_mse', update='append')
        viz.line(X=np.column_stack((e, e, e, e)), Y=np.column_stack((right_running_mse.cpu().detach().numpy() / len(trainloader),
                                                               right_val_mse.cpu() / len(valloader),
                                                               right_test_mse.cpu() / len(testloader),
                                                               right_test_bern_mse.cpu() / len(testloader_bern))),
                 opts=dict(markers=False, legend=['train', 'val', 'test(SADIE 2)', 'test(Bernschutz)'], title='right loss (mse)'),
                 env='main', win='right_mse', update='append')

        left_net.train()
        right_net.train()

    # save model
    # torch.save(net, 'HRTF_01.00.pt')
    torch.save({
        'data_mean_hrtf': means_1,
        'data_std_hrtf': stds_1,
        'data_max': x_train_max,
        'data_min': x_train_min,
        'Model': Net,
        'left_model': left_net,
        'right_model': right_net,
        'epoch': e,
        'left_model_state_dict': left_net.state_dict(),
        'right_model_state_dict': right_net.state_dict(),
        'left optimizer': left_optimizer,
        'right optimizer': right_optimizer,
        'left_optimizer_state_dict': left_optimizer.state_dict(),
        'right_optimizer_state_dict': right_optimizer.state_dict(),
        'loss_function': criterion,
        'left_loss': left_loss,
        'right_loss': right_loss,
        'left_train_losses': left_train_losses,
        'right_train_losses': right_train_losses,
        'left_val_losses': left_val_losses,
        'right_val_losses': right_val_losses,
        'left_test_losses': left_test_losses,
        'right_test_losses': right_test_losses,
        'left_test_tom_losses': left_test_bern_losses,  #should be Bernschutz
        'right_test_tom_losses': right_test_bern_losses, #should be Bernschutz
        'left_train_losses_mse': left_running_mse,
        'right_train_losses_mse': right_running_mse,
        'left_val_losses_mse': left_val_mse,
        'right_val_losses_mse': right_val_mse,
        'left_test_losses_mse': left_test_mse,
        'right_test_losses_mse': right_test_mse,
        'left_test_tom_losses_mse': left_test_bern_mse, #should be Bernschutz
        'right_test_tom_losses_mse': right_test_bern_mse, #should be Bernschutz
    }, 'training_HRTF_test_sparse.pt')

##