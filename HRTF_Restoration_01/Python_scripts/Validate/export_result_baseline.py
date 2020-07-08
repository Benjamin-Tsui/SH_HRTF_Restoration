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

torch.set_printoptions(precision=5)


print(f"Time: {datetime.datetime.now()}")

def export_result(model_name, trained_model, file_loc, output_name):

    # import test data
    print('Import test data (x):')
    sh_hrtf_test = pd.read_csv(file_loc + 'SHed_hrtf_dB.txt', header=None)
    print(sh_hrtf_test.head(5))
    print(sh_hrtf_test.shape)
    print('Import test data (y):')
    hrtf_test = pd.read_csv(file_loc + 'hrtf_dB.txt', header=None)
    print(hrtf_test.head(5))
    print(hrtf_test.shape)

    print('x data:')
    x_test = sh_hrtf_test.values.astype('float')
    x_test = x_test.reshape(np.size(x_test,0), 1, 2, -1)
    print('shape of x_test:')
    print(x_test.shape)

    print('---------------')
    print('y data:')
    y_test = hrtf_test.values.astype('float')
    print('shape of y_hold:')
    print(y_test.shape)

    print('===============')

    # standardise data
    means = trained_model['data_mean_hrtf']
    stds = trained_model['data_std_hrtf']
    x_test = (x_test - means) / stds

    class Load_Dataset(Dataset):

        def __init__(self, input_hrtf, output_hrtf):
            self.input_hrtf = torch.tensor(input_hrtf, dtype=torch.float)
            self.output_hrtf = torch.tensor(output_hrtf, dtype=torch.float)

        def __len__(self):
            return len(self.output_hrtf)

        def __getitem__(self, index):
            return self.input_hrtf[index], self.output_hrtf[index]

    # load data
    testset = Load_Dataset(x_test, y_test)
    test_data_num = np.size(y_test,0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_data_num, shuffle=False)

    test_dataiter = iter(testloader)
    test_input, test_label = test_dataiter.next()


    left_net = trained_model['left_model']
    print(left_net)
    right_net = trained_model['right_model']
    print(right_net)

    left_net.load_state_dict(trained_model['left_model_state_dict'])
    right_net.load_state_dict(trained_model['right_model_state_dict'])
    criterion = nn.MSELoss()

    left_net.eval()
    right_net.eval()

    left_test_loss = 0.0
    right_test_loss = 0.0
    accuracy = 0.0
    test_losses = []

    for test_input, test_label in testloader:
        with torch.no_grad():
            left_net.eval()
            right_net.eval()

            left_output = left_net(test_input)
            left_loss = criterion(left_output, test_label[:, 0:256])
            left_test_loss += left_loss.item()

            right_output = right_net(test_input)
            right_loss = criterion(right_output, test_label[:, 256:512])
            right_test_loss += right_loss.item()

            # left_output = np.log10(left_output) * 20
            # right_output = np.log10(right_output) * 20

            np.savetxt(model_name + output_name + '.csv', np.concatenate((left_output.numpy(), right_output.numpy()), axis=1), delimiter=',')

        print(f"Left Test loss: {left_test_loss / len(testloader)}")
        print(f"Right Test loss: {left_test_loss / len(testloader)}")
        print(f"Left Error mean: {torch.mean(torch.abs(left_output - test_label[:, 0:256]))}")
        print(f"Right Error mean: {torch.mean(torch.abs(left_output - test_label[:, 256:512]))}")
        print(f"Left Error std: {torch.std(torch.abs(left_output - test_label[:, 0:256]))}")
        print(f"Right Error std: {torch.std(torch.abs(left_output - test_label[:, 256:512]))}")

        # plt.hist(output.numpy() - val_label.numpy())
        # plt.show()

        idx = 3
        plt.figure()
        plt.subplot(211)
        plt.plot(left_output.numpy()[idx, :], label='predict')
        plt.plot(test_label.numpy()[idx, 0:256], label='target')
        plt.legend(frameon=False)
        plt.title('hrtf left')
        plt.xscale("log")
        plt.ylim(-50, 10)
        plt.subplot(212)
        plt.plot(right_output.numpy()[idx, :], label='predict')
        plt.plot(test_label.numpy()[idx, 256:512], label='target')
        plt.legend(frameon=False)
        plt.title('hrtf right')
        plt.xscale("log")
        plt.ylim(-50, 10)
        plt.show()

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

model_name = 'training_HRTF_08++_15_sparse'
# trained_model = torch.load('Models/' + model_name + '.pt')
trained_model = torch.load('Models/' + model_name + '.pt', map_location=torch.device('cpu'))
left_net = trained_model['left_model']
right_net = trained_model['right_model']

print(trained_model.keys())

folder_loc = 'C:/Users/.../Downloads/HRTF_Restoration_01/Training_data/Time_aligned/'

output_name = '_bern_out'
file_loc = folder_loc + 'SH_HRTFs_1st_order_512_sparse_in_bern_oct_3/'
export_result(model_name, trained_model, file_loc, output_name)

output_name = '_sub18_out'
file_loc = folder_loc + 'SH_HRTFs_1st_order_512_sparse_in_sub_18_oct_3/'
export_result(model_name, trained_model, file_loc, output_name)

output_name = '_sub19_out'
file_loc = folder_loc + 'SH_HRTFs_1st_order_512_sparse_in_sub_19_oct_3/'
export_result(model_name, trained_model, file_loc, output_name)

output_name = '_sub20_out'
file_loc = folder_loc + 'SH_HRTFs_1st_order_512_sparse_in_sub_20_oct_3/'
export_result(model_name, trained_model, file_loc, output_name)
