import random

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from convert import convertTrainFile
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch import nn
import torch.nn.functional as f



class MLP(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(in_dim,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_dim),
        )


    def forward(self, x):
        out = self.module(x)

        return out

def findTrainFeatureMaxMin(filename):
    data = convertTrainFile(filename)

    features = data.columns.values[:-4]

    x_train = []
    for i in range(len(features)):
        x_train.append(data[features[i]].values)

    x_train = np.array(x_train).T

    return x_train.max(axis=0),x_train.min(axis=0)

if  __name__== '__main__':
    train = False
    matplotlib.rc('font', family='Microsoft YaHei')
    filename = 'files/20220531-213sample14feature.xlsx'
    data = convertTrainFile(filename)

    print(data.columns.values)

    features = data.columns.values[:-4]



    x_train = []
    for i in range(len(features)):
        x_train.append(data[features[i]].values)

    x_train = np.array(x_train).T

    # X = x_train

    max,min = findTrainFeatureMaxMin(filename)

    X_std = (x_train - min) / (max - min)
    X = X_std * (1 - 0) + 0

    Y1 = np.array(data['Target1'].values)
    Y2 = np.array(data['Target2'].values)
    Y3 = np.array(data['Target3'].values)
    Y4 = np.array(data['Target4'].values)
    Y = np.vstack((Y1,Y2,Y3,Y4)).T



    # k-fold test
    # kf = KFold(n_splits=126,shuffle=True)
    # model_best = None
    # best_test_r2 = -np.inf
    # y_test_list = []
    # prediction_test_list  = []
    # train_r2_list = []
    # for train_index, test_index in kf.split(X):
    #     # print("TRAIN:", train_index, "TEST:", test_index)
    #     x_train, x_test = X[train_index], X[test_index]
    #     y_train, y_test = Y[train_index], Y[test_index]
    #
    #     x_train = torch.Tensor(x_train).to(device)
    #     y_train = torch.Tensor(y_train).view(y_train.shape[0], 1).to(device)
    #     x_test = torch.Tensor(x_test).to(device)
    #     y_test = torch.Tensor(y_test).view(y_test.shape[0], 1).to(device)
    #
    #     in_dim = x_train.shape[1]
    #     out_dim = y_train.shape[1]
    #
    #     model = MLP(in_dim=in_dim, out_dim=out_dim).to(device)
    #     loss_fn = nn.MSELoss()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #
    #     epochs = 2000
    #     with tqdm(range(epochs), unit='epoch', total=epochs, desc='Epoch iteration') as epoch:
    #         for ep in epoch:
    #             model.train()
    #             batch_size = 125
    #             step_num = len(x_train) // batch_size
    #             with tqdm(range(step_num),
    #                       unit=' samples',
    #                       total=step_num,
    #                       leave=True,
    #                       desc='Sample Iteration') as tepoch:
    #                 for step in tepoch:
    #                     try:
    #                         x_batch = x_train[step * batch_size:(step * batch_size) + batch_size]
    #                         y_batch = y_train[step * batch_size:(step * batch_size) + batch_size]
    #                     except:
    #                         x_batch = x_train[step * batch_size:]
    #                         y_batch = y_train[step * batch_size:]
    #                     output = model(x_batch)
    #                     loss = loss_fn(output, y_batch)
    #                     optimizer.zero_grad()
    #                     loss.backward()
    #                     optimizer.step()
    #
    #     prediction_test = model(x_test)
    #     prediction_train = model(x_train)
    #
    #
    #     train_r2_list.append(r2_score(y_train.cpu().detach().numpy(),prediction_train.cpu().detach().numpy()))
    #     y_test_list.append(y_test.cpu().detach().numpy()[0])
    #     prediction_test_list.append(prediction_test.cpu().detach().numpy()[0])
    # print(y_test_list)
    # print(prediction_test_list)
    # print('test r2: ',r2_score(y_test_list, prediction_test_list))
    # print('train r2: ',train_r2_list)
    # # print(best_test_r2)
    # quit()

    seed = random.randint(1,1000)
    # seed = 618
    torch.cuda.manual_seed(seed)

    # target1
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=seed)
    x_train = torch.Tensor(x_train).to(device)
    y_train = torch.Tensor(y_train).to(device)

    x_dev, x_test, y_dev, y_test = train_test_split(x_test, y_test, test_size=0.7, random_state=seed)
    x_dev = torch.Tensor(x_dev).to(device)
    y_dev = torch.Tensor(y_dev).to(device)
    x_test = torch.Tensor(x_test).to(device)
    y_test = torch.Tensor(y_test).to(device)


    in_dim = x_train.shape[1]
    out_dim = y_train.shape[1]
    print(in_dim,out_dim)
    if train:
        model = MLP(in_dim=in_dim,out_dim=out_dim).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)
        best_r2 = -np.inf
        best_epoch = None
        epochs = 30000
        with tqdm(range(epochs),unit= 'epoch',total=epochs,desc='Epoch iteration') as epoch:
            for ep in epoch:
                model.train()
                batch_size = 149
                step_num = len(x_train) // batch_size
                with tqdm(range(step_num),
                          unit=' samples',
                          total=step_num,
                          leave=True,
                          desc='Sample Iteration') as tepoch:
                    for step in tepoch:
                        try:
                            x_batch = x_train[step * batch_size:(step * batch_size) + batch_size]
                            y_batch = y_train[step * batch_size:(step * batch_size) + batch_size]
                        except:
                            x_batch = x_train[step * batch_size:]
                            y_batch = y_train[step * batch_size:]
                        output = model(x_batch)
                        loss = loss_fn(output,y_batch)


                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                model.eval()
                y_hat = model(x_dev)
                test_r2 = r2_score(y_dev.cpu().detach().numpy(), y_hat.cpu().detach().numpy())
                if test_r2 > best_r2:
                    print(test_r2)
                    best_r2 = test_r2
                    best_epoch = ep+1
                    torch.save({'MLP':model.state_dict()},f'checkpoints/best_model.pt')

        print('best dev r2 : ', best_r2)
        print('best epoch',best_epoch)
        print('random seed: ', seed)

    model = MLP(in_dim=in_dim, out_dim=out_dim).to(device)
    checkpoint = torch.load(f'checkpoints/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['MLP'])


    model.eval()
    y_hat = model(x_test)
    print('best test r2: ', r2_score(y_test.cpu().detach().numpy(), y_hat.cpu().detach().numpy()))









