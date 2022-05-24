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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch import nn
from torch import optim
from torchvision import datasets, transforms


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(in_dim,64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim)
        )


    def forward(self, x):
        out = self.module(x)
        return out
if  __name__== '__main__':
    matplotlib.rc('font', family='Microsoft YaHei')
    data = pd.read_csv('hd_126samples.csv')

    print(data.columns.values)

    features = data.columns.values[:-2]
    print(features)
    count = 0
    selected_features = []
    corrs = []
    scaler = MinMaxScaler()
    for i in range(len(features)):
        x = data[features[i]].values
        y = data['Target'].values
        corr, p = pearsonr(x, y)
        if abs(corr) >= 0:
            corrs.append(corr)
            selected_features.append(features[i])
            count += 1
        else:
            print(corr)
    print(len(selected_features))


    # for feature in data.columns.values[:-2]:
    #     if feature not in selected_features:
    #         del data[feature]
    print(data)
    x_train = []
    for i in range(len(selected_features)):
        x_train.append(data[selected_features[i]].values)

    x_train = np.array(x_train).T
    X = scaler.fit_transform(x_train)
    Y1 = np.array(data['Target_Volume'].values)
    Y2 = np.array(data['Target'].values)
    Y = np.vstack((Y1,Y2)).T


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

    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=8)

    # x_train = X[:110]
    # y_train = Y[:110]
    # x_test = X[110:]
    # y_test = Y[110:]

    x_train = torch.Tensor(x_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    x_test = torch.Tensor(x_test).to(device)
    y_test = torch.Tensor(y_test).to(device)

    in_dim = x_train.shape[1]
    out_dim = y_train.shape[1]

    model = MLP(in_dim=in_dim,out_dim=out_dim).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)
    best_r2 = -np.inf
    epochs = 10000
    with tqdm(range(epochs),unit= 'epoch',total=epochs,desc='Epoch iteration') as epoch:
        for ep in epoch:
            model.train()
            batch_size = 100
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
            y_hat = model(x_test)
            test_r2 = r2_score(y_test.cpu().detach().numpy(), y_hat.cpu().detach().numpy())
            if test_r2 > best_r2:
                best_r2 = test_r2
                torch.save({'MLP':model.state_dict()},f'checkpoints/best_model.pt')
            # y_hat = model(x_train)
            # print('train r2: ',r2_score(y_train.cpu().detach().numpy(), y_hat.cpu().detach().numpy()))

    print('best test r2: ',best_r2)





