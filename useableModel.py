import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from neuralNetwork import MLP
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




if  __name__== '__main__':
    test_file = pd.read_csv('hd_126samples.csv')

    features = test_file.columns.values[:-2]
    X = []
    for i in range(len(features)):
        X.append(test_file[features[i]].values)


    X = np.array(X).T
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    Y = np.array(test_file['Target'].values)

    x = torch.Tensor(X).to(device)
    y = torch.Tensor(Y).view(Y.shape[0], 1).to(device)

    in_dim = x.shape[1]
    out_dim = 1

    model = MLP(in_dim=in_dim, out_dim=out_dim).to(device)
    checkpoint = torch.load(f'checkpoints/best_model.pt', map_location='cuda')
    model.load_state_dict(checkpoint['MLP'])

    y_hat = model(x)
    # print(r2_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()))
    print(y_hat)

    