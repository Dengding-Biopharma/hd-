import random
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from convert import convertPredictionFile

from neuralNetwork import MLP,findTrainFeatureMaxMin

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print('这个模型可以预测target1-4')
filename = str(
    input('请给一个需要预测的xlsx文件路径，横坐标是15个feature(需要与20220518-126sample31feature给的顺序一样)，纵坐标是sample1,2,3...：\n'))
test_file,sample_name = convertPredictionFile(filename)

features = test_file.columns.values

in_dim = 15
out_dim = 4

model = MLP(in_dim=in_dim, out_dim=out_dim).to(device)
checkpoint = torch.load(f'checkpoints/best_model.pt', map_location=device)
model.load_state_dict(checkpoint['MLP'])


X = []
for i in range(100000):
    temp = []
    for i in range(len(features)):
        temp.append(random.choice(test_file[features[i]].values))
    X.append(temp)
X_before = np.array(X)
X = np.array(X)


max, min = findTrainFeatureMaxMin()
X_std = (X - min) / (max - min)
X = X_std * (1 - 0) + 0
x = torch.Tensor(X).to(device)

y_hat = model(x)


y_hat = y_hat.cpu().detach().numpy()

df = pd.DataFrame()
for i in range(len(features)):
    df[features[i]] = X_before[:,i]
df['target1'] = y_hat[:,0]
df['target2'] = y_hat[:,1]
df['target3'] = y_hat[:,2]
df['target4'] = y_hat[:,3]
df.to_excel('output.xlsx',index=False,na_rep=np.nan)
