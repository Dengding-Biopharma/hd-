import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from convert import convertPredictionFile

from neuralNetwork import MLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print('这个模型可以预测target1和target2')
filename = str(
    input('请给一个需要预测的xlsx文件路径，横坐标是31个feature(需要与20220518-126sample31feature给的顺序一样)，纵坐标是sample1,2,3...：\n'))
test_file = convertPredictionFile(filename)

features = test_file.columns.values

X = []
for i in range(len(features)):
    X.append(test_file[features[i]].values)


X = np.array(X).T
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

x = torch.Tensor(X).to(device)

in_dim = x.shape[1]
out_dim = 1

model1 = MLP(in_dim=in_dim, out_dim=out_dim).to(device)
checkpoint = torch.load(f'checkpoints/best_model_target1.pt', map_location=device)
model1.load_state_dict(checkpoint['MLP'])

model2 = MLP(in_dim=in_dim, out_dim=out_dim).to(device)
checkpoint = torch.load(f'checkpoints/best_model_target2.pt', map_location=device)
model2.load_state_dict(checkpoint['MLP'])

y_hat1 = model1(x)
y_hat2 = model2(x)
# print(r2_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()))
y_hat1 = y_hat1.cpu().detach().numpy()
y_hat2 = y_hat2.cpu().detach().numpy()
print('target1    target2')
for i in range(y_hat1.shape[0]):
    print(y_hat1[i][0], y_hat2[i][0])
