import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

matplotlib.rc('font', family='Microsoft YaHei')
data = pd.read_csv('hd.tsv', delimiter='\t')
print(data['61'])

print(data.columns.values)

features = data.columns.values[1:-2]
print(features)
count = 0
selected_features = []
corrs = []
scaler = MinMaxScaler()
for i in range(len(features)):
    x = data[features[i]].values
    y = data['Target'].values
    corr, p = pearsonr(x, y)
    if abs(corr) >= 0.3:
        corrs.append(corr)
        print(corr, i)
        selected_features.append(features[i])
        count += 1
    else:
        print(corr)
print(len(selected_features))
for feature in data.columns.values[1:-2]:
    if feature not in selected_features:
        del data[feature]
print(data)
x_train = []
for i in range(len(selected_features)):
    x_train.append(data[selected_features[i]].values)

x_train = np.array(x_train).T
X = scaler.fit_transform(x_train)
print(X)
Y = np.array(data['Target'].values)

kf = KFold(n_splits=5,shuffle=True)
model_best = None
best_test_r2 = -np.inf
for train_index, test_index in kf.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    reg = LinearRegression(fit_intercept=False)
    reg.fit(x_train, y_train)
    prediction_test = reg.predict(x_test)
    prediction_train = reg.predict(x_train)
    print('train r2: ',r2_score(y_train, prediction_train))
    print('test r2: ',r2_score(y_test, prediction_test))
    if r2_score(y_test, prediction_test) > best_test_r2:
        model_best = reg
        best_test_r2 = r2_score(y_test, prediction_test)
    # print(prediction_test)
    # print(y_test)
    print('*' * 50)
    figure1 = plt.figure()
    plt.scatter(y_test, prediction_test)
    plt.xlabel('y_true')
    plt.ylabel('y_predict')
    plt.show()
print(best_test_r2)
quit()
x_train = X[:]
y_train = Y[:]
x_test = X[80:]
y_test = Y[80:]

print(x_train.shape)
print(y_train.shape)
reg = LinearRegression(fit_intercept=False)
reg.fit(x_train, y_train)
prediction = reg.predict(x_train)
residual = (y_train - prediction)
figure1 = plt.figure()
plt.scatter(y_train,prediction)
plt.xlabel('y_true')
plt.ylabel('y_predict')
plt.show()

print(r2_score(y_train, prediction))
print(y_train)
print(prediction)

importance = reg.coef_
print(reg.coef_)
print(reg.predict(x_test))
print(y_test)
df = pd.DataFrame()
df['selected features'] = selected_features
df['pearson'] = corrs
df['coefficient'] = reg.coef_
df['absolute coefficient'] = abs(reg.coef_)
df = df.sort_values(by=['absolute coefficient'],ascending=False)
print(df)
plt.bar(df['selected features'], df['coefficient'])
plt.xlabel('选中的feature')
plt.ylabel('重要性(回归因数)')
plt.show()
df.to_excel('hd回归模型.xlsx', index=False)
