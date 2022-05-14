import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

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
data = data.drop('Featue',axis=1)
data = data.drop(['Target_Volume','Target'],axis=1)
data.corr().to_excel('correlation Matrix.xlsx',index=False)
data = pd.read_excel('correlation Matrix.xlsx')
data.insert(0, 'selected Features', selected_features)
data.to_excel('correlation Matrix.xlsx',index=False)
