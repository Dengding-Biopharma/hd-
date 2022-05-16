import numpy as np
import pandas as pd

data = pd.read_excel('20220516-41sample32feature.xlsx')
print(data)

df = pd.DataFrame()

target_name = ['target1','target2']
target_index = [0,1]
samples = []
target_volumn_list = []
target_list = []

for label in data.columns.values[1:]:
    temp = data[label]
    target_volumn = temp[data['Unnamed: 0'] == target_name[0]]
    target = temp[data['Unnamed: 0'] == target_name[1]]
    target_list.append(float(target.values[0]))
    target_volumn_list.append(float(target_volumn.values[0]))
    sample = temp
    samples.append(sample.values)
samples = np.array(samples)

for i in range(len(data.index.values)):
    if i in target_index:
        continue
    column_label = data['Unnamed: 0'].values[i][7:]
    df[str(column_label)] = samples[:,i]

df['Target'] = target_list
df['Target_Volume'] = target_volumn_list
print(df)

df.to_csv('hd_41samples.csv',index=False)