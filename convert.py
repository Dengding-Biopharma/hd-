import numpy as np
import pandas as pd

def convertTrainFile(filename):
    data = pd.read_excel(filename)

    df = pd.DataFrame()

    target_name = ['target1','target2','target3','target4']
    target_index = [0,1,2,3]
    samples = []
    target_1_list = []
    target_2_list = []
    target_3_list = []
    target_4_list = []

    for label in data.columns.values[1:]:
        temp = data[label]
        target_1 = temp[data['Unnamed: 0'] == target_name[0]]
        target_2 = temp[data['Unnamed: 0'] == target_name[1]]
        target_3 = temp[data['Unnamed: 0'] == target_name[2]]
        target_4 = temp[data['Unnamed: 0'] == target_name[3]]
        target_1_list.append(float(target_1.values[0]))
        target_2_list.append(float(target_2.values[0]))
        target_3_list.append(float(target_3.values[0]))
        target_4_list.append(float(target_4.values[0]))
        sample = temp
        samples.append(sample.values)
    samples = np.array(samples)

    for i in range(len(data.index.values)):
        if i in target_index:
            continue
        column_label = data['Unnamed: 0'].values[i][7:]
        df[str(column_label)] = samples[:,i]

    df['Target1'] = target_1_list
    df['Target2'] = target_2_list
    df['Target3'] = target_3_list
    df['Target4'] = target_4_list

    return df

def convertPredictionFile(filename):
    data = pd.read_excel(filename)

    df = pd.DataFrame()

    samples = []

    for label in data.columns.values[1:]:
        samples.append(data[label].values)
    samples = np.array(samples)


    for i in range(len(data.index.values)):
        column_label = data['Unnamed: 0'].values[i][7:]
        df[str(column_label)] = samples[:, i]

    return df,data.columns.values[1:]

if  __name__== '__main__':
    convertPredictionFile('files/20220518-126sample31feature_test.xlsx')
