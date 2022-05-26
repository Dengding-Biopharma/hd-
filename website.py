import os

from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler

from werkzeug.utils import secure_filename
from convert import convertPredictionFile
from neuralNetwork import MLP,findTrainFeatureMaxMin
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

app.config['ALLOWED_FILE_EXTENSIONS'] = ['XLSX']


def allowed_file(filename):

    return '.' in filename and filename.rsplit('.', 1)[-1].upper() in app.config['ALLOWED_FILE_EXTENSIONS']


@app.route('/', methods=['GET', 'POST'])
def file_uploader(output=''):
    # error = None

    if request.method == 'GET':
        return render_template('upload.html', error=output)

    if request.method == 'POST':

        # f = request.files['file']
        f = request.files.getlist("file[]")[0]
        if f.filename == '':
            output = 'No filename.'
            return render_template('upload.html', error=output)

        if allowed_file(f.filename):
            # sort files
            filepath = 'tempFiles/'+secure_filename(f.filename)
            f.save(filepath)

            output += '<div class="container"><table class="table"> <thead><tr><th>Sample</th><th>Target 1</th><th>Target 2</th><th>Target 3</th><th>Target 4</th></tr></thead><tbody>'

            test_file,sample_name = convertPredictionFile(filepath)
            features = test_file.columns.values

            X = []
            for i in range(len(features)):
                X.append(test_file[features[i]].values)

            X = np.array(X).T

            max, min = findTrainFeatureMaxMin()

            X_std = (X - min) / (max - min)
            X = X_std * (1 - 0) + 0
            x = torch.Tensor(X).to(device)

            in_dim = x.shape[1]
            out_dim = 4



            model = MLP(in_dim=in_dim, out_dim=out_dim).to(device)
            checkpoint = torch.load(f'checkpoints/best_model.pt', map_location=device)
            model.load_state_dict(checkpoint['MLP'])


            model.eval()
            y_hat = model(x)

            y_hat = y_hat.cpu().detach().numpy()

            for i in range(len(sample_name)):
                output += '<tr class="table-info"><td>'
                output += sample_name[i]
                output += '</td><td>'
                output += str(y_hat[i][0])
                output += '</td><td>'
                output += str(y_hat[i][1])
                output += '</td><td>'
                output += str(y_hat[i][2])
                output += '</td><td>'
                output += str(y_hat[i][3])

            os.remove(filepath) # delete cache
            return render_template('upload.html', result=output)


        else:
            output = 'The file extension is not allowed.'
            return render_template('upload.html', error=output)





if __name__ == '__main__':
    app.run(debug=True)
