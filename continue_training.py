import torch

from neuralNetwork import MLP
from convert import convertTrainFile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = MLP(in_dim=in_dim, out_dim=out_dim).to(device)
checkpoint = torch.load(f'checkpoints/best_model.pt', map_location=device)
model.load_state_dict(checkpoint['MLP'])