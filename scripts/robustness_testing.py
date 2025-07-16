from data_loader import get_dataloader
from mlp_architecture import MLPClassifier_protbert, MLPClassifier_kmer
from trainer import train
from evaluator import evaluate
import torch
import torch.nn as nn
import torch.optim as optim
import os


'''Certain configurations need to be set by hand. Double check before running script.
These include: metadata_path, protein_sequences_path, results_dir, model_path'''

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#set paths
metadata_path = '../raw_data/protein_sequences/fragmented_test_sequences/fragments_len25.csv'
protein_sequences_path = '../processed_data/embeddings/fragments/fragments_len25_embeddings.npy'
# !!! change for new experiment !!!
results_dir = '../results/fragments_len25/' 

#hyperparameter
batch_size = 64


hidden_dim = 512   
output_dim = 6 #number of protein classes      
dropout = 0.0

#load pre-trained model
model_path = '../results/deduplicated_protbert/mlp.pth'
model = torch.load(model_path, map_location='cpu', weights_only=False)


test_loader = get_dataloader(protein_sequences_path, metadata_path, split=['test'], batch_size=batch_size)


#make directory for results
os.makedirs(results_dir, exist_ok=True)

evaluate(device, model, data_loader=test_loader, results_dir=results_dir + 'test_')