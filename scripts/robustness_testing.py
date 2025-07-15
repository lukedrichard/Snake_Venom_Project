from data_loader import get_dataloader
from mlp_architecture import MLPClassifier_protbert, MLPClassifier_kmer
from trainer import train
from evaluator import evaluate
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Set working directory: your/path/Snake_Venom_Project
os.chdir("/home/ldrich/Summer2025BHT/Workflow_Course/Snake_Venom_Project")

# Check current working directory
print("Current working directory:", os.getcwd())

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#set paths
metadata_path = 'raw_data/protein_sequences/fragmented_test_sequences/fragments_len25.csv'
protein_sequences_path = 'processed_data/embeddings/fragments/fragments_len25_embeddings.npy'
# !!! change for new experiment !!!
results_dir = 'results/fragments_len25/' 

#hyperparameter
batch_size = 64

#change depending on your embeddings
input_dim = 1024 #protBERT embedding dimension
#input_dim = 8420 #kmer embeddings dimension

hidden_dim = 512   
output_dim = 6 #number of protein classes      
dropout = 0.0

#load pre-trained model
model_path = 'results/deduplicated_protbert/mlp.pth'
#model = MLPClassifier_protbert(input_dim, output_dim, dropout)

model = torch.load(model_path, map_location='cpu', weights_only=False)


test_loader = get_dataloader(protein_sequences_path, metadata_path, split=['test'], batch_size=batch_size)


#make directory for results
os.makedirs(results_dir, exist_ok=True)

evaluate(device, model, data_loader=test_loader, results_dir=results_dir + 'test_')