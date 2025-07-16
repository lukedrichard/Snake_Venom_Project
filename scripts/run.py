from data_loader import get_dataloader
from mlp_architecture import get_mlp_protbert, get_mlp_kmer
from trainer import train
from evaluator import evaluate
import torch
import torch.nn as nn
import torch.optim as optim
import os

'''Certain configurations need to be set by hand. Double check before running script.
These include: input_dim, metadata_path, protein_sequences_path, results_dir, model'''


# hyperparameters
batch_size = 512
learning_rate = 1e-3
num_epochs = 500

### change depending on your embeddings
#input_dim = 1024 #protBERT embedding dimension
input_dim = 8420 #kmer feature dimension

output_dim = 6 #number of protein classes      
dropout = 0.7

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set paths
metadata_path = '../raw_data/metadata/clustered_metadata.csv'
protein_sequences_path = '../processed_data/embeddings/kmer_embeddings.npy'
# !!! change for new experiment !!!
results_dir = '../results/deduplicated_kmer/' 

#make directory for results
os.makedirs(results_dir, exist_ok=True)

#create dataloaders
train_loader = get_dataloader(protein_sequences_path, metadata_path, split=['train'], batch_size=batch_size)
val_loader = get_dataloader(protein_sequences_path, metadata_path, split=['val'], batch_size=batch_size)

### instantiate model: depends on features being used
#model = get_mlp_protbert(input_dim, output_dim, dropout)
model = get_mlp_kmer(input_dim, output_dim, dropout)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.to(device) #move model to device


train(device, model, num_epochs, train_loader, val_loader, criterion, optimizer, results_dir)

evaluate(device, model, data_loader=train_loader, results_dir=results_dir + 'train_')
evaluate(device, model, data_loader=val_loader, results_dir=results_dir + 'val_')

#If doing final test evaluation
test_loader = val_loader = get_dataloader(protein_sequences_path, metadata_path, split=['test'], batch_size=batch_size)
evaluate(device, model, data_loader=test_loader, results_dir=results_dir + 'test_')

# Save the model
torch.save(model, results_dir + 'mlp.pth')