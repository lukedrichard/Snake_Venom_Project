import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import numpy as np
from tqdm import tqdm
import os

# Set working directory: your/path/Snake_Venom_Project
os.chdir("/home/ldrich/Summer2025BHT/Workflow_Course/Snake_Venom_Project")

# Check current working directory
print("Current working directory:", os.getcwd())


def make_protBERT_embeddings(protein_seuqences_path, output_path):
    #get .csv files
    #metadata = pd.read_csv(metadata_path)
    sequences_df = pd.read_csv(protein_seuqences_path)

    sequences = sequences_df['protein_sequence'].tolist()
    #for testing script
    #test_sequences = sequences_df['protein_sequence'].tolist()[:64]

    #load protBERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
    #load with safetensors to be compatible with torch==2.2.0
    model = BertModel.from_pretrained("Rostlab/prot_bert_bfd", trust_remote_code=True, use_safetensors=True) 
    model.eval() #set to evaluation mode

    #Configure
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device) #move model to gpu

    #hyperparameters
    batch_size = 16

    class ProteinSequenceDataset(Dataset):
        def __init__(self, sequences):
            self.sequences = sequences

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            sequence = self.sequences[idx]
            sequence = ' '.join(list(sequence))
            return sequence

    #custom collate function: need list of strings not tensor
    def collate_batch(batch_sequences):
        tokenized = tokenizer(batch_sequences, return_tensors='pt', padding=True, truncation=True)
        return tokenized

    #create dataloader
    dataset = ProteinSequenceDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn = collate_batch)


    #create the embeddings in batches
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Embedding Sequences'):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            cls_embeddings = outputs.last_hidden_state[:,0,:]
            all_embeddings.append(cls_embeddings.cpu())


    final_embeddings = torch.cat(all_embeddings, dim=0)
    np.save(output_path,final_embeddings.numpy())

    return


'''
metadata_path = "raw_data/metadata/clustered_metadata.csv"
protein_seuqences_path = "raw_data/protein_sequences/clustered_protein_sequences.csv"
output_path = "processed_data/embeddings/clustered_protbert_embeddings.npy"

make_protBERT_embeddings(protein_seuqences_path, output_path)

#check they are there and correct dimension
embeddings = np.load(output_path)
print(embeddings.shape)  # This should print (num_sequences, embedding_dim)
'''


input_dir = 'raw_data/protein_sequences/fragmented_test_sequences'
output_dir = 'processed_data/embeddings/fragments'
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):

    if file.endswith('.csv'):
        file_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f'{file.replace(".csv", "")}_embeddings.npy')


    make_protBERT_embeddings(file_path, output_path)

    #check they are there and correct dimension
    embeddings = np.load(output_path)
    print(embeddings.shape)  # This should print (num_sequences, embedding_dim)