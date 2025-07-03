import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import numpy as np
from tqdm import tqdm

#get .csv files
metadata = pd.read_csv('raw_data/metadata/metadata.csv')
sequences_df = pd.read_csv('raw_data/protein_sequences/protein_sequences.csv')

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
np.save('processed_data/embeddings/protbert_embeddings.npy',final_embeddings.numpy())

embeddings = np.load('processed_data/embeddings/protbert_embeddings.npy')
print(embeddings.shape)  # This should print (num_sequences, embedding_dim)
