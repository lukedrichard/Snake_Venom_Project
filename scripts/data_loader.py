import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# dataset class
class ProteinEmbeddingsDataset(Dataset):
    def __init__(self, embeddings_path, metadata_path, split):
        self.embeddings = np.load(embeddings_path)
        self.metadata = pd.read_csv(metadata_path)

        # Ensure same order if needed (check your pipeline!)
        assert len(self.embeddings) == len(self.metadata), "Embeddings and metadata must align"

        #filter by split, preserve indices
        self.metadata = self.metadata[self.metadata['fold'].isin(split)]

        self.indices = self.metadata.index.tolist()

        #get labels
        self.labels = self.metadata['label_index'].values

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        embedding_idx = self.indices[idx]
        embedding = torch.tensor(self.embeddings[embedding_idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return embedding, label
    
def get_dataloader(protein_sequences_path, metadata_path, split, batch_size):

    dataset = ProteinEmbeddingsDataset(protein_sequences_path,
                                       metadata_path,
                                       split = split)

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=1, 
                            pin_memory=True)
    
    return dataloader