import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# hyperparameters
batch_size = 64

learning_rate = 1e-4
num_epochs = 10

input_dim = 1024 #embedding dimension
hidden_dim = 512    
output_dim = 6 #number of protein classes      
dropout = 0.3


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#lookup dictionary for converting labels
metadata_df = pd.read_csv('raw_data/metadata/metadata.csv')
label_to_index = {label: index for index, label in enumerate(sorted(metadata_df['protein'].unique()))}
metadata_df['label_index'] = metadata_df['protein'].map(label_to_index)

# dataset class
class ProteinEmbeddingsDataset(Dataset):
    def __init__(self, embeddings_path, metadata_df, split):
        self.embeddings = np.load(embeddings_path)
        self.metadata = metadata_df

        # Ensure same order if needed (check your pipeline!)
        assert len(self.embeddings) == len(self.metadata), "Embeddings and metadata must align"

        #filter by split, preserve indices
        self.metadata = self.metadata[self.metadata['fold'] == split]
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



#model architecture
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
                    nn.Linear(input_dim, 768),
                    nn.ReLU(),
                    nn.Dropout(dropout),

                    nn.Linear(768, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout),

                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),

                    nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.model(x)



#create data loaders
train_dataset = ProteinEmbeddingsDataset('processed_data/embeddings/protbert_embeddings.npy',
                                         metadata_df,
                                         split = 'train')

train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=1, 
                          pin_memory=True)

val_dataset = ProteinEmbeddingsDataset('processed_data/embeddings/protbert_embeddings.npy',
                                       metadata_df,
                                       split = 'val')

val_loader = DataLoader(val_dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=1, 
                          pin_memory=True)



#instantiate model
model = MLPClassifier(input_dim, output_dim, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#move model to device
model.to(device)


#train the model
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()


            _, predicted = torch.max(outputs.data, 1)
            val_total += targets.size(0)
            val_correct += (predicted == targets).sum().item()

    val_epoch_loss = val_loss / len(val_loader)
    val_epoch_acc = 100 * val_correct / val_total

    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
          f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")


#Visualize loss and accuracy
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.savefig('loss_plot.png')  # Save loss plot
plt.close()  # Close the figure so it doesn't display


plt.figure()
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Over Epochs')
plt.savefig('accuracy_plot.png')  # Save accuracy plot
plt.close()  # Close the figure


#confusion matrix
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for batch in val_loader:
        inputs, labels = batch  # assuming labels are already numerical
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

#generate confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print(cm)

#plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Save accuracy plot
plt.close()  # Close the figure


print("\nLabel index to name mapping:")
for index, label in sorted(label_to_index.items()):
    print(f"{index}: {label}")


# Save the model
torch.save(model, 'protein_mlp.pth')