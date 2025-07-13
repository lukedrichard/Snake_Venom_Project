import torch.nn as nn

#model architecture for classifying protbert embeddings
class MLPClassifier_protbert(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(MLPClassifier_protbert, self).__init__()
        self.output_dim = output_dim
        self.model = nn.Sequential(

                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout),

                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),

                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout),

                    nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.model(x)
    

#model architecture for classifying kmer embeddings
class MLPClassifier_kmer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(MLPClassifier_kmer, self).__init__()
        self.output_dim = output_dim
        self.model = nn.Sequential(

                    nn.Linear(input_dim, 2048),
                    nn.ReLU(),
                    nn.Dropout(dropout),

                    nn.Linear(2048, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout),

                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout),

                    nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.model(x)
    
def get_mlp_protbert(input_dim, output_dim, dropout):
    return MLPClassifier_protbert(input_dim, output_dim, dropout)

def get_mlp_kmer(input_dim, output_dim, dropout):
    return MLPClassifier_kmer(input_dim, output_dim, dropout)


