# Snake_Venom_Project

### Depndencies
`pip install -r requirements.txt`  
*PyTorch is downgraded for compatibility with K80 GPU.*

### Google Drive
This google drive folder has some documents used by our team and reasearch papers related to the topic:
[Google Drive Project Folder](https://drive.google.com/drive/u/0/folders/1VdOwGOoSgyxr-MO9OMySLKh4WcaKoVQp)

## Problem Definition
This project aims to correctly classify proteins sequences related to snake venom.<br><br>
We restrict ourselves to six protein families in our analysis: Serine Proteases (SVSP), Metalloproteinases (SVMP), Phospholipase A2 (PLA2), C-Type Lectins, Three-Finger Toxins (3FTX), and Disintegrins.

## Data Collection
Code for the data collection process can be found in `Notebooks/Dataset_Creation.ipynb`<br><br>
Raw protein sequences are collected as `.fasta` files through the [UniProt](https://www.uniprot.org/) rest api. These files are saved in `raw_data`<br>
These files are then parsed and saved in `metadata.csv` and `protein_sequences.csv`<br>
Finally, train/val/test splits of 80%/10%/10% are created. Which split each protein belongs to can be found in the metadata file.<br>
These can be used to train models on the *full dataset*<br><br>

Adittionally, the data is deduplicated using CD-HIT. These files can be found in `raw_data/Clustered_Fasta`<br>
For each protein family, two files are produced by the CD-HIT process.<br>
The filese ending with `.fasta.clst` shows which proteins were gropued together and which protein is used as the cluster representative.<br>
Those ending with `.fasta` are the deduplicated data that is used in most of the experiments.<br>
The `.fasta` files are then parsed to create `clustered_metadata.csv` and `clusterd_protein_sequence.csv`. These can be found in `raw_data`<br>
These files can be used to train models on the *deduplicated dataset*<br><br>

Furthermore, we create fragmented sequences of varying length out of the test sequences<br>
These are used to test model robustness. The data is stored in `raw_data/protein_sequences/fragmented_test_sequences`<br>

## Feature Extraction
Since raw protein sequences are strings of amino acids, this data needs to be vectorized into numerical representation for use in machine learning models. We used *k-mer frequency encoding*, a classic approach for handcrafted features. We also used embeddings from the pre-trained *protBERT* model<br>

For the k-mer feaures, ...<br>

For the *protBERT* embeddings, we simply acces the model through huggingface. The final state of the $[CLS]$ token is saved as the protein sequence embedding. This process can be run with `scripts/make_embeddings.py`. The embeddings can be found in `processed_data/embeddings`<br>

Futhermore, visualizations of the features can be generated with `Notebooks/t_sne.ipynb` and are stored in `plots`

## Models
Random Forest...<br>

We also implement 4-layer MLPs as classifiers. Model training and evaluation can be performed with ~/scripts/main.py. There are two MLP architecures. Both use 4-layers, but the hidden dimensions are different depending on whether k-mer features or embeddings are used. <br><br>

Additonaly the fragmented test sequences can be evaluated by running `scripts/robustness_testing.py`<br><br>

All Training results are stored in `results`<br>

## Future Work
The project can be expanded by considering more classes. Additional uniprot queries just need to be uncommented in `Dataset_Creation.ipynb`<br><br>

There are many other methods for hand-crafted feature extraction of protein sequences that have yet to be explored.<br>
Check the 
[Google Drive Project Folder](https://drive.google.com/drive/u/0/folders/1VdOwGOoSgyxr-MO9OMySLKh4WcaKoVQp) 
for research paper outlining other feature extraction methods and classifier models.<br><br>
Further work with *protBERT* embeddings could be done by keeping the entire embedding array for each protein rather than only using the $[CLS]$ representation.
A 1-Dimensional CNN could then be trained on the full embeddings.
