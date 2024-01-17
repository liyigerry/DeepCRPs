# DeepCRPs

The pre-trained language model used in the article comes from: https://github.com/rostlab/SeqVec. The embeddings used in the article are obtained through SeqVec, and the files in the "embedding" folder contain the embeddings obtained through SeqVec. The training dataset for constructing graphs is created by using these embeddings and PDB files.

The "Data" folder contains the original PDB files, while the "rmsd" folder stores the data for structural comparison mentioned in the article.

To build the graph data, execute the `create_graph.py` script. Choose the features you want to use in the `data_process/graphUtil.py` file.

To perform machine learning for classifying CRPs, run the `ml.py` script.

The `GRU4fold.py` script includes both GRU and LSTM methods for CRP classification. Execute it to use these methods.

To initiate the training of DeepCRPs, run the `train.py` script.

To obtain explanations using GNN, run the `graphexplain.py` script.
