# GNNsMemo


## Memorization score from SSL Encoders -

To compute the memorization score for a GCN as shown in [Localizing Memorization in SSL Vision Encoders](https://arxiv.org/abs/2409.19069), 

```python
cd MemorizationScoreSSL/
python train.py
```
1. By default the model used is a GCN, where we obtain the embeddings from the pre-final layer and compute the score. 
2. We can also use a Graph Autoencoder, by passing ```--model_type encoder```.
3. The test nodes are left untouched because we want to avoid any kind of train/test leakage. The train nodes are divided as candidate nodes, independent nodes and shared nodes. The memorization score is only calculated for the candidate nodes.
4. We get 5 csv files from 5 random seeds. We can then use ```logs/plotavgmemscore.py``` to plot the memorization scores.

## Effective Prediction Depth - 
To calculate effective prediction depth for GNNs based on  [Deep Learning Through the Lens of Example
Difficulty](https://proceedings.neurips.cc/paper/2021/file/5a4b25aaed25c2ee1b74de72dc03c14e-Paper.pdf)

