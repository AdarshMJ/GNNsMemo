# GNNsMemo


## Memorization score from SSL Encoders -

To compute the memorization score for a GCN as shown in [Localizing Memorization in SSL Vision Encoders](https://arxiv.org/abs/2409.19069), 

```python
cd MemorizationScoreSSL/
python train.py
```
Notes
1. By default the model used is a GCN, where we obtain the embeddings from the pre-final layer and compute the score. 
2. We can also use a Graph Autoencoder, by passing ```--model_type encoder```.
3. The test nodes are left untouched because we want to avoid any kind of train/test leakage. The train nodes are divided as candidate nodes, independent nodes and shared nodes. The memorization score is only calculated for the candidate nodes.
4. We get 5 csv files from 5 random seeds. We can then use ```logs/plotavgmemscore.py``` to plot the memorization scores.

## Effective Prediction Depth - 
To calculate effective prediction depth for GNNs based on  [Deep Learning Through the Lens of Example
Difficulty](https://proceedings.neurips.cc/paper/2021/file/5a4b25aaed25c2ee1b74de72dc03c14e-Paper.pdf)

```python
cd NewDepth/
bash run_expts.sh
```
Notes
1. The main files are ```avgentropybaseline.py``` which contains all the training code and effective prediction depth code. The ```analysis.py``` has some functions for plotting. The ```baselinegcn.py``` file is the main file where the data loaders etc are defined.
2. In ```baselinegcn.py```, there is a provision to define the number of seeds to average over. Default is 5 seeds. When ```bash run_expts.sh``` is run, the code runs for 5 random seeds and produces a folder for each seed and a final ```aggregate``` folder which has the main plots.
3. It also has a lot of unnecessary logging of different metrics which might not be useful. The main plots are the effective prediction depth vs gnn accuracy and knn accuracy.
   

