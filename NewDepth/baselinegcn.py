import warnings
warnings.filterwarnings('ignore')
import argparse
import sys
import os
import numpy as np
import random
import time
import csv
from model import *
from dataloader import *
from avgentropybaseline import *
from analysis import *
from onlysim import *

parser = argparse.ArgumentParser(description='Run NodeClassification+Rewiring script')
parser.add_argument('--dataset', type=str, help='Dataset to download')
parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in GCN')
parser.add_argument('--model', type=str, default='SimpleGCNRes', choices=['GCN', 'GATv2','SimpleGCN','SimpleGCNRes','AsymGNN'], help='Model to use')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate = [0.01,0.001]')
parser.add_argument('--out', type=str, help='name of log file')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout = [Cora - 0.4130296, Citeseer - 0.3130296]')
parser.add_argument('--hidden_dimension', type=int, default=32, help='Hidden Dimension size')
parser.add_argument('--device',type=str,default='cuda',help='Device to use')
parser.add_argument('--num_train',type=int,default='20',help='Number of training nodes per class')
parser.add_argument('--num_val',type=int,default='500',help='Number of validation nodes')
parser.add_argument('--num_epochs',type=int,default='100',help='Number of training epochs')
parser.add_argument('--budget_edges_add',type=int,default='0',help='Number of edges to add')
parser.add_argument('--budget_edges_delete',type=int,default='0',help='Number of edges to delete')
#parser.add_argument('--seed',type=int,default='3164711608',help='Random seed')
args = parser.parse_args()




filename = args.out
p = args.dropout
hidden_dimension = args.hidden_dimension
lr = args.lr



print("Loading dataset...")

if args.dataset in ['Cora','Citeseer','Pubmed','CS','Physics','Computers','Photo','DBLP','penn94','reed98']:
    data, num_classes,num_features,num_train_nodes,num_test_nodes,num_val_nodes = load_data(args.dataset,args.num_train,args.num_val)


elif args.dataset in ['Roman-empire','Minesweeper','Amazon-ratings','Questions']:
    data, num_classes,num_features = load_data(args.dataset)

elif args.dataset in ['cornell.npz','texas.npz','wisconsin.npz']:
    path = '/Users/adarshjamadandi/Desktop/Submissions/MemorizationGNNs/Memv2_Nov10/data'
    filepath = os.path.join(path, args.dataset)
    data = np.load(filepath)
    print("Converting to PyG dataset...")
    x = torch.tensor(data['node_features'], dtype=torch.float)
    y = torch.tensor(data['node_labels'], dtype=torch.long)
    edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
    train_mask = torch.tensor(data['train_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    val_mask = torch.tensor(data['val_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    test_mask = torch.tensor(data['test_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    num_classes = len(torch.unique(y))
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.num_classes = num_classes
    print(f"Selecting the LargestConnectedComponent..")
    transform = LargestConnectedComponents()
    data = transform(data)
    print()
    print("Splitting datasets train/val/test...")
    transform2 = RandomNodeSplit(split="test_rest",num_splits=100,num_test=0.3,num_val=0.1)
    data  = transform2(data)
    data = data.to(device)
    print(data)
    num_features = data.num_features
    num_classes = data.num_classes
    print("Done!..")

elif args.dataset in ['chameleon_filtered.npz','squirrel_filtered.npz','actor.npz']:
    path = '/Users/adarshjamadandi/Desktop/Submissions/MemorizationGNNs/Memv2_Nov10/data'
    filepath = os.path.join(path, args.dataset)
    data = np.load(filepath)
    print("Converting to PyG dataset...")
    x = torch.tensor(data['node_features'], dtype=torch.float)
    y = torch.tensor(data['node_labels'], dtype=torch.long)
    edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
    train_mask = torch.tensor(data['train_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    val_mask = torch.tensor(data['val_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    test_mask = torch.tensor(data['test_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    num_classes = len(torch.unique(y))
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.num_classes = num_classes
    print(f"Selecting the LargestConnectedComponent..")
    transform = LargestConnectedComponents()
    data = transform(data)
    print("Splitting datasets train/val/test...")
    # transform2 = RandomNodeSplit(split="test_rest",num_splits=100,num_train_per_class = 10,num_val=20)
    # #transform2 = RandomNodeSplit(split="test_rest",num_splits=100,num_test=0.2,num_val=0.2)
    # data  = transform2(data)
    transform2 = RandomNodeSplit(split="test_rest", num_splits=100, num_train_per_class=20, num_val=500)
    data = transform2(data)

    print()    
    data = data.to(device)
    print(data)
    num_features = data.num_features
    num_classes = data.num_classes
    print("Done!..")

else :
    print("Invalid dataset")
    sys.exit()



print()
num_train_nodes = data.train_mask.sum().item()
num_val_nodes = data.val_mask.sum().item()
num_test_nodes = data.test_mask.sum().item()
print(f"Number of training nodes: {num_train_nodes/100}")
print(f"Number of validation nodes: {num_val_nodes/100}")
print(f"Number of test nodes: {num_test_nodes/100}")
print()

# print("Feature similarity based rewiring...")
# print("Peforming community detection...")
# G = to_networkx(data, to_undirected=True)
# #nxcg_G = nxcg.from_networkx(G) 
# communities = list(nx.community.louvain_communities(G, seed=42))
# #cluster_dict_before = {node: i for i, cluster in enumerate(communities) for node in cluster}
# #cluster_list_before = [cluster_dict_before[node] for node in range(len(data.y))]

# algo_stime = time.time()
# data = modify_graph(data,args.dataset,G,communities,args.budget_edges_add,args.budget_edges_delete)
# algo_etime = time.time()
# rewire_time =  algo_etime - algo_stime
# print(f"Time Taken for Rewiring : {rewire_time}")

# print()

print("Start Training...")


def create_model(model, num_features, num_classes, hidden_dimension):
    if model == 'GATv2':
        return GATv2(num_features, hidden_dimension, num_classes, args.num_layers)
    elif model == 'SimpleGCN':
        return SimpleGCN(num_features, num_classes, hidden_dimension, args.num_layers)
    elif model == 'SimpleGCNRes':
        return SimpleGCNRes(num_features, num_classes, hidden_dimension, args.num_layers)
    elif model == 'AsymGNN':
        return AsymGNN(num_features, num_classes, hidden_dimension, args.num_layers)
    else:
        print("Invalid Model")
        sys.exit()

# Create model class factory
def create_model_instance():
    return create_model(args.model, num_features, num_classes, args.hidden_dimension).to(device)

# Create optimizer factory - modify this to return the optimizer parameters
def create_optimizer_params():
    return {
        'lr': args.lr,
        'weight_decay': 3e-5
    }

print(args.model)

print(f"Model Architecture: {args.model}")
model = create_model_instance()
print(f"Model Details:\n{model}")

gcn_start = time.time()

# Run training with multiple seeds
#seeds = [3164711608, 894959334, 2487307261, 3349051410, 493067366]
seeds = [3164711608]
all_results = run_multiple_seeds(
    data=data,
    model_class=create_model_instance,
    optimizer_class=torch.optim.Adam,  # Just pass the optimizer class
    optimizer_params=create_optimizer_params(),  # Pass optimizer parameters separately
    num_epochs=args.num_epochs,
    dataset_name=args.dataset,
    num_layers=args.num_layers,
    seeds=seeds
)

gcn_end = time.time()

# Calculate final metrics across all seeds
final_test_accs = [results[0][0] for results in all_results]  # First index for test accuracy
final_val_accs = [results[1][0] for results in all_results]   # First index for validation accuracy
final_train_accs = [results[2][0] for results in all_results] # First index for training accuracy

avg_test_acc_after = np.mean(final_test_accs)
std_dev_after = 2 * np.std(final_test_accs) / np.sqrt(len(final_test_accs))

avg_val_acc_after = np.mean(final_val_accs)
std_devval_after = 2 * np.std(final_val_accs) / np.sqrt(len(final_val_accs))

avg_train_acc_after = np.mean(final_train_accs)
std_devtrain_after = 2 * np.std(final_train_accs) / np.sqrt(len(final_train_accs))

# Write results to CSV
headers = ['Method','Model','Dataset','AvgTrainAcc','DevTrain','AvgTestAccAfter','DeviationAfter',
        'HiddenDim','LR','Dropout','TrainNodes','TestNodes','ValNodes','GCNTime']

with open(filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    if file.tell() == 0:
        writer.writerow(headers)
    writer.writerow(['Baseline',args.model, args.dataset,
                    f"{avg_train_acc_after:.4f}", f"{std_devtrain_after:.4f}", 
                    f"{avg_test_acc_after:.4f}", f"{std_dev_after:.4f}",
                    hidden_dimension, lr, p, 
                    num_train_nodes/100, num_test_nodes/100, num_val_nodes/100, 
                    gcn_end - gcn_start])