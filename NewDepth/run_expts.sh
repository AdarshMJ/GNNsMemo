#3164711608,894959334,2487307261,3349051410,493067366
# echo "Running baseline..."
# python baselinegcn.py \
#   --dataset texas.npz \
#   --out changeval.csv \
#   --model GATv2 \
#   --lr 0.001 \
#   --hidden_dimension 32 \
#   --num_epochs 500 \
#   --num_layers 5 \
#   --budget_edges_add 0 \
#   --budget_edges_delete 0

echo "Running baseline..."
python baselinegcn.py \
  --dataset Cora \
  --out changeval.csv \
  --model SimpleGCN \
  --lr 0.01 \
  --hidden_dimension 32 \
  --num_epochs 1200 \
  --num_layers 5 \
  --budget_edges_add 0 \
  --budget_edges_delete 0

# echo "Running baseline..."
# python baselinegcn.py \
#   --dataset Citeseer \
#   --out changeval.csv \
#   --model GATv2\
#   --lr 0.01 \
#   --hidden_dimension 32 \
#   --num_epochs 1200 \
#   --num_layers 5 \
#   --budget_edges_add 0 \
#   --budget_edges_delete 0
