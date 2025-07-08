import igraph as ig
import leidenalg
import torch

with open("./Dataset/MGTAB/edge_index.txt", "w") as f:
  edge_index = torch.load("./Dataset/MGTAB/edge_index.pt")
  edge_type = torch.load("./Dataset/MGTAB/edge_type.pt")
  for i in range(edge_index.shape[1]):
    if edge_type[i] == 1:
      src = edge_index[0, i].item()
      dst = edge_index[1, i].item()
      f.write(f"{src} {dst}\n")

g = ig.Graph().Read_Edgelist("./Dataset/MGTAB/edge_index.txt", directed=True)
community_index = torch.tensor(leidenalg.find_partition(g, leidenalg.ModularityVertexPartition).membership)
_, counts = torch.unique(community_index, return_counts=True)
n_valid_community = (counts >= 10).sum()
community_index[community_index >= n_valid_community] =  -1
torch.save(community_index, "./Dataset/MGTAB/community_index.pt")

print("Detected Communities: ", torch.unique(community_index))