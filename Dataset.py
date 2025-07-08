import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr
from utils import sample_mask


class MGTAB(InMemoryDataset):
    def __init__(self, root, prime=False, transform=None, pre_transform=None):
        self.prime = prime
        super().__init__(root, transform, pre_transform)
        torch.serialization.add_safe_globals([DataEdgeAttr])
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.root = root

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']


    def process(self):
        # Read data into huge `Data` list.

        edge_index = torch.load(self.root + "/edge_index.pt").to(torch.int64)
        edge_type = torch.load(self.root + "/edge_type.pt")
        edge_weight = torch.load(self.root + "/edge_weight.pt")
        stance_label = torch.load(self.root + "/labels_stance.pt")
        bot_label = torch.load(self.root + "/labels_bot.pt")

        features = torch.load(self.root + "/features.pt").to(torch.float32)
        if self.prime:
            communities = torch.load(self.root + "/community_index.pt").to(torch.int64)
            features = torch.cat((features, communities.unsqueeze(1)), dim=1)

        data = Data(x=features, edge_index=edge_index)
        data.edge_type = edge_type
        data.edge_weight = edge_weight
        data.y1 = stance_label
        data.y2 = bot_label
        sample_number = len(data.y1)

        train_idx = range(int(0.7*sample_number))
        val_idx = range(int(0.7*sample_number), int(0.9*sample_number))
        test_idx = range(int(0.9*sample_number), int(sample_number))

        data.train_mask = sample_mask(train_idx, sample_number)
        data.val_mask = sample_mask(val_idx, sample_number)
        data.test_mask = sample_mask(test_idx, sample_number)

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])