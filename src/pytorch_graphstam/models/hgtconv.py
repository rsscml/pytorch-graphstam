import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import Linear, HGTConv

class HGT(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 metadata,
                 target_node_type,
                 feature_extraction_node_types,
                 heads=1,
                 num_layers=1,
                 feature_transform=True):
        super().__init__()
        self.target_node_type = target_node_type
        self.feature_extraction_node_types = feature_extraction_node_types
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        target_edge_types = [edge_type for edge_type in metadata[1] if (edge_type[0] == target_node_type) and (edge_type[2] == target_node_type)]
        partial_metadata = ([target_node_type], target_edge_types)
        self.feature_transform = feature_transform

        if self.feature_transform:
            self.transformed_feat_dict = torch.nn.ModuleDict()
            for node_type in self.node_types:
                if (node_type == target_node_type) or (node_type in self.feature_extraction_node_types):
                    self.transformed_feat_dict[node_type] = torch.nn.Sequential(Linear(-1, in_channels),
                                                                                torch.nn.ReLU(),
                                                                                Linear(in_channels, in_channels),
                                                                                torch.nn.ReLU(),
                                                                                Linear(in_channels, in_channels),
                                                                                torch.nn.ReLU())
                else:
                    # self.transformed_feat_dict[node_type] = Linear(-1, in_channels)
                    # add 2-layer mlp for feature extraction
                    self.transformed_feat_dict[node_type] = torch.nn.Sequential(Linear(-1, in_channels),
                                                                                torch.nn.ReLU(),
                                                                                Linear(in_channels, in_channels),
                                                                                torch.nn.ReLU())

        self.conv_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                conv = HGTConv(in_channels=hidden_channels,
                               out_channels=hidden_channels,
                               metadata=metadata,
                               heads=heads)
            else:
                conv = HGTConv(in_channels=hidden_channels,
                               out_channels=hidden_channels,
                               metadata=partial_metadata,
                               heads=heads)

            self.conv_layers.append(conv)

    def forward(self, x_dict, edge_index_dict):

        if self.feature_transform:
            x_dict = {node_type: self.transformed_feat_dict[node_type] for node_type, x in x_dict.items()}

        for conv in self.conv_layers:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x for key, x in x_dict.items() if key == self.target_node_type}
            edge_index_dict = {key: x for key, x in edge_index_dict.items() if
                               (key[0] == self.target_node_type) and (key[2] == self.target_node_type)}

        return x_dict[self.target_node_type]

