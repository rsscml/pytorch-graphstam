import torch
import copy
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import Linear, HeteroConv, SAGEConv, LayerNorm, aggr, DirGNNConv


class DirSageConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha, aggr):
        super(DirSageConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = SAGEConv(input_dim, output_dim, aggr=aggr, flow="source_to_target", root_weight=False)
        self.conv_dst_to_src = SAGEConv(input_dim, output_dim, aggr=aggr, flow="target_to_source", root_weight=False)
        self.lin_self = Linear(input_dim, output_dim)
        self.alpha = alpha

    def forward(self, x, edge_index):
        return (
                self.lin_self(x) + (1 - self.alpha) * self.conv_src_to_dst(x, edge_index) +
                self.alpha * self.conv_dst_to_src(x, edge_index)
        )

class HeteroForecastSageConv(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout,
                 edge_types,
                 node_types,
                 target_node_type,
                 first_layer,
                 is_output_layer=False):

        super().__init__()

        self.target_node_type = target_node_type

        conv_dict = {}
        for e in edge_types:
            if first_layer:
                if e[0] == e[2]:
                    conv_dict[e] = SAGEConv(in_channels=in_channels,
                                            out_channels=out_channels,
                                            aggr='mean', # aggr.SoftmaxAggregation(learn = True, channels = out_channels), # 'mean',
                                            project=False,
                                            normalize=False,
                                            bias=True)

                    # conv_dict[e] = DirSageConv(input_dim=-1, output_dim=out_channels, alpha=0.5, aggr=aggr.SoftmaxAggregation(learn = True, channels = out_channels))
                else:
                    conv_dict[e] = SAGEConv(in_channels=in_channels,
                                            out_channels=out_channels,
                                            aggr='mean', # aggr.SoftmaxAggregation(learn = True, channels = out_channels), # 'mean',
                                            project=False,
                                            normalize=False,
                                            bias=True)
            else:
                if e[0] == e[2]:
                    conv_dict[e] = SAGEConv(in_channels=out_channels,  # -1
                                            out_channels=out_channels,
                                            aggr='mean', #aggr.SoftmaxAggregation(learn = True, channels = out_channels), # 'mean',
                                            project=False,
                                            normalize=False,
                                            bias=True)

                #conv_dict[e] = DirSageConv(input_dim=-1, output_dim=out_channels, alpha=0.5, aggr=aggr.SoftmaxAggregation(learn = True, channels = out_channels))

        self.conv = HeteroConv(conv_dict)

        if not is_output_layer:
            self.dropout = torch.nn.Dropout(dropout)
            self.norm_dict = torch.nn.ModuleDict({
                node_type:
                    LayerNorm(out_channels, mode='node')
                for node_type in node_types if node_type == target_node_type
            })

        self.is_output_layer = is_output_layer

    def forward(self, x_dict, edge_index_dict):

        x_dict = self.conv(x_dict, edge_index_dict)

        if not self.is_output_layer:
            for node_type, norm in self.norm_dict.items():
                x_dict[node_type] = norm(self.dropout(x_dict[node_type]).relu())
        else:
            x_dict[self.target_node_type] = x_dict[self.target_node_type].relu()

        return x_dict


class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_layers,
                 num_rnn_layers,
                 dropout,
                 node_types,
                 edge_types,
                 target_node_type,
                 feature_extraction_node_types,
                 skip_connection=True,
                 feature_transform=True):
        super().__init__()

        self.target_node_type = target_node_type
        self.feature_extraction_node_types = feature_extraction_node_types
        self.skip_connection = skip_connection
        self.num_layers = num_layers
        self.num_rnn_layers = num_rnn_layers
        self.feature_transform = feature_transform

        if num_layers == 1:
            self.skip_connection = False

        if self.feature_transform:
            self.transformed_feat_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                if (node_type == target_node_type) or (node_type in self.feature_extraction_node_types):
                    #self.transformed_feat_dict[node_type] = torch.nn.LSTM(input_size=1,
                    #                                                      hidden_size=in_channels,
                    #                                                      num_layers=self.num_rnn_layers,
                    #                                                      batch_first=True)
                    # add 3-layer mlp for feature extraction
                    self.transformed_feat_dict[node_type] = torch.nn.Sequential(Linear(-1, in_channels),
                                                                                torch.nn.ReLU(),
                                                                                Linear(in_channels, in_channels),
                                                                                torch.nn.ReLU(),
                                                                                Linear(in_channels, in_channels),
                                                                                torch.nn.ReLU())
                else:
                    #self.transformed_feat_dict[node_type] = Linear(-1, in_channels)
                    # add 2-layer mlp for feature extraction
                    self.transformed_feat_dict[node_type] = torch.nn.Sequential(Linear(-1, in_channels),
                                                                                torch.nn.ReLU(),
                                                                                Linear(in_channels, in_channels),
                                                                                torch.nn.ReLU())


        # Conv Layers
        self.conv_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroForecastSageConv(in_channels=in_channels if i == 0 else hidden_channels,
                                          out_channels=hidden_channels,
                                          dropout=dropout,
                                          node_types=node_types,
                                          edge_types=edge_types,
                                          target_node_type=target_node_type,
                                          first_layer=i == 0,
                                          is_output_layer=i == num_layers - 1,
                                          )

            self.conv_layers.append(conv)

    def forward(self, x_dict, edge_index_dict):

        res_dict = x_dict.copy()

        if self.feature_transform:
            # transform target node
            for node_type, x in x_dict.items():
                x_dict[node_type] = self.transformed_feat_dict[node_type](x)
                """
                if (node_type == self.target_node_type) or (node_type in self.feature_extraction_node_types):
                    o, _ = self.transformed_feat_dict[node_type](torch.unsqueeze(x, dim=2))  # lstm input is 3-d (N,L,1)
                    x_dict[node_type] = o[:, -1, :]  # take last o/p (N,H)
                else:
                    x_dict[node_type] = self.transformed_feat_dict[node_type](x)
                """

        # run convolutions
        for i, conv in enumerate(self.conv_layers):
            x_dict = conv(x_dict, edge_index_dict)

            # apply skip connections
            if self.skip_connection:
                res_dict = {key: res_dict[key] for key in x_dict.keys()}
                x_dict = {key: x + res_x for (key, x), (res_key, res_x) in zip(x_dict.items(), res_dict.items()) if key == res_key}

                # update residual dict
                res_dict = x_dict.copy()

        return x_dict[self.target_node_type]
