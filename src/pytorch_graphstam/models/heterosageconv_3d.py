import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import Linear, HeteroConv, LayerNorm, aggr
from ..utils.sageconv_3d import SAGEConv3D
from ..utils.tft_components import apply_time_distributed

class DirSageConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha, aggr):
        super(DirSageConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = SAGEConv3D(input_dim, output_dim, aggr=aggr, flow="source_to_target", root_weight=False)
        self.conv_dst_to_src = SAGEConv3D(input_dim, output_dim, aggr=aggr, flow="target_to_source", root_weight=False)
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
            if e[0] == e[2]:
                conv_dict[e] = SAGEConv3D(in_channels=-1,
                                          out_channels=out_channels,
                                          aggr='mean', #aggr.SoftmaxAggregation(learn = True, channels = out_channels), # 'mean',
                                          project=False,
                                          normalize=False,
                                          bias=True)

                #conv_dict[e] = DirSageConv(input_dim=-1, output_dim=out_channels, alpha=0.5, aggr=aggr.SoftmaxAggregation(learn = True, channels = out_channels))
            else:
                if first_layer:
                    if e[0] == e[2]:
                        conv_dict[e] = SAGEConv3D(in_channels=-1,
                                                  out_channels=out_channels,
                                                  aggr='mean', #aggr.SoftmaxAggregation(learn = True, channels = out_channels), # 'mean',
                                                  project=False,
                                                  normalize=False,
                                                  bias=True)

                        #conv_dict[e] = DirSageConv(input_dim=-1, output_dim=out_channels, alpha=0.5, aggr=aggr.SoftmaxAggregation(learn = True, channels = out_channels))
                    else:
                        conv_dict[e] = SAGEConv3D(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  aggr='mean', #aggr.SoftmaxAggregation(learn = True, channels = out_channels), # 'mean',
                                                  project=False,
                                                  normalize=False,
                                                  bias=True)

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
                 fh,
                 in_channels,
                 hidden_channels,
                 downsample_factor,
                 num_layers,
                 dropout,
                 node_types,
                 edge_types,
                 target_node_type,
                 feature_extraction_node_types,
                 encoder_only_node_types,
                 global_node_types):
        super().__init__()

        self.target_node_type = target_node_type
        self.feature_extraction_node_types = feature_extraction_node_types
        self.encoder_node_types = encoder_only_node_types
        self.decoder_node_types = list(set(feature_extraction_node_types) - set(encoder_only_node_types))
        self.global_node_types = global_node_types
        self.num_layers = num_layers
        self.fh = fh
        self.downsample_dim = int(hidden_channels / downsample_factor)
        self.downsample_factor = downsample_factor

        self.transformed_feat_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            if node_type == target_node_type or (node_type in self.feature_extraction_node_types):
                self.transformed_feat_dict[node_type] = torch.nn.Sequential(Linear(-1, self.downsample_dim),
                                                                            torch.nn.ReLU(),
                                                                            Linear(self.downsample_dim, self.downsample_dim),
                                                                            torch.nn.ReLU(),
                                                                            Linear(self.downsample_dim, self.downsample_dim),
                                                                            torch.nn.ReLU()
                                                                            )
            else:
                self.transformed_feat_dict[node_type] = torch.nn.Sequential(Linear(-1, self.downsample_dim),
                                                                            torch.nn.ReLU(),
                                                                            Linear(self.downsample_dim, self.downsample_dim),
                                                                            torch.nn.ReLU()
                                                                            )

        if self.downsample_factor > 1:
            # re-dim the decoder & global nodes for TFT
            self.upsample_feat_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                if (node_type in self.decoder_node_types) or (node_type in self.global_node_types):
                    self.upsample_feat_dict[node_type] = torch.nn.Sequential(Linear(-1, hidden_channels),
                                                                                torch.nn.ReLU(),
                                                                                Linear(hidden_channels, hidden_channels),
                                                                                torch.nn.ReLU())
            # re-dim target node
            self.upsample_layer = Linear(-1, hidden_channels)

        # Conv Layers
        self.conv_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroForecastSageConv(in_channels=in_channels if i == 0 else self.downsample_dim,
                                          out_channels=self.downsample_dim,
                                          dropout=dropout,
                                          node_types=node_types,
                                          edge_types=edge_types,
                                          target_node_type=target_node_type,
                                          first_layer=i == 0,
                                          is_output_layer=i == num_layers - 1,
                                          )

            self.conv_layers.append(conv)

    def forward(self, x_dict, edge_index_dict):

        # transform target node
        for node_type, x in x_dict.items():
            #print(f"preprocessing {node_type} : {x.shape}")
            x_dict[node_type] = self.transformed_feat_dict[node_type](x)
            #x_dict[node_type] = apply_time_distributed(self.transformed_feat_dict[node_type], x)  # node attributes shape: [num_nodes, time_steps, hidden_size]

        # obtain temporal variables for decoder; each tensor is shaped (B,FH,hidden_size)
        decoder_temporal_vars_dict = {key: x[:, -self.fh:, :] for key, x in x_dict.items() if key in self.decoder_node_types}

        # obtain static variables for decoder; each tensor is shaped (B,hidden_size)
        decoder_static_vars_dict = {key: x[:, -1, :] for key, x in x_dict.items() if key in self.global_node_types}

        # limit x_dict to historical time steps
        x_dict = {key: x[:, :-self.fh, :] if (key in self.decoder_node_types) or (key in self.global_node_types) else x for key, x in x_dict.items()}

        # run convolutions
        for i, conv in enumerate(self.conv_layers):
            x_dict = conv(x_dict, edge_index_dict)

        if self.downsample_factor > 1:
            # redim target
            x_target = apply_time_distributed(self.upsample_layer, x_dict[self.target_node_type])

            # redim decoder_temporal_vars
            for node_type, x in decoder_temporal_vars_dict.items():
                decoder_temporal_vars_dict[node_type] = self.upsample_feat_dict[node_type](x)

            # redim static/global vars
            for node_type, x in decoder_static_vars_dict.items():
                decoder_static_vars_dict[node_type] = self.upsample_feat_dict[node_type](x)
        else:
            x_target = x_dict[self.target_node_type]

        return x_target, decoder_temporal_vars_dict, decoder_static_vars_dict

