import torch
import torch_geometric
from heterosageconv import HeteroGraphSAGE
from hgtconv import HGT

class STGNN(torch.nn.Module):
    def __init__(self,
                 metadata,
                 target_node,
                 feature_extraction_node_types,
                 time_steps,
                 feature_dim=64,
                 hidden_channels=64,
                 num_layers=1,
                 num_rnn_layers=1,
                 n_quantiles=1,
                 heads=1,
                 dropout=0.0,
                 tweedie_out=False,
                 skip_connection=False,
                 feature_transform=True,
                 layer_type='SAGE',
                 chunk_size=None):

        super(STGNN, self).__init__()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.time_steps = time_steps
        self.n_quantiles = n_quantiles
        self.tweedie_out = tweedie_out
        self.layer_type = layer_type
        self.num_rnn_layers = num_rnn_layers
        self.chunk_size = chunk_size
        self.feature_transform = feature_transform

        # lstm & projection layers
        self.sequence_layer = torch.nn.LSTM(input_size=int(hidden_channels),
                                            hidden_size=hidden_channels,
                                            num_layers=num_rnn_layers,
                                            batch_first=True)

        self.global_context_transform_layer = torch.nn.Sequential(torch_geometric.nn.Linear(-1, hidden_channels),
                                                                  torch.nn.ReLU(),
                                                                  torch_geometric.nn.Linear(hidden_channels, hidden_channels),
                                                                  torch.nn.ReLU())

        self.known_covar_transform_layer = torch.nn.Sequential(torch_geometric.nn.Linear(-1, hidden_channels),
                                                               torch.nn.ReLU(),
                                                               torch_geometric.nn.Linear(hidden_channels, hidden_channels),
                                                               torch.nn.ReLU())

        self.h_layer = torch.nn.Sequential(torch_geometric.nn.Linear(-1, hidden_channels),
                                           torch.nn.ReLU(),
                                           torch_geometric.nn.Linear(hidden_channels, hidden_channels),
                                           torch.nn.ReLU())

        self.c_layer = torch.nn.Sequential(torch_geometric.nn.Linear(-1, hidden_channels),
                                           torch.nn.ReLU(),
                                           torch_geometric.nn.Linear(hidden_channels, hidden_channels),
                                           torch.nn.ReLU())

        self.projection_layer = torch_geometric.nn.Linear(hidden_channels, self.n_quantiles)

        if layer_type == 'SAGE':
            self.gnn_model = HeteroGraphSAGE(in_channels=feature_dim,
                                             hidden_channels=hidden_channels,
                                             num_layers=num_layers,
                                             num_rnn_layers=num_rnn_layers,
                                             dropout=dropout,
                                             node_types=self.node_types,
                                             edge_types=self.edge_types,
                                             target_node_type=target_node,
                                             feature_extraction_node_types=feature_extraction_node_types,
                                             skip_connection=skip_connection,
                                             feature_transform=feature_transform)
        elif layer_type == 'HGT':
            self.gnn_model = HGT(in_channels=-1,
                                 metadata=metadata,
                                 target_node_type=target_node,
                                 feature_extraction_node_types=feature_extraction_node_types,
                                 num_layers=num_layers,
                                 hidden_channels=hidden_channels,
                                 heads=heads,
                                 feature_transform=feature_transform)

    @staticmethod
    def sum_over_index(x, x_wt, x_index):
        # re-scale outputs
        x = torch.mul(x, x_wt)
        return torch.index_select(x, 0, x_index).sum(dim=0)

    @staticmethod
    def log_transformed_sum_over_index(x, x_wt, x_index):
        """
        For tweedie, the output is expected to be the log of required prediction, so, reverse log transform before aggregating.
        """
        x = torch.exp(x)
        x = torch.mul(x, x_wt)
        return torch.index_select(x, 0, x_index).sum(dim=0)

    def forward(self, x_dict, edge_index_dict):
        # get keybom
        keybom = x_dict['keybom']
        keybom = keybom.type(torch.int64)
        scaler = x_dict['scaler']
        known_covars_future = x_dict['known_covars_future']
        global_context = x_dict['global_context']

        # get key_aggregation_status
        key_level_index = x_dict['key_aggregation_status']
        key_level_list = key_level_index.unique().tolist()

        # del keybom from x_dict
        del x_dict['keybom']
        del x_dict['key_aggregation_status']
        del x_dict['scaler']
        del x_dict['known_covars_future']
        del x_dict['global_context']

        # gnn model
        gnn_embedding = self.gnn_model(x_dict, edge_index_dict)

        # get device
        device_int = gnn_embedding.get_device()
        if device_int == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')

        global_context = self.global_context_transform_layer(global_context)
        known_covars_future = self.known_covar_transform_layer(known_covars_future)

        # loop over rest of time steps
        inp = gnn_embedding
        out_list = []
        h = self.h_layer(gnn_embedding)  # [batch, embed_dim]
        c = self.c_layer(gnn_embedding)
        h = torch.unsqueeze(h, dim=0)  # [1, batch, embed_dim]
        c = torch.unsqueeze(c, dim=0)
        h = h.repeat(self.num_rnn_layers, 1, 1)
        c = c.repeat(self.num_rnn_layers, 1, 1)

        #h = torch.zeros((self.num_rnn_layers, gnn_embedding.shape[0], gnn_embedding.shape[1])).to(device)
        #c = torch.zeros((self.num_rnn_layers, gnn_embedding.shape[0], gnn_embedding.shape[1])).to(device)

        for i in range(self.time_steps):
            current_inp = torch.add(torch.add(inp, known_covars_future[:, i, :]), global_context)
            o, (h, c) = self.sequence_layer(torch.unsqueeze(current_inp, dim=1), (h, c))
            out_list.append(o[:, -1, :])
            inp = torch.add(inp, o[:, -1, :])

        out = torch.stack(out_list, dim=1)  # (n_nodes, ts, hidden_channels)
        # projection
        out = self.projection_layer(out)  # (n_nodes, ts, n_quantiles)

        if keybom.shape[-1] == 1:
            # for non-hierarchical or single level hierarchy cases
            return out

        elif keybom.shape[-1] == 0:
            # for single ts cases
            return out

        else:
            # vectorized approach follows:
            dummy_out = torch.zeros(1, out.shape[1], out.shape[2]).to(device)
            # add a zero vector to the out tensor as workaround to the limitation of vmap of not being able to process
            # nested/dynamic shape tensors
            out = torch.cat([out, dummy_out], dim=0)
            # add dummy scale for last row in out
            scaler = torch.unsqueeze(scaler, 2)
            dummy_scaler = torch.ones(1, scaler.shape[1], scaler.shape[2]).to(device)
            scaler = torch.cat([scaler, dummy_scaler], dim=0)

            for key_level in key_level_list:
                if key_level == 0:
                    pass
                else:
                    keybom_kl = keybom[key_level_index[:, 0] == key_level]
                    keybom_kl[keybom_kl == -1] = int(out.shape[0] - 1)

                    # call vmap on sum_over_index function
                    if self.tweedie_out:
                        batched_sum_over_index = torch.vmap(self.log_transformed_sum_over_index,
                                                            in_dims=(None, None, 0),
                                                            randomness='error',
                                                            chunk_size=self.chunk_size)
                        kl_out = batched_sum_over_index(out, scaler, keybom_kl)
                        out[:-1][key_level_index[:, 0] == key_level] = kl_out / scaler[:-1][key_level_index[:, 0] == key_level]
                    else:
                        batched_sum_over_index = torch.vmap(self.sum_over_index,
                                                            in_dims=(None, None, 0),
                                                            randomness='error',
                                                            chunk_size=self.chunk_size)
                        kl_out = batched_sum_over_index(out, scaler, keybom_kl)
                        out[:-1][key_level_index[:, 0] == key_level] = kl_out / scaler[:-1][key_level_index[:, 0] == key_level]

            # call vmap on sum_over_index function
            if self.tweedie_out:
                out = out[:-1]
                # again do the log_transform on the aggregates
                out = torch.log(out)
                return out

            else:
                out = out[:-1]
                return out

