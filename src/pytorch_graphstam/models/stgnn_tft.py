import torch
import torch.nn.functional as F
import torch_geometric
from ..utils.tft_components import *
from .heterosageconv_3d import HeteroGraphSAGE

class STGNN(torch.nn.Module):
    def __init__(self,
                 metadata,
                 target_node,
                 global_node_types,
                 feature_extraction_node_types,
                 encoder_only_node_types,
                 hist_len,
                 fh,
                 hidden_channels=64,
                 num_layers=1,
                 num_rnn_layers=1,
                 num_attn_layers=1,
                 n_quantiles=1,
                 heads=1,
                 dropout=0.0,
                 tweedie_out=False,
                 layer_type='SAGE',
                 downsample_factor=1,
                 chunk_size=None,
                 device='cuda'):

        super(STGNN, self).__init__()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.hist_len = hist_len
        self.fh = fh
        self.n_quantiles = n_quantiles
        self.tweedie_out = tweedie_out
        self.layer_type = layer_type
        self.num_rnn_layers = num_rnn_layers
        self.num_attn_layers = num_attn_layers
        self.chunk_size = chunk_size
        self.global_node_types = global_node_types
        self.feature_extraction_node_types = feature_extraction_node_types
        self.encoder_only_node_types = encoder_only_node_types
        self.device = torch.device(device)

        # variable selection layers for decoder
        self.static_var_select_layer = VariableSelectionStatic(hidden_layer_size=hidden_channels,
                                                               dropout_rate=dropout,
                                                               device=self.device)

        self.temporal_var_select_layer = VariableSelectionTemporal(hidden_layer_size=hidden_channels,
                                                                   static_context=True,
                                                                   dropout_rate=dropout,
                                                                   device=self.device)
        # static context layer
        self.static_context = StaticContexts(hidden_layer_size=hidden_channels,
                                             dropout_rate=dropout,
                                             device=self.device)

        # LSTM Layer of TFT
        self.lstm_layer = LSTMLayer(hidden_layer_size=hidden_channels,
                                    rnn_layers=num_rnn_layers,
                                    dropout_rate=dropout,
                                    device=self.device)

        # static enrichment layer
        self.static_enrichment_layer = StaticEnrichmentLayer(hidden_layer_size=hidden_channels,
                                                             context=True,
                                                             dropout_rate=dropout,
                                                             device=self.device)

        # Attention Layer
        self.self_attention_layer = AttentionStack(num_layers=num_attn_layers,
                                                   hidden_layer_size=hidden_channels,
                                                   n_head=heads,
                                                   dropout_rate=dropout,
                                                   device=self.device)

        # Final Gating Layer
        self.pff_layer = FinalGatingLayer(hidden_layer_size=hidden_channels,
                                          dropout_rate=dropout,
                                          device=self.device)

        # Quantile Projections
        self.projection_layer = torch_geometric.nn.Linear(hidden_channels, self.n_quantiles)

        if layer_type == 'SAGE':
            self.gnn_model = HeteroGraphSAGE(fh=fh,
                                             in_channels=(-1,-1),
                                             hidden_channels=hidden_channels,
                                             downsample_factor=downsample_factor,
                                             num_layers=num_layers,
                                             dropout=dropout,
                                             node_types=self.node_types,
                                             edge_types=self.edge_types,
                                             target_node_type=target_node,
                                             feature_extraction_node_types=feature_extraction_node_types,
                                             encoder_only_node_types=self.encoder_only_node_types,
                                             global_node_types=self.global_node_types)

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

        # get key_aggregation_status
        key_level_index = x_dict['key_aggregation_status']
        key_level_list = key_level_index.unique().tolist()

        # del keybom from x_dict
        del x_dict['keybom']
        del x_dict['key_aggregation_status']
        del x_dict['scaler']

        # gnn model
        gnn_embedding, decoder_temporal_vars_dict, decoder_static_vars_dict = self.gnn_model(x_dict, edge_index_dict)

        #print("cuda usage after gnn compute: ", torch.cuda.max_memory_allocated() / 1024 ** 3)
        # print gnn_embedding shape
        #print("gnn embedding: ", gnn_embedding.shape)

        #for k, v in decoder_temporal_vars_dict.items():
        #    print(f"decoder temporal embedding {k} : {v.shape}")

        #for k, v in decoder_static_vars_dict.items():
        #    print(f"decoder static embedding {k} : {v.shape}")

        # static variable selection layer for decoder
        static_var_names = []
        static_var_tensors = []
        for k, v in decoder_static_vars_dict.items():
            static_var_names.append(k)
            static_var_tensors.append(v)

        static_vec, static_weights = self.static_var_select_layer(static_var_tensors)
        #print("static vec: ", static_vec.shape)

        # static_context layer for init
        context_vec, enrich_vec, h_vec, c_vec = self.static_context(static_vec)
        #print("context vec: ", context_vec.shape)

        # temp variable selection layer for decoder
        temporal_var_names = []
        temporal_var_tensors = []
        for k, v in decoder_temporal_vars_dict.items():
            temporal_var_names.append(k)
            temporal_var_tensors.append(v)

        decoder_lstm_input, temporal_weights = self.temporal_var_select_layer([temporal_var_tensors, context_vec])
        #print("decoder lstm input: ", decoder_lstm_input.shape)

        # lstm layer
        lstm_init_states = [h_vec, c_vec]
        lstm_inputs = [gnn_embedding, decoder_lstm_input, lstm_init_states] # (encoder_in, decoder_in, init_states)
        temporal_features = self.lstm_layer(lstm_inputs)
        #print("temporal features: ", temporal_features.shape)

        # static enrichment
        enriched_features = self.static_enrichment_layer([temporal_features, enrich_vec])
        #print("enrichment features: ", enriched_features.shape)

        # causal mask for decoder length
        mask = causal_mask(self.hist_len + self.fh)
        mask = mask.to(self.device)
        #print("mask: ", mask.shape)

        # Attention stack
        attn_out = self.self_attention_layer(enriched_features, mask, padding_mask=None)
        #print("attention out: ", attn_out.shape)
        #print("cuda usage after attn compute: ", torch.cuda.max_memory_allocated() / 1024 ** 3)

        # pff
        attn_out = self.pff_layer([attn_out, temporal_features])
        #print("pff out: ", attn_out.shape)

        # forecast quantiles
        attn_out = attn_out[:, -self.fh:, :]
        out = self.projection_layer(attn_out)
        #print("projection out: ", out.shape)

        device = self.device

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
