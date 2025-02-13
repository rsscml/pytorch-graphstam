import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

##################################################
# Helper: Imitate Keras "TimeDistributed" in PyTorch
##################################################
def apply_time_distributed(module, x):
    """
    Imitate tf.keras.layers.TimeDistributed by:
      1) Flattening [B, T, ...] -> [B*T, ...]
      2) Applying the module
      3) Reshaping back to [B, T, ...]
    """
    if x.ndimension() <= 2:
        # e.g., [B, features], just apply
        return module(x)

    b, t, *others = x.size()
    #x_reshaped = x.view(b * t, *others)  # Flatten out the time dimension
    x_reshaped = x.reshape(b*t, *others)  #
    y = module(x_reshaped)
    if y.ndimension() == 2:
        # [B*T, out_dim] -> [B, T, out_dim]
        #y = y.view(b, t, y.size(-1))
        y = y.reshape(b, t, y.size(-1))
    else:
        # If there's an extra dimension, adjust accordingly
        #y = y.view(b, t, *y.shape[1:])
        y = y.reshape(b, t, *y.shape[1:])
    return y


##################################################
# 1) Masking Utils
##################################################
def create_padding_mask(seq):
    """
    Imitates:
      seq = tf.cast(tf.math.less(seq, 0), tf.float32)
      return seq[:, tf.newaxis, :]  # shape (batch_size, 1, seq_len)

    For simplicity, we assume you pass in a torch.Tensor `seq`.
    We'll produce shape [B, 1, seq_len], with 1.0 where seq < 0, else 0.
    """
    mask = (seq < 0).float()  # 1 if seq < 0, else 0
    # expand dims -> [B, 1, seq_len]
    return mask.unsqueeze(1)


def causal_mask(size):
    """
    Imitates:
      mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
      return mask  # shape (seq_len, seq_len), upper-triangular 1s
    """
    # create [size, size] with 1's above the diagonal
    # PyTorch: we can use torch.triu
    mask = torch.ones(size, size).triu(diagonal=1)
    # This mask has 0 on diagonal+below, 1 above diag => "causal" means we block future tokens
    return mask


##################################################
# 2) Scaled Dot-Product Attention
##################################################
def scaled_dot_product_attention(q, k, v, causal_mask=None, padding_mask=None):
    """
    Equivalent to the TF version:
      matmul_qk = tf.matmul(q, k, transpose_b=True)
      ...
      if causal_mask is not None:
          scaled_attention_logits += (causal_mask * -1e9)
      if padding_mask is not None:
          scaled_attention_logits += (padding_mask * -1e9)
      ...
      attention_weights = tf.nn.softmax(...)
      output = tf.matmul(attention_weights, v)
    """
    # q, k, v shapes: [..., seq_len_q, depth], [..., seq_len_k, depth]
    # We'll assume the leading dimensions (e.g. batch, heads) are already handled.
    matmul_qk = torch.matmul(q, k.transpose(-1, -2))
    dk = k.size(-1)
    scaled_logits = matmul_qk / (dk ** 0.5)

    #print("matmul_qk shape: ", matmul_qk.shape)
    #print("dk: ", dk)

    # Broadcast / add masks
    if causal_mask is not None:
        # causal_mask shape: [seq_len_q, seq_len_k] or broadcastable
        # We assume 1 where we should block, else 0 => multiply by -1e9
        scaled_logits = scaled_logits + (causal_mask * -1e9)

    if padding_mask is not None:
        # padding_mask shape: e.g. [B, 1, seq_len_k], must be broadcastable to scaled_logits
        scaled_logits = scaled_logits + (padding_mask * -1e9)

    # Softmax across seq_len_k dimension (the last dimension)
    attention_weights = F.softmax(scaled_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    #print("scaled dot prod attn output: ", output.shape)

    return output, attention_weights


##################################################
# 3) Multi-Head Attention
##################################################
class TFTMultiHeadAttention(nn.Module):
    """
    PyTorch equivalent of your 'TFTMultiHeadAttention' layer.
    """

    def __init__(self, n_head, d_model, device, dropout_rate=0.1):
        """
        n_head: number of attention heads
        d_model: total hidden dimension (must be divisible by n_head)
        """
        super().__init__()

        self.device = device
        self.n_head = n_head
        self.d_model = d_model
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_k = self.d_v = d_model // n_head
        self.dropout_rate = dropout_rate

        # We replicate your code: same vs_layer across all heads
        self.qs_layers = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)]).to(self.device)
        self.ks_layers = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)]).to(self.device)
        # One vs_layer shared by all heads
        vs_layer = nn.Linear(d_model, self.d_v, bias=False).to(self.device)
        self.vs_layers = nn.ModuleList([vs_layer for _ in range(n_head)])

        self.dropout = nn.Dropout(dropout_rate)
        self.w_o = nn.Linear(d_model, d_model, bias=False).to(self.device)

    def forward(self, q, k, v, causal_mask=None, padding_mask=None):
        """
        q, k, v: [batch, seq_len, d_model]
        causal_mask: [seq_len, seq_len] or broadcastable
        padding_mask: [batch, 1, seq_len] or broadcastable
        return: (outputs, attn)
          outputs: [batch, seq_len, d_model]
          attn: [n_head, batch, seq_len_q, seq_len_k]
        """
        heads = []
        attns = []

        for i in range(self.n_head):
            qs = self.qs_layers[i](q)  # [B, T, d_k]
            ks = self.ks_layers[i](k)  # [B, T, d_k]
            vs = self.vs_layers[i](v)  # [B, T, d_v]

            head, attn = scaled_dot_product_attention(qs, ks, vs, causal_mask, padding_mask)
            head = self.dropout(head)  # dropout on each head
            heads.append(head)
            attns.append(attn)

        # Stack heads => shape [n_head, B, T, d_v]
        head = torch.stack(heads, dim=0)  # (n_head, B, T, d_v)
        attn = torch.stack(attns, dim=0)  # (n_head, B, T, T)
        #print("head : ", head.shape)
        #print("attn : ", attn.shape)
        # Average across heads => shape [B, T, d_v]
        # ( or you could concatenate them if you prefer the standard MHA practice,
        #   but your code does reduce_mean across heads)
        if self.n_head > 1:
            out = torch.mean(head, dim=0)  # [B, T, d_v]
        else:
            out = head.squeeze(0)  # [B, T, d_v]

        out = self.w_o(out)  # project back to d_model
        out = self.dropout(out)

        return out, attn


##################################################
# 4) Gated Residual Network (GRN) + Sub-layers
##################################################

def get_activation_fn(activation_str):
    """
    Utility to map string to PyTorch activation function
    """
    if activation_str is None:
        return None
    activation_str = activation_str.lower()
    if activation_str == 'elu':
        return F.elu
    elif activation_str == 'tanh':
        return torch.tanh
    elif activation_str == 'sigmoid':
        return torch.sigmoid
    elif activation_str == 'softmax':
        return F.softmax
    # Add more as needed
    else:
        raise ValueError(f"Unsupported activation: {activation_str}")


class TFTLinearLayer(nn.Module):
    """
    Equivalent to 'tft_linear_layer' in Keras code:
    A Dense layer with optional time-distribution and optional activation.
    """

    def __init__(self, hidden_layer_size, device, activation=None, use_time_distributed=False, use_bias=True):
        super().__init__()
        self.use_time_distributed = use_time_distributed
        self.device = device
        self.layer = nn.Linear(in_features=0, out_features=0, bias=use_bias).to(self.device)
        # Will set in_features dynamically at forward if desired (or pass it in the init).
        # For simplicity, we assume you know the in_features at construction time in real usage.

        self.hidden_layer_size = hidden_layer_size
        self.activation_fn = get_activation_fn(activation)

        # We will build the linear layer on-the-fly once we see the input dimension
        # if you need strict initialization, you can do so after knowing input_dim.

    def forward(self, x):
        # If we haven't set the in_features/out_features yet, do so here
        if self.layer.in_features == 0:
            # x shape: [B, T, in_features] or [B, in_features]
            if x.ndimension() == 3:
                # [B, T, F]
                in_features = x.shape[-1]
            else:
                # [B, F]
                in_features = x.shape[-1]
            self.layer = nn.Linear(in_features, self.hidden_layer_size, bias=self.layer.bias is not None).to(self.device)

        if self.use_time_distributed:
            out = apply_time_distributed(self.layer, x)
        else:
            out = self.layer(x)

        if self.activation_fn is not None:
            out = self.activation_fn(out)
        return out


class TFTApplyMLP(nn.Module):
    """
    Equivalent to 'tft_apply_mlp'.
    """

    def __init__(self, hidden_size, output_size, output_activation=None, hidden_activation='tanh',
                 use_time_distributed=False):
        super().__init__()
        self.use_time_distributed = use_time_distributed

        # Hidden layer
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        if hidden_activation.lower() == "tanh":
            self.hidden_activation_fn = torch.tanh
        elif hidden_activation.lower() == "elu":
            self.hidden_activation_fn = F.elu
        else:
            raise ValueError(f"Unsupported activation {hidden_activation}")

        # Output layer
        self.out_layer = nn.Linear(hidden_size, output_size)
        if output_activation is None:
            self.out_activation_fn = None
        elif output_activation.lower() == "sigmoid":
            self.out_activation_fn = torch.sigmoid
        elif output_activation.lower() == "softmax":
            self.out_activation_fn = lambda x: F.softmax(x, dim=-1)
        else:
            raise ValueError(f"Unsupported activation {output_activation}")

    def forward(self, x):
        if self.use_time_distributed:
            x = apply_time_distributed(self.hidden_layer, x)
            x = self.hidden_activation_fn(x)
            x = apply_time_distributed(self.out_layer, x)
            if self.out_activation_fn is not None:
                x = self.out_activation_fn(x)
        else:
            x = self.hidden_layer(x)
            x = self.hidden_activation_fn(x)
            x = self.out_layer(x)
            if self.out_activation_fn is not None:
                x = self.out_activation_fn(x)
        return x


class TFTApplyGatingLayer(nn.Module):
    """
    Equivalent to 'tft_apply_gating_layer'.
    """

    def __init__(self, hidden_layer_size, device, dropout_rate=0.0, use_time_distributed=True, activation=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_time_distributed = use_time_distributed
        self.device = device
        self.hidden_layer_size = hidden_layer_size

        self.activation_fc = nn.Linear(in_features=0, out_features=0).to(self.device)
        self.gated_fc = nn.Linear(in_features=0, out_features=0).to(self.device)

        if activation is None:
            self.activation_fn = None
        elif activation.lower() == "elu":
            self.activation_fn = F.elu
        # Add other activations as desired
        else:
            raise ValueError(f"Unsupported activation {activation}")

    def forward(self, x):
        # If we haven't set the in_features/out_features yet, do so here
        if self.activation_fc.in_features == 0:
            # x shape: [B, T, in_features] or [B, in_features]
            if x.ndimension() == 3:
                # [B, T, F]
                in_features = x.shape[-1]
            else:
                # [B, F]
                in_features = x.shape[-1]
            self.activation_fc = nn.Linear(in_features, self.hidden_layer_size).to(self.device)
            self.gated_fc = nn.Linear(in_features, self.hidden_layer_size).to(self.device)

        x = self.dropout(x)
        if self.use_time_distributed:
            a = apply_time_distributed(self.activation_fc, x)
            g = apply_time_distributed(self.gated_fc, x)
        else:
            a = self.activation_fc(x)
            g = self.gated_fc(x)

        if self.activation_fn is not None:
            a = self.activation_fn(a)
        g = torch.sigmoid(g)  # gating uses sigmoid

        out = a * g
        return out, g


class TFTAddAndNormLayer(nn.Module):
    """
    Equivalent to 'tft_add_and_norm_layer'.
    """

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.layer_norm = None  # We'll build dynamically based on last dimension

    def forward(self, inputs):
        # inputs is a list or tuple: [skip, gating_layer]
        skip, gating_layer = inputs
        out = skip + gating_layer

        if self.layer_norm is None:
            # Build LayerNorm if needed, expecting to normalize over last dim
            norm_shape = out.shape[-1]
            self.layer_norm = nn.LayerNorm(norm_shape).to(self.device)

        out = self.layer_norm(out)
        return out


class TFTGRNLayer(nn.Module):
    """
    Equivalent to 'tft_grn_layer'.
    """

    def __init__(self,
                 device,
                 hidden_layer_size,
                 output_size=None,
                 dropout_rate=0.0,
                 use_time_distributed=True,
                 additional_context=False,
                 return_gate=False):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size if output_size is not None else hidden_layer_size
        self.use_time_distributed = use_time_distributed
        self.additional_context = additional_context
        self.return_gate = return_gate
        self.device = device

        # Main linear (skip connection)
        input_size = hidden_layer_size*self.output_size if output_size is not None else hidden_layer_size
        self.skip_layer = nn.Linear(input_size, self.output_size)

        # Hidden layers
        self.hidden_1 = TFTLinearLayer(hidden_layer_size=hidden_layer_size,
                                       activation=None,
                                       use_time_distributed=use_time_distributed,
                                       device=self.device)

        self.hidden_2 = TFTLinearLayer(hidden_layer_size=hidden_layer_size,
                                       activation=None,
                                       use_time_distributed=use_time_distributed,
                                       device=self.device)

        if additional_context:
            self.context_layer = TFTLinearLayer(hidden_layer_size=hidden_layer_size,
                                                activation=None,
                                                use_time_distributed=use_time_distributed,
                                                device=self.device,
                                                use_bias=False)
        else:
            self.context_layer = None

        # Gate
        self.gate = TFTApplyGatingLayer(hidden_layer_size=self.output_size,
                                        dropout_rate=dropout_rate,
                                        use_time_distributed=use_time_distributed,
                                        activation=None,
                                        device=self.device)

        # Add & Norm
        self.add_norm = TFTAddAndNormLayer(device=self.device)

    def forward(self, inputs):
        """
        If additional_context == True, inputs is (x, c)
        else inputs is just x
        """
        if self.additional_context:
            x, c = inputs
        else:
            x = inputs

        # skip
        skip = x
        # If skip dimension != output_size,
        # you might need another linear or shape transform.
        # Here we assume x.shape[-1] == hidden_layer_size
        # and skip_layer out_features == self.output_size.

        # Convert skip to [B, T, output_size] if time_distributed
        if self.use_time_distributed:
            skip_out = apply_time_distributed(self.skip_layer, skip)
        else:
            skip_out = self.skip_layer(skip)

        # hidden path
        h = self.hidden_1(x)
        if self.context_layer is not None:
            # broadcast c if necessary
            if h.ndimension() == 3 and c.ndimension() == 2:
                # expand c to [B, 1, C]
                c = c.unsqueeze(1)
            c_out = self.context_layer(c)
            h = h + c_out

        h = F.elu(h)
        h = self.hidden_2(h)

        gating_layer, gate = self.gate(h)
        out = self.add_norm([skip_out, gating_layer])

        if self.return_gate:
            return out, gate
        return out


##################################################
# 5) Variable Selection: Static
##################################################
class VariableSelectionStatic(nn.Module):
    """
    Equivalent to 'variable_selection_static'.
    """

    def __init__(self, hidden_layer_size, dropout_rate, device):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        # We will build sub-layers once we know num_vars
        self.num_vars = None
        self.grn_flat = None
        self.grn_vars = nn.ModuleList()
        self.device = device

    def build_layers(self, num_vars):
        self.num_vars = num_vars
        self.grn_flat = TFTGRNLayer(hidden_layer_size=self.hidden_layer_size,
                                    output_size=self.num_vars,
                                    dropout_rate=self.dropout_rate,
                                    use_time_distributed=False,
                                    additional_context=False,
                                    return_gate=False,
                                    device=self.device).to(self.device)

        for _ in range(num_vars):
            self.grn_vars.append(
                TFTGRNLayer(hidden_layer_size=self.hidden_layer_size,
                            output_size=None,
                            dropout_rate=self.dropout_rate,
                            use_time_distributed=False,
                            additional_context=False,
                            return_gate=False,
                            device=self.device).to(self.device)
            )

    def forward(self, inputs):
        """
        inputs: list of static vars, each shape [B, var_dim].
        We concatenate => [B, sum_of_var_dims].
        """
        if self.grn_flat is None:
            self.build_layers(num_vars=len(inputs))

        flatten = torch.cat(inputs, dim=1)  # [B, sum_of_var_dims]
        mlp_outputs = self.grn_flat(flatten)  # [B, num_vars]

        # Softmax over num_vars dimension
        static_weights = F.softmax(mlp_outputs, dim=-1)  # [B, num_vars]
        weights = static_weights.unsqueeze(-1)  # [B, num_vars, 1]

        # Transform each var
        transformed = []
        for i in range(self.num_vars):
            e = self.grn_vars[i](inputs[i])  # [B, hidden_layer_size]
            transformed.append(e)

        # Stack => [B, num_vars, hidden_layer_size]
        trans_embedding = torch.stack(transformed, dim=1)

        # Weighted sum
        combined = trans_embedding * weights  # broadcast
        static_vec = combined.sum(dim=1)  # [B, hidden_layer_size]

        return static_vec, static_weights


##################################################
# 6) Static Contexts
##################################################
class StaticContexts(nn.Module):
    """
    Equivalent to 'static_contexts'.
    """

    def __init__(self, hidden_layer_size, dropout_rate, device):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.device = device

        self.stat_vec_layer = TFTGRNLayer(hidden_layer_size=self.hidden_layer_size,
                                          output_size=None,
                                          dropout_rate=self.dropout_rate,
                                          use_time_distributed=False,
                                          additional_context=False,
                                          return_gate=False,
                                          device=self.device)

        self.enrich_vec_layer = TFTGRNLayer(hidden_layer_size=self.hidden_layer_size,
                                            output_size=None,
                                            dropout_rate=self.dropout_rate,
                                            use_time_distributed=False,
                                            additional_context=False,
                                            return_gate=False,
                                            device=self.device)

        self.h_vec_layer = TFTGRNLayer(hidden_layer_size=self.hidden_layer_size,
                                       output_size=None,
                                       dropout_rate=self.dropout_rate,
                                       use_time_distributed=False,
                                       additional_context=False,
                                       return_gate=False,
                                       device=self.device)

        self.c_vec_layer = TFTGRNLayer(hidden_layer_size=self.hidden_layer_size,
                                       output_size=None,
                                       dropout_rate=self.dropout_rate,
                                       use_time_distributed=False,
                                       additional_context=False,
                                       return_gate=False,
                                       device=self.device)

    def forward(self, inputs):
        # inputs: static_vec => [B, hidden_layer_size]
        stat_vec = self.stat_vec_layer(inputs)
        enrich_vec = self.enrich_vec_layer(inputs)
        h_vec = self.h_vec_layer(inputs)
        c_vec = self.c_vec_layer(inputs)

        return stat_vec, enrich_vec, h_vec, c_vec


##################################################
# 7) Variable Selection: Temporal
##################################################
class VariableSelectionTemporal(nn.Module):
    """
    Equivalent to 'variable_selection_temporal'.
    Takes inputs as a list of [B, T, dim] plus a context vector if `static_context=True`.
    """

    def __init__(self, hidden_layer_size, static_context, dropout_rate, device):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.static_context = static_context
        self.dropout_rate = dropout_rate
        # We'll build on first forward pass
        self.num_vars = None
        self.grn_flat = None
        self.grn_vars = nn.ModuleList()
        self.device = device

    def build_layers(self, num_vars):
        self.num_vars = num_vars
        self.grn_flat = TFTGRNLayer(
            hidden_layer_size=self.hidden_layer_size,
            output_size=self.num_vars,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            additional_context=self.static_context,
            return_gate=False,
            device=self.device
        ).to(self.device)

        for _ in range(num_vars):
            self.grn_vars.append(
                TFTGRNLayer(hidden_layer_size=self.hidden_layer_size,
                            output_size=None,
                            dropout_rate=self.dropout_rate,
                            use_time_distributed=True,
                            additional_context=False,
                            return_gate=False,
                            device=self.device).to(self.device)
            )

    def forward(self, x):
        """
        x:
         - if static_context=True, (list_of_tensors, context)
         - else, list_of_tensors
        list_of_tensors: each shape [B, T, dim_i]
        context: shape [B, context_dim]
        """
        if self.static_context:
            inputs, context = x
            if self.grn_flat is None:
                self.build_layers(len(inputs))

            # Expand context => [B, 1, context_dim]
            context = context.unsqueeze(1)
            flatten = torch.cat(inputs, dim=-1)  # [B, T, sum_of_dims]
            mlp_outputs = self.grn_flat((flatten, context))
        else:
            inputs = x
            if self.grn_flat is None:
                self.build_layers(len(inputs))
            flatten = torch.cat(inputs, dim=-1)  # [B, T, sum_of_dims]
            mlp_outputs = self.grn_flat(flatten)

        # TimeDistributed(Activation('softmax')) => apply softmax over last dim => num_vars
        dynamic_weights = F.softmax(mlp_outputs, dim=-1)  # [B, T, num_vars]
        weights = dynamic_weights.unsqueeze(-1)  # [B, T, num_vars, 1]

        # Transform each variable
        transformed = []
        for i in range(self.num_vars):
            e = self.grn_vars[i](inputs[i])  # [B, T, hidden_size]
            transformed.append(e)

        # [B, T, num_vars, hidden_size]
        trans_embedding = torch.stack(transformed, dim=2)

        # Weighted sum across num_vars
        combined = weights * trans_embedding
        lstm_input = combined.sum(dim=2)  # [B, T, hidden_size]

        return lstm_input, dynamic_weights


##################################################
# 8) LSTM Layer
##################################################
class LSTMLayer(nn.Module):
    """
    Equivalent to 'lstm_layer' in your code.
    """

    def __init__(self, hidden_layer_size, device, rnn_layers, dropout_rate):
        super().__init__()
        self.rnn_layers = rnn_layers
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.device = device

        # Single-layer LSTM for encoder & decoder.
        # If you want multiple layers, you can expand to 'num_layers=rnn_layers'.
        self.tft_encoder = nn.LSTM(input_size=hidden_layer_size,
                                   hidden_size=hidden_layer_size,
                                   num_layers=rnn_layers,
                                   batch_first=True)
        self.tft_decoder = nn.LSTM(input_size=hidden_layer_size,
                                   hidden_size=hidden_layer_size,
                                   num_layers=rnn_layers,
                                   batch_first=True)

        self.gate = TFTApplyGatingLayer(hidden_layer_size=self.hidden_layer_size,
                                        dropout_rate=self.dropout_rate,
                                        use_time_distributed=True,
                                        activation=None,
                                        device=self.device)

        self.add_norm = TFTAddAndNormLayer(device=self.device)

    def forward(self, inputs):
        """
        inputs: (encoder_in, decoder_in, init_states)
          encoder_in: [B, T_enc, hidden_layer_size]
          decoder_in: [B, T_dec, hidden_layer_size]
          init_states: tuple (h0, c0) each shape [B, hidden_layer_size]
                       (for a single-layer LSTM).
        """
        encoder_in, decoder_in, init_states = inputs
        # PyTorch LSTM expects states in shape (num_layers, B, hidden_size).
        # So we must expand dims: [B, hidden_size] -> [1, B, hidden_size]
        h0, c0 = init_states
        h0 = h0.unsqueeze(0).repeat(self.rnn_layers, 1, 1)
        c0 = c0.unsqueeze(0).repeat(self.rnn_layers, 1, 1)

        # Run encoder
        encoder_out, (enc_h, enc_c) = self.tft_encoder(encoder_in, (h0, c0))
        # Run decoder, using final encoder states
        decoder_out, _ = self.tft_decoder(decoder_in, (enc_h, enc_c))

        # Concat => [B, T_enc+T_dec, hidden_layer_size]
        lstm_out = torch.cat([encoder_out, decoder_out], dim=1)
        # Residual skip input
        lstm_in = torch.cat([encoder_in, decoder_in], dim=1)

        # Apply gating
        # gating expects shape [B, T, hidden_size]; we do time-distributed gating
        out, _ = self.gate(lstm_out)

        # Add & norm
        temporal_features = self.add_norm([out, lstm_in])
        return temporal_features

##################################################
# 8.1) LSTM Layer for DeepAR
##################################################
class LSTMLayerDeepAR(nn.Module):
    def __init__(self, hidden_layer_size, device, rnn_layers, dropout_rate):
        super().__init__()
        self.rnn_layers = rnn_layers
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.device = device

        # Single-layer LSTM for encoder & decoder.
        # If you want multiple layers, you can expand to 'num_layers=rnn_layers'.
        self.tft_encoder = nn.LSTM(input_size=hidden_layer_size,
                                   hidden_size=hidden_layer_size,
                                   num_layers=rnn_layers,
                                   batch_first=True)

        self.gate = TFTApplyGatingLayer(hidden_layer_size=self.hidden_layer_size,
                                        dropout_rate=self.dropout_rate,
                                        use_time_distributed=True,
                                        activation=None,
                                        device=self.device)

        self.add_norm = TFTAddAndNormLayer(device=self.device)

    def forward(self, inputs):
        """
        inputs: (encoder_in, decoder_in, init_states)
          encoder_in: [B, T_enc, hidden_layer_size]
          decoder_in: [B, T_dec, hidden_layer_size]
          init_states: tuple (h0, c0) each shape [B, hidden_layer_size]
                       (for a single-layer LSTM).
        """
        encoder_in, init_states = inputs
        # PyTorch LSTM expects states in shape (num_layers, B, hidden_size).
        # So we must expand dims: [B, hidden_size] -> [1, B, hidden_size]
        h0, c0 = init_states
        h0 = h0.unsqueeze(0).repeat(self.rnn_layers, 1, 1)
        c0 = c0.unsqueeze(0).repeat(self.rnn_layers, 1, 1)

        # Run encoder
        encoder_out, (enc_h, enc_c) = self.tft_encoder(encoder_in, (h0, c0))

        # Apply gating
        # gating expects shape [B, T, hidden_size]; we do time-distributed gating
        out, _ = self.gate(encoder_out)

        # Add & norm
        temporal_features = self.add_norm([out, encoder_in])
        return temporal_features, (enc_h, enc_c)


##################################################
# 8.2) CNN Equivalent of LSTMLayer
##################################################
class CNNLayer(nn.Module):
    """
    Replaces the LSTM encoder/decoder with simple CNN blocks.
    """

    def __init__(self, hidden_layer_size, device, rnn_layers, dropout_rate):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.device = device

        # Example CNN "encoder" block
        # We do a single 1D convolution with kernel_size=3 (you can adjust as needed),
        # followed by ReLU and optional dropout.
        self.tft_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_layer_size,  # was input_size for LSTM
                out_channels=hidden_layer_size,  # was hidden_size
                kernel_size=3,
                padding=1  # keep same length after convolution
            ),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate)
        )

        # Example CNN "decoder" block
        self.tft_decoder = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_layer_size,
                out_channels=hidden_layer_size,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate)
        )

        # Gating & add-norm layers remain the same
        self.gate = TFTApplyGatingLayer(
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            activation=None,
            device=self.device
        )
        self.add_norm = TFTAddAndNormLayer(device=self.device)

    def forward(self, inputs):
        """
        inputs: (encoder_in, decoder_in, init_states)
          encoder_in: [B, T_enc, hidden_layer_size]
          decoder_in: [B, T_dec, hidden_layer_size]
          init_states: tuple (h0, c0), each [B, hidden_layer_size] (unused by CNN)
        """
        encoder_in, decoder_in, init_states = inputs

        # CNN does not use init_states (h0, c0).
        # We'll ignore them or you can remove them entirely.

        # Permute to [B, C, T] for 1D convolution:
        #   - 'C' corresponds to hidden_layer_size
        #   - 'T' corresponds to time dimension
        # e.g. from (B, T_enc, hidden_dim) -> (B, hidden_dim, T_enc)
        encoder_in_cnn = encoder_in.permute(0, 2, 1)
        decoder_in_cnn = decoder_in.permute(0, 2, 1)

        # Pass through encoder CNN
        encoder_out_cnn = self.tft_encoder(encoder_in_cnn)
        # Pass through decoder CNN
        decoder_out_cnn = self.tft_decoder(decoder_in_cnn)

        # Permute back to [B, T, C]
        encoder_out = encoder_out_cnn.permute(0, 2, 1)  # [B, T_enc, hidden_dim]
        decoder_out = decoder_out_cnn.permute(0, 2, 1)  # [B, T_dec, hidden_dim]

        # Concat => [B, (T_enc + T_dec), hidden_layer_size]
        cnn_out = torch.cat([encoder_out, decoder_out], dim=1)

        # For skip connection, replicate original input shape
        cnn_in = torch.cat([encoder_in, decoder_in], dim=1)

        # Gating
        out, _ = self.gate(cnn_out)

        # Add & norm
        temporal_features = self.add_norm([out, cnn_in])

        return temporal_features


##################################################
# 9) Static Enrichment Layer
##################################################
class StaticEnrichmentLayer(nn.Module):
    """
    Equivalent to 'static_enrichment_layer'.
    """

    def __init__(self, hidden_layer_size, context, dropout_rate, device):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.context = context
        self.dropout_rate = dropout_rate
        self.device = device

        self.grn_enrich = TFTGRNLayer(hidden_layer_size=self.hidden_layer_size,
                                      output_size=None,
                                      dropout_rate=self.dropout_rate,
                                      use_time_distributed=True,
                                      additional_context=self.context,
                                      return_gate=False,
                                      device=self.device)

    def forward(self, inputs):
        """
        inputs:
          if self.context=True, (temporal_features, static_enrichment_vec)
          else, just temporal_features
        shapes:
          temporal_features => [B, T, hidden_size]
          static_enrichment_vec => [B, context_dim]
        """
        if self.context:
            x, c = inputs
            # expand c => [B, 1, context_dim]
            c = c.unsqueeze(1)
            enriched = self.grn_enrich((x, c))
        else:
            x = inputs
            enriched = self.grn_enrich(x)

        return enriched

########################################################
# 10) Attention Layer & Stack
########################################################

# Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self, hidden_layer_size, device, n_head, dropout_rate):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.hidden_layer_size = hidden_layer_size
        self.device = device
        self.mha = TFTMultiHeadAttention(n_head=n_head, d_model=hidden_layer_size, dropout_rate=dropout_rate, device=self.device)
        self.gate = TFTApplyGatingLayer(hidden_layer_size=hidden_layer_size,
                                        dropout_rate=dropout_rate,
                                        use_time_distributed=True,
                                        activation=None,
                                        device=self.device)
        self.add_norm = TFTAddAndNormLayer(device=self.device)

    def forward(self, x, attn_mask, padding_mask):

        attn_out, _ = self.mha(x, x, x, attn_mask, padding_mask) # (q,k,v,mask,training)

        # gating layer
        attn_out, _ = self.gate(attn_out)
        # add_norm
        attn_out = self.add_norm([attn_out, x])
        return attn_out


# Attention Stack
class AttentionStack(nn.Module):
    def __init__(self, num_layers, hidden_layer_size, n_head, dropout_rate, device):
        super().__init__()

        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        self.attn_layers = [AttentionLayer(hidden_layer_size, device, n_head, dropout_rate) for _ in range(num_layers)]
        self.device = device
        self.grn_final = TFTGRNLayer(hidden_layer_size=hidden_layer_size,
                                     output_size=None,
                                     dropout_rate=dropout_rate,
                                     use_time_distributed=True,
                                     additional_context=False,
                                     return_gate=False,
                                     device=self.device)

    def forward(self, x, attn_mask, padding_mask):

        attn_out = x
        for i in range(self.num_layers):
            attn_out = self.attn_layers[i](attn_out, attn_mask, padding_mask)

        # final GRN layer
        attn_out = self.grn_final(attn_out)
        return attn_out

######################################################
# 11) Final Gating Layer
######################################################

# Final Gating Layer
class FinalGatingLayer(nn.Module):
    def __init__(self, hidden_layer_size, device, dropout_rate):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.device = device
        self.gate = TFTApplyGatingLayer(hidden_layer_size=hidden_layer_size,
                                        dropout_rate=dropout_rate,
                                        use_time_distributed=True,
                                        activation=None,
                                        device=self.device)
        self.add_norm = TFTAddAndNormLayer(device=self.device)

    def forward(self, inputs):

        attn_out, temporal_features = inputs

        # final gating layer
        attn_out, _ = self.gate(attn_out)

        # final add & norm
        out = self.add_norm([attn_out, temporal_features])

        return out


######################################################
# 12) Native Pytorch MHA Stack implementation (reference implementation) -- Not In Use
######################################################

# Pytorch Native MHAttention Stack
def generate_square_subsequent_mask(sz, device = None, dtype = None):
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )

class MHA(nn.Module):
    def __init__(self, num_layers, hidden_layer_size, n_head, dropout_rate, device):
        super().__init__()

        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        self.attn_layers = [nn.MultiheadAttention(embed_dim=hidden_layer_size, num_heads=n_head, dropout=dropout_rate, bias=True, batch_first=True).to(device) for _ in range(num_layers)]
        self.device = device
        self.grn_final = TFTGRNLayer(hidden_layer_size=hidden_layer_size,
                                     output_size=None,
                                     dropout_rate=dropout_rate,
                                     use_time_distributed=True,
                                     additional_context=False,
                                     return_gate=False,
                                     device=self.device)

    def forward(self, x):
        seq_len = x.size(1)
        attn_mask = generate_square_subsequent_mask(sz=seq_len, device=self.device)
        attn_out = x
        for i in range(self.num_layers):
            attn_out, _ = self.attn_layers[i](query=attn_out, key=attn_out, value=attn_out, attn_mask=attn_mask, need_weights=False, is_causal=True)

        # final GRN layer
        attn_out = self.grn_final(attn_out)
        return attn_out
