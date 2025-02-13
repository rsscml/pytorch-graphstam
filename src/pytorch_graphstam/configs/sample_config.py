SAMPLE_CONFIG = """\
# Define the following structures

# features dictionary
# id_col: Column containing unique ids of time series in the dataframe
# key_combinations: List of tuples where each tuple represents a level in the hierarchy. If needed the tuple can have a single element.
# lowest_key_combination: Tuple, from "key_combinations", corresponding to the lowest key level in the hierarchy
# highest_key_combination: Tuple, from "key_combinations", corresponding to the lowest key level in the hierarchy
# target_col: Column containing the target variable to predict
# time_index_col: Column containing datetime information; the datatype can be datetime, string or integer.
# global_context_col_list: List of columns containing high-level static variables. Specify empty list if inapplicable.
# static_cat_col_list: List of columns containing low-level static variables. Specify empty list if inapplicable.
# temporal_known_num_col_list: List of time varying numeric features with known future values
# temporal_unknown_num_col_list: List of time varying numeric features with unknown future values. Specify empty list if inapplicable.
# wt_col: Weight column representing relative weight to assign to a unique id. Hierarchical weights will be partly derived from this column. Specify None if inapplicable.

features_config = {
                    "id_col": str,
                    "key_combinations": list,       
                    "lowest_key_combination": tuple,        
                    "highest_key_combination": tuple,       
                    "target_col": str,      
                    "time_index_col": str,      
                    "global_context_col_list": list,   
                    "static_cat_col_list": list,       
                    "temporal_known_num_col_list": list,  
                    "temporal_unknown_num_col_list": list,  
                    "wt_col": str
}

# Rolling statistics
# Provided as list of tuples (e.g., [('time_index_col', 'mean', 7, 0), ...]) where each tuple contains upto 5 values corresponding to : ('time_index_col', statistic, rolling_window_size, window_offset, quantile)
# Optional, an empty list can be provided
# Supported statistics: 'mean','std','quantile'. If using 'quantile' statistic, the 5th argument in the range [0, 1] corresponding to quantile must be provided. For other statistics, the first 4 arguments must be provided.
# "window_offset" allows for shifted rolling windows. If not required, keep this as 0. 

rolling_features = list of tuples


# data config
# max_lags: max. no. of lags to consider as part of context
# fh: forecast horizon specified as no. of periods to forecast into the future
# train_till: time_index_col value to use as cutoff for training set
# test_till: time_index_col value to use as cutoff for testing set. Ensure that test_till >= train_till + fh
# scaling_method: method used to scale numeric features. 'mean_scaling' is the default. Specify 'no_scaling' if inapplicable.
# tweedie_out: Applicable only if training using tweedie loss function. False by default.
# tweedie_variance_power: If tweedie_out == True, specify a list of one or more values for tweedie variance power in the range (1.0, 2.0)
# interleave: Used to reduce number of samples used in training when dealing with very long time-series. A value of 1 (default) suggests no interleaving i.e. missed samples. 
# hierarchical_weights: If True, apply hierarchical weights to training loss. Set it to True for hierarchical forecasting scenarios.
# recency_weights: If True, apply more weight to more recent training samples.
# recency_alpha: If recency_weights == True, set to an integer value >= 1. Greater values imply greater weights for recent samples.
# 

data_config = {
                "max_lags": int,
                "max_covar_lags": int,        
                "fh": int,
                "train_till": str,
                "test_till": str,
                "scaling_method": str,
                "tweedie_out": bool,
                "tweedie_variance_power": list,   # list of floats (e.g., [1.1])
                "interleave": int,
                "recency_weights": bool,
                "recency_alpha": float
}

# model config -- defines hyperparameters for model construction
# layer_type: Defaults to 'SAGE', currently the only provided option
# model_dim: Model's hidden dimension. All the layers in the model are derived from this. Typical choices are: 32, 64, 128 etc.
# num_layers: No. of GNN Layers to use. 1 (default) is fine for most use cases.
# num_attn_layers: No. of masked self-attention blocks to use. Values in the range 1 (default) to 4 work well in most cases.
# num_rnn_layers: No. of rnn (LSTM) layers to use. If unsure, use a value of 1 (default).
# heads: No. of attention heads to use in the self-attention block. This integer should be a factor of model_dim. default: 1.
# forecast_quantiles: A list of various quantiles if learning a distribution

model_config = {
                "model_dim": int (default = 64),
                "num_gnn_layers": int (default = 1),
                "num_attn_layers": int (default = 1),
                "num_rnn_layers": int (default = 1),
                "heads": int (default = 1),
                "forecast_quantiles": list (default = [0.5]),
                "dropout": float (default = 0),
                "chunk_size": (int, type(None)) (default = None),
                "skip_connection": bool (default = False),
                "device": str (default = 'cuda')
    }
    
    
# train config -- defines hyperparameters for model training & checkpointing
# lr: Learning rate
# min_epochs: Minimum epochs to train the model for regardless of change in loss.
# max_epochs: Maximum epochs to train the model for.
# patience: Number of epochs to continue training for when loss is not decreasing. After these many epochs, if epochs > min_epochs, the training stops.
# model_prefix: Used to derive the filepath for saving the model.
# loss_type: The metric to minimize.
# huber_delta: Applicable only if loss_type is 'Huber'. The hyperparameter for Huber loss fn.
# use_amp: If True, will cast all float32 values to float16 to save cpu memory/gpu VRAM while achieving speedup at the expense of accuracy; the impact on accuracy in most cases should be negligible but always compare!
# use_lr_scheduler: If True, uses a linear scheduler for learning rate. If the loss doesn't decrease over epochs, learning rate is reduced as per the parameters defined in scheduler_params_config
# sample_weights: If True, weights the samples to compute the final loss before performing gradient computations & updates.
# stop_training_criteria: If set to 'loss', will use the loss value as basis to identify & save the best model; this can optionally be set to 'mse' or 'mae'. 
    
train_config = {
                "lr": float (default = 0.001),
                "min_epochs": int (default = 10),
                "max_epochs": int (default = 100),
                "patience": int (default = 5),
                "model_prefix": str (default = f"./{model_type}"),
                "loss_type": str (default = 'Quantile', Options = ['Quantile', 'Huber', 'Tweedie', 'RMSE']),
                "huber_delta": float (default = 0.5),
                "use_amp": bool (default = False),
                "use_lr_scheduler": bool (default = True),
                "scheduler_params": dict (default = see scheduler_params_config),
                "sample_weights": bool (default = False),
                "stop_training_criteria": str (default = 'loss', Options = ['loss', 'mae', 'mse'])
}    
               
scheduler_params_config = {
                            "factor": float (default = 0.5), 
                            "patience": int (default = 5), 
                            "threshold": float (default = 0.0001), 
                            "min_lr": float (default = 0.0001), 
                            "clip_gradients": bool (default = False), 
                            "max_norm": float (default = 2.0), 
                            "norm_type": int (default = 2, Options = [1, 2])
} 


# infer config
# infer_start: Str or int datetime/timeindex value from which to start generating forecast.
# infer_end: Forecast generation stops after this period
# quantiles: Forecast Quantiles to output; can be omitted if Quantiles don't apply, for e.g. with non-Quantile loss types.
# forecast_filepath: Forecasts are written here

infer_config = {
                infer_start: str,int,
                infer_end: str,int,
                quantiles: list of float,
                forecast_filepath: str
}

# model_type
# Select one of ['GraphTFT','SimpleGraph','GraphSeq2Seq']

# Final Config

config = {
            "model_type": 'GraphTFT',
            "features_config": features_config,
            "rolling_features": rolling_features,
            "data_config": data_config,
            "model_config": model_config,
            "train_config": train_config,
            "infer_config": infer_config
}
              
"""

def print_sample_config():
    print(SAMPLE_CONFIG)
