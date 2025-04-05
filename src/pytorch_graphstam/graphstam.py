import os
import pandas as pd
import numpy as np
import copy
from .graphbuilder.graphtft import graphmodel as GraphTFT
from .graphbuilder.graphrecursive import graphmodel as SimpleGraph
from .graphbuilder.graphseq2seq import graphmodel as GraphSeq2Seq
from .configs.sample_config import print_sample_config
from .configs.logger import Logger
from datetime import datetime as dt

def setup_logger(project_dir) :
    logger = Logger(f"{project_dir}/logs/graphstam_{dt.now():%Y-%m-%d}.log").get_logger()

    return logger

def show_config_template():
    print_sample_config()

class gml(object):
    def __init__(self, config):
        self.config = config
        self.model_type = self.config["model_type"]
        self.project_dir = self.config["working_dir"]
        self.col_dict = self.config["features_config"]
        self.rolling_features = self.config.get("rolling_features", None)
        self.data_config = self.config["data_config"]
        self.model_config = self.config["model_config"]
        self.train_config = self.config["train_config"]
        self.infer_config = self.config["infer_config"]

        # create working dir if it doesn't exist:
        try:
            if not os.path.exists(self.project_dir):
                os.makedirs(self.project_dir)
            # also make models & logs dir under project_dir
            if not os.path.exists(f"{self.project_dir}/models"):
                os.makedirs(f"{self.project_dir}/models")
            if not os.path.exists(f"{self.project_dir}/logs"):
                os.makedirs(f"{self.project_dir}/logs")
        except OSError as e:
            print(f"Error creating working dir: {e}")
            return

        # setup logger
        logger = setup_logger(self.project_dir)
        logger.info("Starting execution configuration: %s", self.config)

        # backup original feature set
        self.baseline_col_dict = copy.deepcopy(self.col_dict)

        # init gmlobj to None
        self.gmlobj = None

        # extract arguments for GraphTFT from config
        max_lags = self.data_config["max_lags"]
        max_covar_lags = self.data_config.get("max_covar_lags", max_lags)
        max_leads = self.data_config["fh"]
        fh = self.data_config["fh"]
        pad_data = self.data_config.get("pad_data", True)
        autoregressive_target = self.data_config.get("autoregressive_target", True)
        train_till = self.data_config["train_till"]
        test_till = self.data_config["test_till"]
        scaling_method = self.data_config.get("scaling_method", "mean_scaling")
        tweedie_out = self.data_config.get("tweedie_out", False)
        tweedie_variance_power = self.data_config.get("tweedie_variance_power", [1.1])
        interleave = self.data_config.get("interleave", 1)
        recency_weights = self.data_config.get("recency_weights", False)
        recency_alpha = self.data_config.get("recency_alpha", 1)
        hierarchical_weights = self.data_config.get("hierarchical_weights", True)
        data_parallel_processes = int(os.cpu_count() / 1.5)
        pin_graph_in_memory = self.data_config.get("pin_graph_in_memory", True)

        # create graphmodel object
        if self.model_type == 'GraphTFT':

            self.gmlobj = GraphTFT(col_dict = self.col_dict,
                                   max_target_lags = max_lags,
                                   max_covar_lags = max_lags,
                                   max_leads = max_leads,
                                   train_till = train_till,
                                   test_till = test_till,
                                   autoregressive_target = autoregressive_target,
                                   lag_offset = 0,
                                   rolling_features_list = self.rolling_features,
                                   min_history = 1,
                                   fh = fh,
                                   pad_data = pad_data,
                                   recursive = False,
                                   batch_size = 1000,               # UNUSED
                                   subgraph_sampling = False,       # UNUSED
                                   samples_per_period = 10,         # UNUSED
                                   grad_accum = False,              # UNUSED
                                   accum_iter = 1,                  # UNUSED
                                   scaling_method = scaling_method,
                                   outlier_threshold = 1e7,         # UNUSED
                                   log1p_transform = False,         # UNUSED
                                   tweedie_out = tweedie_out,
                                   estimate_tweedie_p = False,      # UNUSED
                                   tweedie_p_range = [1.01, 1.99],
                                   tweedie_variance_power = tweedie_variance_power,
                                   iqr_high = 0.75,                 # UNUSED
                                   iqr_low = 0.25,                  # UNUSED
                                   categorical_onehot_encoding = True,
                                   directed_graph = True,
                                   shuffle = True,
                                   interleave = interleave,
                                   hierarchical_weights = hierarchical_weights,
                                   recency_weights = recency_weights,
                                   recency_alpha = recency_alpha,
                                   output_clipping = False,         # UNUSED
                                   pin_graph_in_memory = pin_graph_in_memory,
                                   PARALLEL_DATA_JOBS = data_parallel_processes,
                                   PARALLEL_DATA_JOBS_BATCHSIZE = 128)

        elif self.model_type == 'SimpleGraph':

            self.gmlobj = SimpleGraph(col_dict = self.col_dict,
                                      max_target_lags = max_lags,
                                      max_covar_lags = max_covar_lags,
                                      max_leads = max_leads,
                                      train_till = train_till,
                                      test_till = test_till,
                                      autoregressive_target = autoregressive_target,
                                      lag_offset = 0,
                                      rolling_features_list = self.rolling_features,
                                      min_history = 1,
                                      fh = 1,
                                      pad_data=pad_data,
                                      recursive = True,
                                      batch_size = 1000,
                                      subgraph_sampling = False,
                                      samples_per_period = 10,
                                      grad_accum = False,
                                      accum_iter = 1,
                                      scaling_method = scaling_method,
                                      outlier_threshold = 1e7,
                                      log1p_transform = False,
                                      tweedie_out = tweedie_out,
                                      estimate_tweedie_p = False,
                                      tweedie_p_range = [1.01, 1.99],
                                      tweedie_variance_power = tweedie_variance_power,
                                      iqr_high = 0.75,
                                      iqr_low = 0.25,
                                      categorical_onehot_encoding = True,
                                      directed_graph = True,
                                      shuffle = True,
                                      interleave = interleave,
                                      hierarchical_weights = hierarchical_weights,
                                      recency_weights = recency_weights,
                                      recency_alpha = recency_alpha,
                                      output_clipping = False,
                                      pin_graph_in_memory = pin_graph_in_memory,
                                      PARALLEL_DATA_JOBS = data_parallel_processes,
                                      PARALLEL_DATA_JOBS_BATCHSIZE = 128)

        elif self.model_type == 'GraphSeq2Seq':

            self.gmlobj = GraphSeq2Seq(col_dict = self.col_dict,
                                      max_target_lags = max_lags,
                                      max_covar_lags = max_covar_lags,
                                      max_leads = max_leads,
                                      train_till = train_till,
                                      test_till = test_till,
                                      autoregressive_target = autoregressive_target,
                                      lag_offset = 0,
                                      rolling_features_list = self.rolling_features,
                                      min_history = 1,
                                      fh = fh,
                                      pad_data=pad_data,
                                      recursive = False,
                                      batch_size = 1000,
                                      subgraph_sampling = False,
                                      samples_per_period = 10,
                                      grad_accum = False,
                                      accum_iter = 1,
                                      scaling_method = scaling_method,
                                      outlier_threshold = 1e7,
                                      log1p_transform = False,
                                      tweedie_out = tweedie_out,
                                      estimate_tweedie_p = False,
                                      tweedie_p_range = [1.01, 1.99],
                                      tweedie_variance_power = tweedie_variance_power,
                                      iqr_high = 0.75,
                                      iqr_low = 0.25,
                                      categorical_onehot_encoding = True,
                                      directed_graph = True,
                                      shuffle = True,
                                      interleave = interleave,
                                      hierarchical_weights = hierarchical_weights,
                                      recency_weights = recency_weights,
                                      recency_alpha = recency_alpha,
                                      output_clipping = False,
                                      pin_graph_in_memory = pin_graph_in_memory,
                                      PARALLEL_DATA_JOBS = data_parallel_processes,
                                      PARALLEL_DATA_JOBS_BATCHSIZE = 128)


    def build(self, data):
        """
        1. Builds Graph Dataset Iterators for Training & Testing
        2. Builds & initializes the model using parameters configured in model_config

        :param data: preprocessed dataframe
        """
        # build datasets for train/test
        self.gmlobj.build_dataset(data)
        # build model
        # extract build arguments
        model_dim = self.model_config.get("model_dim", 64)
        feature_dim = self.model_config.get("feature_dim", model_dim)
        num_gnn_layers = self.model_config.get("num_gnn_layers", 1)
        num_attn_layers = self.model_config.get("num_attn_layers", 1)
        num_rnn_layers = self.model_config.get("num_rnn_layers", 1)
        attn_heads = self.model_config.get("heads", 1)
        forecast_quantiles = self.model_config.get("forecast_quantiles", [0.5])
        dropout = self.model_config.get("dropout", 0)
        chunk_size = self.model_config.get("chunk_size", None)
        downsample_factor = self.model_config.get("downsample_factor", 1)
        skip_connection = self.model_config.get("skip_connection", False)
        feature_transform = self.model_config.get("feature_transform", True)
        device = self.model_config.get("device", 'cuda')
        gbt = self.model_config.get("gbt", False)
        n_boosters = self.model_config.get("n_boosters", 0)
        batched_train = self.model_config.get("batched_train", False)

        if self.model_type == 'GraphTFT':
            self.gmlobj.build(layer_type='SAGE',
                              model_dim=model_dim,
                              num_layers=num_gnn_layers,
                              num_attn_layers=num_attn_layers,
                              num_rnn_layers=num_rnn_layers,
                              heads=attn_heads,
                              forecast_quantiles=forecast_quantiles,
                              dropout=dropout,
                              chunk_size=chunk_size,
                              downsample_factor=downsample_factor,
                              device=device,
                              gbt=gbt,
                              n_boosters=n_boosters,
                              batched_train=batched_train)

        elif self.model_type == 'SimpleGraph':
            self.gmlobj.build(layer_type='SAGE',
                              feature_dim=feature_dim,
                              model_dim=model_dim,
                              num_layers=num_gnn_layers,
                              num_rnn_layers=num_rnn_layers,
                              forecast_quantiles=forecast_quantiles,
                              dropout=dropout,
                              chunk_size=chunk_size,
                              skip_connection=skip_connection,
                              feature_transform=feature_transform,
                              device=device,
                              gbt=gbt,
                              n_boosters=n_boosters)

        elif self.model_type == 'GraphSeq2Seq':
            self.gmlobj.build(layer_type='SAGE',
                              feature_dim=feature_dim,
                              model_dim=model_dim,
                              num_layers=num_gnn_layers,
                              num_rnn_layers=num_rnn_layers,
                              forecast_quantiles=forecast_quantiles,
                              dropout=dropout,
                              chunk_size=chunk_size,
                              skip_connection=skip_connection,
                              feature_transform=feature_transform,
                              device=device,
                              gbt=gbt,
                              n_boosters=n_boosters)


    def train(self):
        """
        Trains model using configuration provided in train_config.
        """
        lr = self.train_config.get("lr", 0.001)
        default_scheduler_params = {'factor': 0.5,
                                    'patience': 5,
                                    'threshold': 0.0001,
                                    'min_lr': 0.0001,
                                    'clip_gradients': False,
                                    'max_norm': 2.0,
                                    'norm_type': 2}

        min_epochs = self.train_config.get("min_epochs", 10)
        max_epochs = self.train_config.get("max_epochs", 100)
        patience = self.train_config.get("patience", 10)
        min_delta = self.train_config.get("min_delta", 0)
        model_prefix = self.train_config.get("model_prefix", f"{self.model_type}")
        loss = self.train_config.get("loss_type", 'Quantile')
        use_normalized_multi_loss = self.train_config.get("use_normalized_multi_loss", False)
        beta = self.train_config.get("beta", 0.9)
        ema_iterations = self.train_config.get("ema_iterations", 10)
        delta = self.train_config.get("huber_delta", 0.5)
        use_amp = self.train_config.get("use_amp", False)
        use_lr_scheduler = self.train_config.get("use_lr_scheduler", True)
        scheduler_params = self.train_config.get("scheduler_params", default_scheduler_params)
        sample_weights = self.train_config.get("sample_weights", False)
        stop_training_criteria = self.train_config.get("stop_training_criteria", 'loss')
        # for batched training -- only for TFT
        batched_train = self.model_config.get("batched_train", False)
        max_batch_size = self.train_config.get("max_batch_size", None)

        if self.model_type == 'GraphTFT' and batched_train:
            self.gmlobj.batched_train(lr=lr,
                                      min_epochs=min_epochs,
                                      max_epochs=max_epochs,
                                      patience=patience,
                                      min_delta=min_delta,
                                      model_prefix=f"{self.project_dir}/models/{model_prefix}",
                                      max_batch_size=max_batch_size,
                                      loss_type=loss,
                                      use_normalized_multi_loss=use_normalized_multi_loss,
                                      beta=beta,
                                      ema_iterations=ema_iterations,
                                      delta=delta,
                                      use_amp=use_amp,
                                      use_lr_scheduler=use_lr_scheduler,
                                      scheduler_params=scheduler_params,
                                      sample_weights=sample_weights,
                                      stop_training_criteria=stop_training_criteria)

        elif (self.model_type == 'GraphTFT') or (self.model_type == 'GraphSeq2Seq'):
            self.gmlobj.train(lr=lr,
                              min_epochs=min_epochs,
                              max_epochs=max_epochs,
                              patience=patience,
                              min_delta=min_delta,
                              model_prefix=f"{self.project_dir}/models/{model_prefix}",
                              loss_type=loss,
                              use_normalized_multi_loss=use_normalized_multi_loss,
                              beta=beta,
                              ema_iterations=ema_iterations,
                              delta=delta,
                              use_amp=use_amp,
                              use_lr_scheduler=use_lr_scheduler,
                              scheduler_params=scheduler_params,
                              sample_weights=sample_weights,
                              stop_training_criteria=stop_training_criteria)

        else:
            self.gmlobj.train(lr=lr,
                              min_epochs=min_epochs,
                              max_epochs=max_epochs,
                              patience=patience,
                              min_delta=min_delta,
                              model_prefix=f"{self.project_dir}/models/{model_prefix}",
                              loss_type=loss,
                              delta=delta,
                              use_amp=use_amp,
                              use_lr_scheduler=use_lr_scheduler,
                              scheduler_params=scheduler_params,
                              sample_weights=sample_weights,
                              stop_training_criteria=stop_training_criteria)

    def infer(self, forecast_filepath, forecast_lower_bound=0, forecast_upper_bound=np.inf):
        """
        Loads the best saved model from 'train' run & performs inference based on infer_config
        :return: dataframe with forecasts
        """
        infer_start = self.infer_config["infer_start"]
        infer_end = self.infer_config["infer_end"]
        quantiles_list =  self.infer_config.get("quantiles", [0.5])
        forecast_filepath = self.infer_config.get("forecast_filepath", forecast_filepath)

        forecast_list = []
        for i, q in enumerate(quantiles_list):
            forecast_df = self.gmlobj.infer(infer_start=infer_start, infer_end=infer_end, select_quantile=q)
            # clip forecasts
            forecast_df['forecast'] = np.clip(forecast_df['forecast'], a_min=forecast_lower_bound, a_max=forecast_upper_bound)
            if i > 0:
                forecast_df.rename(columns={'forecast': f'forecast_{q}'}, inplace=True)
            forecast_list.append(forecast_df)

        # merge into a single forecast file
        total_forecast = pd.DataFrame()
        for i, (q, f_df) in enumerate(zip(quantiles_list, forecast_list)):
            if i == 0:
                total_forecast = pd.concat([total_forecast, f_df], axis=1)
            else:
                total_forecast = pd.concat([total_forecast, f_df[[f'forecast_{q}']]], axis=1)

        total_forecast.to_csv(forecast_filepath, index=False)

        return total_forecast





















