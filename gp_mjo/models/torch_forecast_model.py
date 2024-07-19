import warnings
warnings.filterwarnings("ignore")

import torch

import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import EarlyStopping

from pytorch_forecasting import TimeSeriesDataSet, DeepAR, NHiTS, TemporalFusionTransformer, RecurrentNetwork
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import MultivariateNormalDistributionLoss, MQF2DistributionLoss, QuantileLoss




class TorchForecastModel:
    def __init__(self, df, model_name='DeepAR', width=40, lead_time=60, 
                n=10000, start_train=0, n_offset=0,
                v=2500, batch_size=128, seed=99, ) -> None:
        
        self.df = df
        self.model_name = model_name
        self.width = width
        self.lead_time = lead_time
        self.n = n
        self.start_train = start_train
        self.n_offset = n_offset
        self.v = v
        self.batch_size = batch_size
        self.seed = seed
        
        # create dataset and dataloaders
        training_cutoff = start_train + n + lead_time + width - 2 #df["time_idx"].max() - lead_time

        start_forecast_val = training_cutoff + 1 + n_offset + width # start_train + n + lead_time + n_offset 
        validation_cutoff = start_forecast_val + v + lead_time - 2

        static_categoricals = ['group'] if model_name in ['DeepAR', 'TFT'] else []

        # create the dataset from the pandas dataframe
        training = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff],
            target="value",
            time_idx="time_idx",
            group_ids=["group"],
            categorical_encoders={"group": NaNLabelEncoder().fit(df.group)},
            static_categoricals=static_categoricals,
            min_encoder_length=width,
            max_encoder_length=width,
            min_prediction_length=lead_time,
            max_prediction_length=lead_time,
            time_varying_unknown_reals=["value"],
            target_normalizer=None,
        )


        validation = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= validation_cutoff],
            target="value",
            time_idx="time_idx",
            group_ids=["group"],
            categorical_encoders={"group": NaNLabelEncoder().fit(df.group)},
            static_categoricals=static_categoricals,
            min_encoder_length=width,
            max_encoder_length=width,
            min_prediction_length=lead_time,
            max_prediction_length=lead_time,
            min_prediction_idx = start_forecast_val,
            time_varying_unknown_reals=["value"],
            target_normalizer=None,
        )


        # synchronize samples in each batch over time - only necessary for DeepAR, not for DeepAR
        train_dataloader = training.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0, batch_sampler="synchronized",
            shuffle=True, generator=torch.Generator().manual_seed(seed),
        )
        val_dataloader = validation.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized",
            shuffle=True, generator=torch.Generator().manual_seed(seed),
        )

        self.training = training
        self.validation = validation
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def opt_lr(self, **kwargs):
        self.device = kwargs.get('device', "cpu") #["cpu", "mps", "cuda", "tpu", "hpu"]
        self.hidden_size = kwargs.get('hidden_size', 64)
        self.lr = kwargs.get('lr', 3e-2)
        self.rnn_layers = kwargs.get('rnn_layers', 1)
        self.weight_decay = kwargs.get('weight_decay', 1e-2)
        
        verbose = kwargs.get('verbose', True)
        seed = self.seed
        training = self.training
        train_dataloader = self.train_dataloader
        val_dataloader = self.val_dataloader
        model_name = self.model_name

        pl.seed_everything(seed)
        trainer = pl.Trainer(accelerator=self.device, gradient_clip_val=1e-1)

        if model_name == 'DeepAR':
            net = DeepAR.from_dataset(
                training,
                learning_rate=self.lr,
                hidden_size=self.hidden_size,
                rnn_layers=self.rnn_layers,
                loss=MultivariateNormalDistributionLoss(rank=30),
                optimizer="Adam",
            )
        elif model_name == 'NHiTS':
            net = NHiTS.from_dataset(
                training,
                learning_rate=self.lr,
                weight_decay=self.weight_decay,
                loss=MQF2DistributionLoss(prediction_length=self.lead_time),
                backcast_loss_ratio=0.0,
                hidden_size=self.hidden_size,
                optimizer="AdamW",
            )

        # find optimal learning rate
        res = Tuner(trainer).lr_find(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            min_lr=1e-5,
            max_lr=1e0,
            early_stop_threshold=100,
        )
        
        if verbose:
            print(f"suggested learning rate: {res.suggestion()}")
            fig = res.plot(show=True, suggest=True)
            fig.show()
        net.hparams.learning_rate = res.suggestion()
        
        self.res = res

    def train(self, **kwargs):
        
        model_name = self.model_name
        seed = self.seed
        device = self.device
        res = self.res
        hidden_size = self.hidden_size
        rnn_layers = self.rnn_layers
        weight_decay = self.weight_decay
        lead_time = self.lead_time

        training = self.training
        train_dataloader = self.train_dataloader
        val_dataloader = self.val_dataloader

        max_epochs = kwargs.get('max_epochs', 10)
        gradient_clip_val = kwargs.get('gradient_clip_val', 0.1)
        limit_train_batches = kwargs.get('limit_train_batches', 50)

        if model_name == 'NHiTS':
            max_epochs = 5

        pl.seed_everything(seed)
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=device,
            enable_model_summary=True,
            gradient_clip_val=gradient_clip_val,
            callbacks=[early_stop_callback],
            limit_train_batches=limit_train_batches,
            enable_checkpointing=True,
        )

        if model_name == 'DeepAR':
            net = DeepAR.from_dataset(
                training,
                learning_rate=res.suggestion(),
                log_interval=10,
                log_val_interval=1,
                hidden_size=hidden_size,
                rnn_layers=rnn_layers,
                optimizer="Adam",
                loss=MultivariateNormalDistributionLoss(rank=30),
            )
        elif model_name == 'NHiTS':
            net = NHiTS.from_dataset(
                training,
                learning_rate=res.suggestion(),
                log_interval=10,
                log_val_interval=1,
                weight_decay=weight_decay,
                backcast_loss_ratio=0.0,
                hidden_size=hidden_size,
                optimizer="AdamW",
                loss=MQF2DistributionLoss(prediction_length=lead_time),
            )

        trainer.fit(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        self.max_epochs = max_epochs
        self.net = net
        self.trainer = trainer

    def pred(self, start_test=None,
            n_pred=1, m=5000, **kwargs):
        
        model_name = self.model_name
        df = self.df
        width = self.width
        lead_time = self.lead_time
        start_train = self.start_train
        n = self.n
        v = self.v
        n_offset = self.n_offset
        batch_size = self.batch_size
        seed = self.seed

        trainer = self.trainer
        device = self.device
        net = self.net

        n_samples = kwargs.get('n_samples', 100)
        
        ############################################
        # Create test dataset and test dataloader
        ############################################
        if start_test is None:
            start_forecast_test = start_train + width + n + n_offset + width \
                            + v + n_offset + 10 + width
        else:
            start_forecast_test = start_test + width
        test_cutoff = start_forecast_test + n_pred + lead_time - 2

        test = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= test_cutoff],
            target="value",
            time_idx="time_idx",
            group_ids=["group"],
            categorical_encoders={"group": NaNLabelEncoder().fit(df.group)},
            static_categoricals=[
                "group"
            ],  # as we plan to forecast correlations, it is important to use series characteristics (e.g. a series identifier)
            min_encoder_length=width,
            max_encoder_length=width,
            min_prediction_length=lead_time,
            max_prediction_length=lead_time,
            min_prediction_idx = start_forecast_test,
            time_varying_unknown_reals=["value"],
            target_normalizer=None,
        )

        test_dataloader = test.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized",
            shuffle=True, generator=torch.Generator().manual_seed(seed),
        )

        ############################################
        # predict on test_dataloader
        ############################################
        pl.seed_everything(seed)
        best_model_path = trainer.checkpoint_callback.best_model_path
        if model_name == 'DeepAR':
            best_model = DeepAR.load_from_checkpoint(best_model_path)
        elif model_name == 'NHiTS':
            best_model = NHiTS.load_from_checkpoint(best_model_path)

        # best_model = net
        predictions = best_model.predict(test_dataloader, mode="prediction", 
                                trainer_kwargs=dict(accelerator=device), 
                                return_index=True, return_x=True, return_y=True)
        
        raw_predictions = net.predict(
            test_dataloader, mode="raw", 
            return_index=True, return_x=True, return_y=True, 
            n_samples=n_samples, 
            trainer_kwargs=dict(accelerator=device)
        )

        # obs_outputs = predictions.x['decoder_target'] # [2*n_pred, lead_time]-size tensor, true observations
        # obs_outputs[0,:] == npzfile['RMM1'][start_forecast_test : start_forecast_test+lead_time]
        # obs_outputs[1,:] == npzfile['RMM2'][start_forecast_test : start_forecast_test+lead_time]
        # obs_outputs[2,:] == npzfile['RMM1'][start_forecast_test+1 : start_forecast_test+1+lead_time]
        # obs_outputs[3,:] == npzfile['RMM2'][start_forecast_test+1 : start_forecast_test+1+lead_time]


        predictions_outputs = predictions.output # [2*n_pred, lead_time, n_samples]-size tensor, predictions
        raw_predictions_outputs = raw_predictions.output[0] # [2*n_pred, lead_time, n_samples]-size tensor, predictions

        preds_y ={}
        raw_preds_y = {}
        obs_y = {}
        for rmm in ['RMM1', 'RMM2']:
            preds_y[rmm] = predictions_outputs[predictions.index.group == rmm]
            raw_preds_y[rmm] = raw_predictions_outputs[raw_predictions.index.group == rmm]
            obs_y[rmm] = predictions.x['decoder_target'][predictions.index.group == rmm]

        self.predictions = predictions
        self.raw_predictions = raw_predictions
        self.preds_y = preds_y
        self.raw_preds_y = raw_preds_y
        self.obs_y = obs_y