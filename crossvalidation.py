import logging
#import matplotlib.pyplot as plt
import pandas as pd
#from utilsforecast.plotting import plot_series

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from neuralforecast.auto import AutoNHITS, AutoRNN, AutoLSTM, AutoGRU, AutoTCN, AutoDeepAR, AutoDilatedRNN, AutoBiTCN
from neuralforecast.auto import AutoMLP, AutoNBEATS, AutoNBEATSx, AutoDLinear, AutoNLinear, AutoTiDE, AutoDeepNPTS
from neuralforecast.auto import AutoMLPMultivariate, AutoSOFTS, AutoTimeMixer, AutoTSMixer, AutoTSMixerx
from neuralforecast.auto import AutoTFT, AutoVanillaTransformer, AutoInformer, AutoAutoformer, AutoFEDformer, AutoTimeXer
from neuralforecast.auto import AutoPatchTST, AutoiTransformer, AutoTimesNet
from neuralforecast.auto import AutoHINT, AutoKAN, AutoRMoK, AutoStemGNN

# ----------

import logging
import os
import warnings
import torch

from datetime import timedelta

warnings.filterwarnings('ignore')

# Change the default logging directory
os.environ["LIGHTNING_LOGS_DIR"] = "/workdir/my_lightning_logs"  # Or any other desired path

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

torch.set_float32_matmul_precision('high')  # 'high' | 'highest' for better performance

# ----------

TRAIN_TEST_SPLIT_DATE = '2025-04-08'

#folder = './datasets/binance/15m/2024_2/'
folder = './datasets/ufpe/'

cross_folder = folder + 'cross-validation'#-train'
#refit_folder = folder + 'cross-validation/refit'
output_folder = cross_folder

model_folder = '/models/'
forecast_folder = '/forecasts/'

os.makedirs(output_folder + forecast_folder, exist_ok=True)

df = pd.read_csv(folder + 'dataset.csv')

# for col in df.columns:
#   if col != 'datetime':
#     df[col] = df[col].astype('float32')

# ----------

date_column_name = df.columns[0]
date_format = '%Y-%m-%d' if date_column_name.lower() == 'date' else '%Y-%m-%d %H:%M:%S'

# ----------

# Convert column to datetime removing timezone information
if date_column_name.lower() == 'date':
    df[date_column_name] = pd.to_datetime(df[date_column_name])
else:
    df[date_column_name] = pd.to_datetime(df[date_column_name]).dt.tz_localize(None)

# ----------

# separate training and test data
# from start until 2025-01-01
train_df = df[df[date_column_name] < TRAIN_TEST_SPLIT_DATE] 
#test_df = df[df[date_column_name] >= '2025-01-01']
print()
print(f"Training data shape: {train_df.shape}")
print(f"Training from {train_df.iloc[0, 0]} to {train_df.iloc[-1, 0]}")
print()

# ----------

# conversion to long format, pure pandas version (faster, more memory-friendly)
def convert_nixtla(df):
  # Convert from wide to long format
  df_long = df.melt(id_vars=[date_column_name], var_name="ticker", value_name="price")

  # Rename columns for Nixtla’s long format and return
  return df_long.rename(columns={date_column_name: "ds", "ticker": "unique_id", "price": "y"})

# ----------

Y_df = convert_nixtla(df)
#Y_df = convert_nixtla(train_df)
#Y_df['y'] = Y_df['y'].astype('float32')

n_series = df.shape[1] - 1

# ----------

# Configs

horizont = 1
input_size = 24 # using one day of historical data

#A. RNN-Based
rnn_config = AutoRNN.get_default_config(h = horizont, backend="ray")
lstm_config = AutoLSTM.get_default_config(h = horizont, backend="ray")
gru_config = AutoGRU.get_default_config(h = horizont, backend="ray")
tcn_config = AutoTCN.get_default_config(h = horizont, backend="ray")
deep_ar_config = AutoDeepAR.get_default_config(h = horizont, backend="ray")
dilated_rnn_config = AutoDilatedRNN.get_default_config(h = horizont, backend="ray")

#B. MLP-Based
mlp_config = AutoMLP.get_default_config(h = horizont, backend="ray")
nbeats_config = AutoNBEATS.get_default_config(h = horizont, backend="ray")
nbeatsx_config = AutoNBEATSx.get_default_config(h = horizont, backend="ray")
nhits_config = AutoNHITS.get_default_config(h = horizont, backend="ray")
dlinear_config = AutoDLinear.get_default_config(h = horizont, backend="ray")
nlinear_config = AutoNLinear.get_default_config(h = horizont, backend="ray")
tide_config = AutoTiDE.get_default_config(h = horizont, backend="ray")
deep_npts_config = AutoDeepNPTS.get_default_config(h = horizont, backend="ray")

mlpmulti_config = AutoMLPMultivariate.get_default_config(h = horizont, n_series=n_series, backend="ray") # <-- ADD THIS
softs_config = AutoSOFTS.get_default_config(h = horizont, n_series=n_series, backend="ray") # <-- ADD THIS
time_mixer_config = AutoTimeMixer.get_default_config(h = horizont, n_series=n_series, backend="ray") # <-- ADD THIS
ts_mixer_config = AutoTSMixer.get_default_config(h = horizont, n_series=n_series, backend="ray") # <-- ADD THIS
ts_mixerx_config = AutoTSMixerx.get_default_config(h = horizont, n_series=n_series, backend="ray") # <-- ADD THIS

#C. Transformer models
tft_config = AutoTFT.get_default_config(h = horizont, backend="ray")
vanilla_config = AutoVanillaTransformer.get_default_config(h = horizont, backend="ray")
informer_config = AutoInformer.get_default_config(h = horizont, backend="ray")
autoformer_config = AutoAutoformer.get_default_config(h = horizont, backend="ray")
fedformer_config = AutoFEDformer.get_default_config(h = horizont, backend="ray")
patch_tst_config = AutoPatchTST.get_default_config(h = horizont, backend="ray")
itransformer_config = AutoiTransformer.get_default_config(h = horizont, n_series=n_series, backend="ray")

time_xer_config = AutoTimeXer.get_default_config(h = horizont, n_series=n_series, backend="ray") # <-- ADD THIS

#D. CNN Based
bitcn_config = AutoBiTCN.get_default_config(h = horizont, backend="ray")
timesnet_config = AutoTimesNet.get_default_config(h = horizont, backend="ray")

#E. Any
#hint_config = nhits_config # <-- ADD THIS

#F. KAN
kan_config = AutoKAN.get_default_config(h = horizont, backend="ray") # <-- ADD THIS
rmok_config = AutoRMoK.get_default_config(h = horizont, n_series=n_series, backend="ray") # <-- ADD THIS

#H. GNN
stemgnn_config = AutoStemGNN.get_default_config(h = horizont, n_series=n_series, backend="ray") # <-- ADD THIS

configs = [
    #A. RNN-Based
    rnn_config,
    lstm_config,
    gru_config,
    tcn_config,
    deep_ar_config,
    dilated_rnn_config,
    bitcn_config,

    #B. MLP-Based
    mlp_config,
        #nbeats_config,
        #nbeatsx_config,
    nhits_config,
    dlinear_config,
    nlinear_config,
    tide_config,
    deep_npts_config,

    mlpmulti_config,
    softs_config,
        #time_mixer_config,
    ts_mixer_config,
    ts_mixerx_config,

    #C. Transformer models
    tft_config,
    vanilla_config,
    informer_config,
    autoformer_config,
    fedformer_config,
    patch_tst_config,
    itransformer_config,
        #time_xer_config,

    #D. CNN Based
        #timesnet_config,

    #E. Any
        #hint_config,

    #F. KAN
        #kan_config,
    rmok_config,

    #H. GNN
    stemgnn_config
]

# changing input size for all models

for config in configs:
    config['input_size'] = input_size
    #config['num_samples'] = 50
    #config['trainer']['max_epochs'] = 5
    #config['trainer']['accelerator'] = 'gpu' if torch.cuda.is_available() else 'cpu'
    #config['trainer']['devices'] = 1

deep_ar_config['input_size'] = 6 # specifically for DeepAR, which handles less history

# ----------

# Models

verbose = False

#A. RNN-Based
rnn_model = AutoRNN(h=horizont, config=rnn_config, verbose=verbose)
lstm_model = AutoLSTM(h=horizont, config=lstm_config, verbose=verbose)
gru_model = AutoGRU(h=horizont, config=gru_config, verbose=verbose)
tcn_model = AutoTCN(h=horizont, config=tcn_config, verbose=verbose)
deep_ar_model = AutoDeepAR(h=horizont, config=deep_ar_config, verbose=verbose)
dilated_rnn_model = AutoDilatedRNN(h=horizont, config=dilated_rnn_config, verbose=verbose)
bitcn_model = AutoBiTCN(h=horizont, config=bitcn_config, verbose=verbose)

#B. MLP-Based
mlp_model = AutoMLP(h=horizont, config=mlp_config, verbose=verbose)
nbeats_model = AutoNBEATS(h=horizont, config=nbeats_config, verbose=verbose)
nbeatsx_model = AutoNBEATSx(h=horizont, config=nbeats_config, verbose=verbose)
nhits_model = AutoNHITS(h=horizont, config=nhits_config, verbose=verbose)
dlinear_model = AutoDLinear(h=horizont, config=dlinear_config, verbose=verbose)
nlinear_model = AutoNLinear(h=horizont, config=nlinear_config, verbose=verbose)
tide_model = AutoTiDE(h=horizont, config=tide_config, verbose=verbose)
deep_npts_model = AutoDeepNPTS(h=horizont, config=deep_npts_config, verbose=verbose)

mlpmulti_model = AutoMLPMultivariate(h=horizont, n_series=n_series, config=mlpmulti_config, verbose=verbose)
softs_model = AutoSOFTS(h=horizont, n_series=n_series, config=softs_config, verbose=verbose)
time_mixer_model = AutoTimeMixer(h=horizont, n_series=n_series, config=time_mixer_config, verbose=verbose)
ts_mixer_model = AutoTSMixer(h=horizont, n_series=n_series, config=ts_mixer_config, verbose=verbose)
ts_mixerx_model = AutoTSMixerx(h=horizont, n_series=n_series, config=ts_mixerx_config, verbose=verbose)

#C. Transformer models
tft_model = AutoTFT(h=horizont, config=tft_config, verbose=verbose)
vanilla_model = AutoVanillaTransformer(h=horizont, config=vanilla_config, verbose=verbose)
informer_model = AutoInformer(h=horizont, config=informer_config, verbose=verbose)
autoformer_model = AutoAutoformer(h=horizont, config=autoformer_config, verbose=verbose)
fedformer_model = AutoFEDformer(h=horizont, config=fedformer_config, verbose=verbose)
patch_tst_model = AutoPatchTST(h=horizont, config=patch_tst_config, verbose=verbose)
itransformer_model = AutoiTransformer(h=horizont, n_series=n_series, config=itransformer_config, verbose=verbose)

time_xer_model = AutoTimeXer(h=horizont, n_series=n_series, config=time_xer_config, verbose=verbose)

#D. CNN Based
timesnet_model = AutoTimesNet(h=horizont, config=timesnet_config, verbose=verbose)

#E. Any
#hint_model = AutoHINT(h=horizont, config=hint_config, verbose=verbose)

#F. KAN
kan_model = AutoKAN(h=horizont, config=kan_config, verbose=verbose)
rmok_model = AutoRMoK(h=horizont, n_series=n_series, config=rmok_config, verbose=verbose)

#H. GNN
stemgnn_model = AutoStemGNN(h=horizont, n_series=n_series, config=stemgnn_config, verbose=verbose)

# ----------

MODEL_NAMES = [
    
    #A. RNN-Based
    'rnn',
    'lstm',
    'gru',
    'tcn',
    'deep_ar',
    
    'dilated_rnn',
    'bitcn',

    #B. MLP-Based
    'mlp',
        #'nbeats',
        #'nbeatsx',
    'nhits',
    'dlinear',
    'nlinear',
    'tide',
    'deep_npts',

    'mlpmulti',
    #'softs',
        #'time_mixer',
    'ts_mixer',
    'ts_mixerx',

    #C. Transformer models
    'tft',
    'vanilla',
    'informer',
    'autoformer',
    'fedformer',
    'patch_tst',
    'itransformer',
        #'time_xer',

    #D. CNN Based
        #'timesnet',

    #E. Any
        #'hint',

    #F. KAN
        #kan_model, # out-of-memory :-(
    'rmok',

    #H. GNN
    'stemgnn'
]

#MODEL_NAMES = [ 'mlpmulti' ]  # TEMPORARY LIMITATION TO DEEPAR ONLY

# ----------

models = [
    #A. RNN-Based
    rnn_model,
    lstm_model,
    gru_model,
    tcn_model,
    deep_ar_model,
    
    dilated_rnn_model,
    bitcn_model,

    #B. MLP-Based
    mlp_model,
        #nbeats_model,
        #nbeatsx_model,
    nhits_model,
    dlinear_model,
    nlinear_model,
    tide_model,
    deep_npts_model,

    mlpmulti_model,
    softs_model,
        #time_mixer_model,
    ts_mixer_model,
    ts_mixerx_model,

    #C. Transformer models
    tft_model,
    vanilla_model,
    informer_model,
    autoformer_model,
    fedformer_model,
    patch_tst_model,
    itransformer_model,
        #time_xer_model,

    #D. CNN Based
        #timesnet_model,

    #E. Any
        #hint_model,

    #F. KAN
        #kan_model, # out-of-memory :-(
    rmok_model,

    #H. GNN
    stemgnn_model
]

#models = [ mlpmulti_model ]  # TEMPORARY LIMITATION

# ----------

def format_time_delta(start, end):
    delta = end - start
    hours, remainder = divmod(delta.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{int(hours)}:{int(minutes)}:{seconds:.2f}"

# ----------

h = horizont
n_windows=24*32*6 # +-6 months of hourly data
refit=False
#refit=24*30 #refit every 32 days (to make it compatible to tensors 32 batch size)
verbose=False

# ----------

train_individually = True

if train_individually:
    print("Training each model individually")

    # Configure logging
    logging.basicConfig(filename='crossvalidation.log', level=logging.INFO, filemode='w')

    nfs = []
    cv_dfs = []

    start_time = pd.Timestamp.now()

    # for model in models:
    len_models = len(models)
    try:
        for i in range(len_models):
            now = pd.Timestamp.now()
            print(f"[{i+1}/{len_models}] Starting cross validation for {models[i]} at {now}")
            logging.info(f"[{i+1}/{len_models}] Starting cross validation for {models[i]} at {now}")
            nf = NeuralForecast(models=[models[i]], freq='h');

            cv_df = nf.cross_validation(Y_df, n_windows=n_windows, step_size=h, refit=refit, verbose=verbose)
            cv_dfs.append(cv_df)
            nfs.append(nf)
            nfs[i].save(output_folder + model_folder + MODEL_NAMES[i].lower(), overwrite=True)
            cv_dfs[i].to_csv(output_folder + forecast_folder + MODEL_NAMES[i] + '.csv', index=False)
            end_time = pd.Timestamp.now()
            print(f"Finished cross validation of {models[i]} in {format_time_delta(now, end_time)}")
            logging.info(f"Finished cross validation of {models[i]} in {format_time_delta(now, end_time)}")
    # when a model fails, all other consecutive models fails too, so abort training
    except Exception as e:
        print(f"Error in cross validation for {models[i]}: {e}")
        logging.error(f"Error in cross validation for {models[i]}: {e}")

else:
    print("Training all models at once")

    nf = NeuralForecast(models=models, freq='h');
    cv_df = nf.cross_validation(Y_df, n_windows=n_windows, step_size=h, refit=refit, verbose=verbose)

    nf.save(cross_folder + model_folder + 'ALL', overwrite=True)
    cv_df.to_csv(cross_folder + forecast_folder + 'ALL.csv', index=False)

print(f"Cross validation completed for all models at {pd.Timestamp.now()}. Total Time: {format_time_delta(start_time, end_time)}")
logging.info(f"Cross validation completed for all models at {pd.Timestamp.now()}. Total Time: {format_time_delta(start_time, end_time)}")
