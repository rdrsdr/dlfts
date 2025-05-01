from datetime import datetime

notebook_start = datetime.now()

import os

# this makes it so that the outputs of the predict methods have the id as a column
# instead of as the index
os.environ['NIXTLA_ID_AS_COL'] = '1'

import pandas as pd

#folder = './datasets/247/daily/'
#folder = './datasets/247/hourly/'
folder = './datasets/m6/hourly/'
folder = './datasets/crypto/hourly/'

#dataset = 'm6dataset.csv'
dataset = folder + 'dataset.csv'

df = pd.read_csv(dataset)
#df

date_column_name = df.columns[0]
date_format = '%Y-%m-%d' if date_column_name.lower() == 'date' else '%Y-%m-%d %H:%M:%S'

#date_column_name, date_format

# Converter a coluna para datetime removendo o fuso horário
if date_column_name.lower() == 'datetime':
  df[date_column_name] = pd.to_datetime(df[date_column_name]).dt.tz_localize(None)
else:
  df[date_column_name] = pd.to_datetime(df[date_column_name], format=date_format)
#df

split_date = pd.to_datetime('2024-08-26')
#split_date = pd.to_datetime('2025-01-01')

# Find the first row where the date is equal or greater than split_date
def get_split_date_index(df, split_date):
  for i in range(len(df)):
    if df.iloc[i, 0] >= split_date:
      return i

split_idx = get_split_date_index(df, split_date)
#split_idx, df.iloc[split_idx, 0]

train_df = df.iloc[:split_idx].drop('Real', axis=1)
#train_df

test_df = df.iloc[split_idx:].drop('Real', axis=1)
#test_df

"""# NIXTLA"""

#pip install neuralforecast

from neuralforecast import NeuralForecast
from ray.tune.search.hyperopt import HyperOptSearch
from neuralforecast.losses.pytorch import MAE
from ray import tune
import torch

#torch.set_float32_matmul_precision('medium' | 'high' | 'highest')
torch.set_float32_matmul_precision('highest')

"""## Auto Models"""

import warnings
warnings.filterwarnings('ignore')

import neuralforecast.auto
from neuralforecast.auto import AutoNHITS, AutoRNN, AutoLSTM, AutoGRU, AutoTCN, AutoDeepAR, AutoDilatedRNN, AutoBiTCN
from neuralforecast.auto import AutoMLP, AutoNBEATS, AutoNBEATSx, AutoDLinear, AutoNLinear, AutoTiDE, AutoDeepNPTS
from neuralforecast.auto import AutoTFT, AutoVanillaTransformer, AutoInformer, AutoAutoformer, AutoFEDformer
from neuralforecast.auto import AutoPatchTST, AutoiTransformer, AutoTimesNet

horizont = 1

# --- CONFIGS ---

# Extract the default hyperparameter settings

#A. RNN-Based
rnn_config = AutoRNN.get_default_config(h = horizont, backend="ray")
lstm_config = AutoLSTM.get_default_config(h = horizont, backend="ray")
gru_config = AutoGRU.get_default_config(h = horizont, backend="ray")
tcn_config = AutoTCN.get_default_config(h = horizont, backend="ray")
deep_ar_config = AutoDeepAR.get_default_config(h = horizont, backend="ray")
dilated_rnn_config = AutoDilatedRNN.get_default_config(h = horizont, backend="ray")
bitcn_config = AutoBiTCN.get_default_config(h = horizont, backend="ray")

#B. MLP-Based
mlp_config = AutoMLP.get_default_config(h = horizont, backend="ray")
nbeats_config = AutoNBEATS.get_default_config(h = horizont, backend="ray")
nbeatsx_config = AutoNBEATSx.get_default_config(h = horizont, backend="ray")
nhits_config = AutoNHITS.get_default_config(h = horizont, backend="ray")
dlinear_config = AutoDLinear.get_default_config(h = horizont, backend="ray")
nlinear_config = AutoNLinear.get_default_config(h = horizont, backend="ray")
tide_config = AutoTiDE.get_default_config(h = horizont, backend="ray")
deep_npts_config = AutoDeepNPTS.get_default_config(h = horizont, backend="ray")

#C. Transformer models
tft_config = AutoTFT.get_default_config(h = horizont, backend="ray")
vanilla_config = AutoVanillaTransformer.get_default_config(h = horizont, backend="ray")
informer_config = AutoInformer.get_default_config(h = horizont, backend="ray")
autoformer_config = AutoAutoformer.get_default_config(h = horizont, backend="ray")
fedformer_config = AutoFEDformer.get_default_config(h = horizont, backend="ray")
patch_tst_config = AutoPatchTST.get_default_config(h = horizont, backend="ray")

itransformer_config = AutoiTransformer.get_default_config(h = horizont, n_series=1, backend="ray")

#D. CNN Based
timesnet_config = AutoTimesNet.get_default_config(h = horizont, backend="ray")

# --- MODELS ---
#A. RNN-Based
rnn_model = AutoRNN(h=horizont, config=rnn_config)
lstm_model = AutoLSTM(h=horizont, config=lstm_config)
gru_model = AutoGRU(h=horizont, config=gru_config)
tcn_model = AutoTCN(h=horizont, config=tcn_config)
deep_ar_model = AutoDeepAR(h=horizont, config=deep_ar_config)
dilated_rnn_model = AutoDilatedRNN(h=horizont, config=dilated_rnn_config)
bitcn_model = AutoBiTCN(h=horizont, config=bitcn_config)

#B. MLP-Based
mlp_model = AutoMLP(h=horizont, config=mlp_config)
nbeats_model = AutoNBEATS(h=horizont, config=nbeats_config)
nbeatsx_model = AutoNBEATSx(h=horizont, config=nbeats_config)
nhits_model = AutoNHITS(h=horizont, config=nhits_config)
dlinear_model = AutoDLinear(h=horizont, config=dlinear_config)
nlinear_model = AutoNLinear(h=horizont, config=nlinear_config)
tide_model = AutoTiDE(h=horizont, config=tide_config)
deep_npts_model = AutoDeepNPTS(h=horizont, config=deep_npts_config)

#C. Transformer models
tft_model = AutoTFT(h=horizont, config=tft_config)
vanilla_model = AutoVanillaTransformer(h=horizont, config=vanilla_config)
informer_model = AutoInformer(h=horizont, config=informer_config)
autoformer_model = AutoAutoformer(h=horizont, config=autoformer_config)
fedformer_model = AutoFEDformer(h=horizont, config=fedformer_config)
patch_tst_model = AutoPatchTST(h=horizont, config=patch_tst_config)

itransformer_model = AutoiTransformer(h=horizont, n_series=1, config=itransformer_config)

#D. CNN Based
timesnet_model = AutoTimesNet(h=horizont, config=timesnet_config)

# pure pandas version (faster, more memory-friendly)
def convert_nixtla(df):
  # Convert from wide to long format
  df_long = df.melt(id_vars=[date_column_name], var_name="ticker", value_name="price")

  # Rename columns for Nixtla’s long format and return
  return df_long.rename(columns={date_column_name: "ds", "ticker": "unique_id", "price": "y"})

MODEL_NAMES = [
  'LSTM',
  'GRU',
  'MLP',
  'DLINEAR',
  'NLINEAR',
  'INFORMER',
  'AUTOFORMER',
  'FEDFORMER',
  'BITCN',
  'RNN',

  'TCN',
  'DEEPAR',
  'DILATEDRNN',
  'NBEATS',
  'NBEATSX',
  'NHITS',
  'TiDE',
  'DEEPNPTS',
  'TFT',
  'VANILLA',
  'PATCHTST',
  'ITRANSFORMER',
  'TIMESNET'
]

MODELS = [
  lstm_model, # ........ 0
  gru_model, # ......... 1
  mlp_model, # ......... 2
  dlinear_model, # ..... 3
  nlinear_model, # ..... 4
  informer_model, # .... 5
  autoformer_model, # .. 6
  fedformer_model, # ... 7
  bitcn_model, # ....... 8
  rnn_model, # ......... 9

  tcn_model,
  deep_ar_model,
  dilated_rnn_model,
  nbeats_model,
  nbeatsx_model,
  nhits_model,
  tide_model,
  deep_npts_model,
  tft_model,
  vanilla_model,
  patch_tst_model,
  itransformer_model,
  timesnet_model,
]

# Converter a coluna para datetime removendo o fuso horário
if date_column_name.lower() == 'datetime':
  frequency = 'h'
  print()
  print('----------------------------------------------------')
  print('---------- Training HOURLY data frequency ----------')
  print('----------------------------------------------------')
  print()
else:
  frequency = 'D' # 'B' business day frequency (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
  print()
  print('---------------------------------------------------')
  print('---------- Training DAILY data frequency ----------')
  print('---------------------------------------------------')
  print()

models = [
    #MODELS[0]
    #lstm_model,  # ......... 1
    #gru_model,  # .......... 2
    #tcn_model,
    #deep_ar_model,
    #dilated_rnn_model,
    #mlp_model,  # .......... 3
    #nbeats_model,
    #nbeatsx_model,
    #nhits_model,
    #dlinear_model,  # ...... 4
    #nlinear_model,  # ...... 5
    #tide_model,
    #deep_npts_model,
    #tft_model,
    #vanilla_model,
    #informer_model,  # ..... 6
    #autoformer_model,  # ... 7
    #fedformer_model,  # .... 8
    #bitcn_model,  # ........ 9
    #rnn_model,  # .......... 10
    #patch_tst_model,
    #itransformer_model,
    #timesnet_model
    ]

# Commented out IPython magic to ensure Python compatibility.
# %%time
start_time = datetime.now()
error = False

for i in range(len(MODELS)):
    print()
    print(f'---------------------------------')
    print(f'--- Training {MODEL_NAMES[i]} ---')
    print(f'---------------------------------')
    print()

    nf = NeuralForecast(
          models= [ MODELS[i] ],
          freq=frequency
        )

    ndf=convert_nixtla(train_df)
    #nf.fit(df=ndf)
    #nf.save(folder + 'models/' + MODEL_NAMES[i].lower())
    try:
        nf.fit(df=ndf)
        nf.save(folder + 'models/' + MODEL_NAMES[i].lower())
    except Exception as e:
        # Log the error with timestamp and model name
        with open('exceptions.log', 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'[{timestamp}] Error training {MODEL_NAMES[i]}: {str(e)}\n')
        error = True    

if error:
    print()
    print('----------------------------------------------------------------------')
    print('--- Training finished with errors. See exceptions.log for details. ---')
    print('----------------------------------------------------------------------')
    print()

stop_time = datetime.now()
elapsed_time = stop_time - start_time

print(f"Cell started at: {start_time}")
print(f"Cell stopped at: {stop_time}")
print(f"Elapsed time: {elapsed_time}")

import IPython
IPython.display.Audio("file_example_MP3_1MG.mp3", autoplay=True)
