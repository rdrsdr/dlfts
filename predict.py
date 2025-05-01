from neuralforecast import NeuralForecast
from ray.tune.search.hyperopt import HyperOptSearch
from neuralforecast.losses.pytorch import MAE
from ray import tune
import torch
import pandas as pd
import numpy as np
import yfinance as yf

# import neuralforecast.auto
from neuralforecast.auto import AutoNHITS, AutoRNN, AutoLSTM, AutoGRU, AutoTCN, AutoDeepAR, AutoDilatedRNN, AutoBiTCN
from neuralforecast.auto import AutoMLP, AutoNBEATS, AutoNBEATSx, AutoDLinear, AutoNLinear, AutoTiDE, AutoDeepNPTS
from neuralforecast.auto import AutoTFT, AutoVanillaTransformer, AutoInformer, AutoAutoformer, AutoFEDformer
from neuralforecast.auto import AutoPatchTST, AutoiTransformer, AutoTimesNet

import logging
import ray

import warnings
import IPython
from IPython.display import clear_output
import sys
import datetime

#---------- Command line parameters helper functions ----------

def get_args(argv, model_names):
    # for arg in argv:
    #     print(arg)
    if (len(argv) != 5):
        print("Error: missing arguments")
        print_usage()
        return None

    model_number = get_model_number(argv[1], model_names)

    workdir = argv[2]

    if model_number < 0:
        return None
    
    try:
        datetime.date.fromisoformat(argv[4])
        date = pd.to_datetime(argv[4])
    except ValueError:
        print("Error: incorrect data format, should be YYYY-MM-DD")
        print_usage()
        return None
    
    try:
        dataset = pd.read_csv(workdir + argv[3])

        return model_number, workdir, dataset, date
    except Exception as e:
        print("Error: reading dataset: ", e)
        print_usage()
        return None

def print_usage():
    print()
    print('Usage: python predict.py <model_number> <workdir> <dataset> <split_date>')
    print('Example: python predict.py 0 ./m6/daily/ dataset.csv 2024-08-26')
    print()

def get_model_number(arg, model_names):
    if arg.isnumeric() == False or int(arg) < 0 or int(arg) > len(model_names):
        print('Error: model must be a number from 0 to 9:')
        printModelNames(model_names)
        print_usage()
        return -1
    
    return int(arg)

def printModelNames(model_names):
    for i in range(len(model_names)):
        print(f'\t{i}: {model_names[i]}')

#---------- Dataset helper functions ----------

# Find the first row where the date is equal or greater than split_date
def get_split_date_index(df, split_date):
  for i in range(len(df)):
    if df.iloc[i, 0] >= split_date:
      return i

# decomposes a pandas yfinance dataframe in numpy arrays so they can be manipulated more efficiently
# than directly on the dataframe for producing nixtla's bizzare [unique_df, ds, y] dataframe
def np_decompose(df, idx):
  ncols = np.array(df.columns[1:])
  ndates = df.iloc[:idx, 0].to_numpy()
  ndata = df.iloc[:idx, 1:].to_numpy().transpose()

  return ncols, ndates, ndata

# convert yfinance format to nixtla's bizzarre dataframe format
def gen_nixtlas_bizzarre_dataframe(dec_df):
  ncols, ndates, ndata = dec_df
  rows, cols = ndata.shape

  unique_id = np.repeat(ncols, cols)
  ds = np.tile(ndates, rows)
  y = ndata.reshape(-1)

  return pd.DataFrame({'unique_id': unique_id, 'ds': ds, 'y': y})

# numpy-pandas version
def convert_nixtla_np(df, idx):
  return gen_nixtlas_bizzarre_dataframe(np_decompose(df, idx))

# pure pandas version
def convert_nixtla_pd(df, idx):
  # Convert from wide to long format
  df_long = df.iloc[:idx, :].melt(id_vars=[date_column_name], var_name="ticker", value_name="price")

  # Rename columns for Nixtla’s long format and return
  return df_long.rename(columns={date_column_name: "ds", "ticker": "unique_id", "price": "y"})

#---------- Forecast helper functions ----------

# forecast function
def forecast(nf, df, idx, model_number):
  test_rows = len(df) - idx
  counter = 0

  # creating prediction dataframe
  preds = df.iloc[split_idx:, :].copy() # alocating
  preds.reset_index(drop=True, inplace=True) # reseting index
  preds[:] = 0 # zeroing values

  for i in range(test_rows):

    # just printing % progress bar
    div = (i * 1000) // test_rows
    if (div > counter):
        clear_output(wait=True)
        print(f'{MODEL_NAMES[model_number]} [{model_number}]: {((i * 100) / test_rows):.1f}%')
        counter = div

    # forecasting
    nixtla_df=convert_nixtla_np(df, split_idx + i)
    pred = nf.predict(df=nixtla_df)

    # transposing the prediction and adjusting columns
    #date_value = pd.to_datetime(pred['ds'].iloc[0]).strftime('%Y-%m-%d')
    date_value = pd.to_datetime(pred['ds'].iloc[0]).strftime('%Y-%m-%d %H:%M:%S')
    pred.set_index('unique_id', inplace=True)
    predt = pred.drop(columns=['ds']).T

    # copying forecasted row to preds dataframe
    preds.iloc[i, 0] = date_value
    preds.iloc[i, 1:] = predt.iloc[0, :]

  preds.columns = [preds.columns[0]] + list(predt.columns)

  return preds.reset_index(drop=True)

#---------- Constants ----------

MODEL_NAMES = [
    'lstm', # ......... 0
    'gru', # .......... 1
    'mlp', # .......... 2
    'dlinear', # ...... 3
    'nlinear', # ...... 4
    'informer', # ..... 5
    'autoformer', # ... 6
    'fedformer', # .... 7
    'bitcn', # ........ 8
    'rnn', # .......... 9

    'tcn',
    'deepar',
    'dilatedrnn',
    'nbeats',
    'nbeatsx',
    'nhits',
    'tide',
    'deepnpts',
    'tft',
    'vanilla',
    'patchtst',
    'itransformer',
    'timesnet'
]

# pre-trained models' folder
mfolder = 'models/'

# forecasts' folder
ffolder = 'forecasts/'

#---------- Check if command line params args are ok ----------

# check first if args are ok, if so, then proceed
args = get_args(sys.argv, MODEL_NAMES)

if args is None:
    sys.exit()

model_number, workdir, dataset, split_date = args

dataset = dataset.dropna(axis=1)

if dataset.isnull().values.any():
    print('Error: dataset contains NaN values')
    sys.exit()

mfolder = workdir + mfolder
ffolder = workdir + ffolder

#---------- dataset loading ----------

df = dataset.drop('Real', axis=1)
date_column_name = df.columns[0]
#date_format = '%Y-%m-%d' if date_column_name.lower() == 'date' else '%Y-%m-%d %H:%M:%S'

# Convert Date[time] column from str to datetime
#df[date_column_name] = pd.to_datetime(df[date_column_name]).dt.tz_convert(None)

if date_column_name.lower() == 'date':
    df[date_column_name] = pd.to_datetime(df[date_column_name])
else:
    df[date_column_name] = pd.to_datetime(df[date_column_name]).dt.tz_convert(None)

split_idx = get_split_date_index(df, split_date)

#---------- Pythorch and Ray initialization ----------

#torch.set_float32_matmul_precision('medium' | 'high' | 'highest')
torch.set_float32_matmul_precision('highest')

logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
ray.init(log_to_driver=False)

warnings.filterwarnings('ignore')

#---------- Main forecast loop ----------

i = model_number
print(f'Forecasting {MODEL_NAMES[i].upper()} since {split_date}')
nf = NeuralForecast.load(mfolder + MODEL_NAMES[i])
fc = forecast(nf, df, split_idx, model_number)
fc.to_csv(ffolder + 'forecast-' + MODEL_NAMES[i] + '.csv', index=False)

#-------------------------------------------------

IPython.display.Audio("file_example_MP3_1MG.mp3", autoplay=True)
