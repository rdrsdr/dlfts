import logging
#import matplotlib.pyplot as plt
import pandas as pd
#from utilsforecast.plotting import plot_series

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from neuralforecast.auto import AutoNHITS, AutoRNN, AutoLSTM, AutoGRU, AutoTCN, AutoDeepAR, AutoDilatedRNN, AutoBiTCN
from neuralforecast.auto import AutoMLP, AutoNBEATS, AutoNBEATSx, AutoDLinear, AutoNLinear, AutoTiDE, AutoDeepNPTS
from neuralforecast.auto import AutoTFT, AutoVanillaTransformer, AutoInformer, AutoAutoformer, AutoFEDformer
from neuralforecast.auto import AutoPatchTST, AutoiTransformer, AutoTimesNet

# ----------

import logging
import os
import warnings

warnings.filterwarnings('ignore')

# Change the default logging directory
os.environ["LIGHTNING_LOGS_DIR"] = "/workdir/my_lightning_logs"  # Or any other desired path

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# ----------

folder = './datasets/crypto/hourly/'

cross_folder = folder + '/cross-validation/plain'
refit_folder = folder + '/cross-validation/refit'
output_folder = cross_folder

model_folder = '/models/'
forecast_folder = '/forecasts/'

os.makedirs(output_folder + forecast_folder, exist_ok=True)

df = pd.read_csv(folder + 'dataset.csv')

# ----------

date_column_name = df.columns[0]
date_format = '%Y-%m-%d' if date_column_name.lower() == 'date' else '%Y-%m-%d %H:%M:%S'

# ----------

# Convert column to datetime removing timezone information
if date_column_name.lower() == 'date':
    df[date_column_name] = pd.to_datetime(df[date_column_name])
else:
    df[date_column_name] = pd.to_datetime(df[date_column_name]).dt.tz_convert(None)

# ----------

# conversion to long format, pure pandas version (faster, more memory-friendly)
def convert_nixtla(df):
  # Convert from wide to long format
  df_long = df.melt(id_vars=[date_column_name], var_name="ticker", value_name="price")

  # Rename columns for Nixtla’s long format and return
  return df_long.rename(columns={date_column_name: "ds", "ticker": "unique_id", "price": "y"})

# ----------

Y_df = convert_nixtla(df.drop('Real', axis=1))

# ----------

# Configs

horizont = 1

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

#C. Transformer models
tft_model = AutoTFT(h=horizont, config=tft_config, verbose=verbose)
vanilla_model = AutoVanillaTransformer(h=horizont, config=vanilla_config, verbose=verbose)
informer_model = AutoInformer(h=horizont, config=informer_config, verbose=verbose)
autoformer_model = AutoAutoformer(h=horizont, config=autoformer_config, verbose=verbose)
fedformer_model = AutoFEDformer(h=horizont, config=fedformer_config, verbose=verbose)
patch_tst_model = AutoPatchTST(h=horizont, config=patch_tst_config, verbose=verbose)

itransformer_model = AutoiTransformer(h=horizont, n_series=1, config=itransformer_config, verbose=verbose)

#D. CNN Based
timesnet_model = AutoTimesNet(h=horizont, config=timesnet_config, verbose=verbose)

# ----------

MODEL_NAMES = [
    'lstm',
    'gru',
    'mlp',
    'dlinear',
    'nlinear',
    'informer',
    'autoformer',
    'fedformer',
    'bitcn',
    'rnn',

    'tcn',
    'deep_ar',
    'dilated_rnn',
        #nbeats,
        #nbeatsx,
    'nhits',
    'tide',
    'deep_npts',
    'tft',
    'vanilla',
    'patch_tst',
    #'itransformer'
]

# ----------

models = [
    lstm_model,         #
    gru_model,          #
    mlp_model,          #
    dlinear_model,      #
    nlinear_model,      #
    informer_model,     #
    autoformer_model,   #
    fedformer_model,    #
    bitcn_model,        #
    rnn_model,          #

    tcn_model,          #
    deep_ar_model,      #
    dilated_rnn_model,  #
        #nbeats,
        #nbeatsx,
    nhits_model,        #
    tide_model,         #
    deep_npts_model,    #
    tft_model,          #
    vanilla_model,      #
    patch_tst_model,    #
    #itransformer_model
]

# ----------

h = horizont
n_windows=5888
refit=False
#refit=24*30 #refit every 32 days (to make it compatible to tensors 32 batch size)
verbose=False

# ----------

train_individually = True

if train_individually:
    print("Training each model individually")

    # Configure logging
    logging.basicConfig(filename='crossvalidation.log', level=logging.ERROR, filemode='w')

    nfs = []
    cv_dfs = []

    # for model in models:
    for i in range(len(models)):
        print(f"Starting cross validation for model {type(models[i])}")
        logging.info(f"Starting cross validation for model {type(models[i])}")
        nf = NeuralForecast(models=[models[i]], freq='h');

        try:
            cv_df = nf.cross_validation(Y_df, n_windows=n_windows, step_size=h, refit=refit, verbose=verbose)
            cv_dfs.append(cv_df)
            nfs.append(nf)
            nfs[i].save(output_folder + model_folder + MODEL_NAMES[i].lower(), overwrite=True)
            cv_dfs[i].to_csv(output_folder + forecast_folder + MODEL_NAMES[i] + '.csv', index=False)
            print(f"Finished cross validation of model {type(models[i])}")
            logging.info(f"Finished cross validation of model {type(models[i])}")
        except Exception as e:
            print(f"Error in cross validation for model {type(models[i])}: {e}")
            logging.error(f"Error in cross validation for model {type(models[i])}: {e}")

else:
    print("Training all models at once")

    nf = NeuralForecast(models=models, freq='h');
    cv_df = nf.cross_validation(Y_df, n_windows=n_windows, step_size=h, refit=refit, verbose=verbose)

    nf.save(cross_folder + model_folder + 'ALL', overwrite=True)
    cv_df.to_csv(cross_folder + forecast_folder + 'ALL.csv', index=False)

# ----------



# ----------



# ----------



# ----------

