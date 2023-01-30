import warnings
import numpy as np
from urllib.parse import urlparse
import mlflow
import argparse
import webbrowser
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
'''
A seguir será estruturada a rede LSTM a partir da ferramenta MLflow. Todo o pipeline utilizado aqui foi herdado do notebook da questão 01
'''

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#Parâmetros de entrada
parser = argparse.ArgumentParser()
parser.add_argument('--n_future', type=int, default=20, help='número de amostras futuras que serão previstas')
parser.add_argument('--epochs', type=int, default=100, help='número épocas de treinamento')
parser.add_argument('--neurons', type=int, default=13, help='quantidade de neurônios das duas primeiras camadas da LSTM')
parser.add_argument('--offset', type=int, default=2,
                    help='offset de amostras anteriores que não serão utilizadas na previsão')
parser.add_argument('--look_back', type=int, default=5,
                    help='Quantidade de amostras anteriores utilizadas para previsão das amostras futuras')
opt = parser.parse_args()
print(opt)
n_future, look_back, offset, epochs, neurons = opt.n_future, opt.look_back, opt.offset, opt.epochs, opt.neurons

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40) #fixa uma semente para reprodutibilidade

    #Leitura e pré-processamento dos dados
    base = pd.read_csv("jena_climate_2009_2016.csv")
    base = base[0::6]
    base['Date Time'] = pd.to_datetime(base['Date Time'], format='%d.%m.%Y %H:%M:%S')
    base = base.set_index(pd.DatetimeIndex(base['Date Time'].values))
    base = base.drop('Date Time', axis=1)
    datelist_train = base.index
    columns_titles = list(base)[:]  # invertendo as colunas 0 e 1 para facilitar passos seguintes
    aux = columns_titles[1]
    columns_titles[1] = columns_titles[0]
    columns_titles[0] = aux
    base = base.reindex(columns=columns_titles)

    #Normlização
    scale = StandardScaler()
    base_norm = scale.fit_transform(base)
    scale2 = StandardScaler()  # obtendo scale2 para uso futuro na retomada dos dados ao valor original
    y_norm = scale2.fit_transform(base[['T (degC)']])

    #Separação entre entrada e saída
    y_base, x_base = [], []
    for i in range(len(base_norm) - look_back - 1):
        a = base_norm[i:(i + look_back), 1:14]
        x_base.append(a)
        y_base.append(base_norm[i + look_back, 0])
    x_base = np.array(x_base)
    y_base = np.array(y_base)
    with mlflow.start_run():

        #Criação do modelo
        model = Sequential()
        model.add(LSTM(neurons, return_sequences=True, input_shape=(look_back, base_norm.shape[1] - 1)))
        model.add(LSTM(neurons, return_sequences=False))
        model.add(Dropout(0.1))
        model.add(Dense(1))

        #Treinamento do modelo
        model.compile(loss='mean_squared_error', optimizer='adam')
        es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=0)
        rlr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=0)
        tb = TensorBoard('logs')
        val = model.fit(x_base, y_base, epochs=epochs, callbacks=[es, rlr, tb], verbose=1, batch_size=256)

        #Rotulação das amostras previstas
        datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1H').tolist()
        datelist_future_ = []
        for this_timestamp in datelist_future:
            datelist_future_.append(this_timestamp.date())

        #Previsão das amostras futuras utilizando o modelo treinado
        predictions_future = model.predict(x_base[-n_future - offset:-offset])

        #Processo inverso de normalização, gerando as amostras com seus valores originais
        y_pred_future = scale2.inverse_transform(predictions_future)
        PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['T (degC)']).set_index(pd.Series(datelist_future))

        #Definição das métricas e parâmetros
        mlflow.log_metric("MSE", val.history['loss'][-1])
        mlflow.log_param("Look Back", look_back)
        mlflow.log_param("Future Samples", n_future)
        mlflow.log_param("Offset", offset)
        mlflow.log_param("Epochs", epochs)

        #Inicia a interface do MLflow
        subprocess.run(["mlflow", "ui"])
