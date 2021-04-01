import pickle
import numpy as np
import pandas as pd


class DefaultModel:
    """
    Descripción: Modelo para predecir si el mes que viene el cliente no pagará a partir
                 de los datos transaccionales de 6 meses atrás.

    df: Dataframe de pandas con los datos del cliente.
    xgb: Modelo XGBoost.
    cols: Atributos que utiliza el modelo para hacer la predicción.
    scaler: Modelo para estandarizar los datos RobustScaler.

    outlier_cat: Función para quitar valores atípicos de las variables categóricas.
    outlier_con: Función para quitar valres atípicos de las variables continuas.
    feature_engineering: Funciónn para crear nuevas características a partir de los datos del cliente.
    predict: Función para regresar la predicción del modelo.

    Ejemplo:
    obj1 = DefaultModel(dataframe)
    obj1.predict()
    """

    def __init__(self, df):
        """
        Descripción: Inicializa las variables necesarias para la prediccón.
        """
        self.df = df.copy()
        self.feature_engineering()
        self.outliers_cat()
        self.outliers_con()
        filename = 'Pickles Files/XGB.sav'
        with open(filename, 'rb') as file:
            self.xgb = pickle.load(file)
        filename = 'Pickles Files/cols.sav'
        with open(filename, 'rb') as file:
            self.cols = pickle.load(file)
        filename = 'Pickles Files/robustscaler_X.sav'
        with open(filename, 'rb') as file:
            self.scaler = pickle.load(file)

    def outliers_cat(self):
        """
        Descrpción: Elimina los valores atípicos de las variables EDUCACION, MARRIAGE
                    y los estatus de los pagos PAY_0, PAY_2, etc.
        """
        self.df.loc[(self.df.EDUCATION > 4) | (self.df.EDUCATION == 0), "EDUCATION"] = 4
        self.df.loc[self.df.MARRIAGE == 0, "MARRIAGE"] = 3
        for i in ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:
            self.df.loc[self.df[i].isin([3, 4, 5, 6, 7, 8, 9]), i] = 2

    def outliers_con(self):
        '''
        Descripción: Remueve valores atípicos de los datos del cliente.
        self.df: Pandas dataframe con los datos del cliente.
        q99: Aquí se guarda el percentil 99 de cada una de las variables y se agrega
             un límite superior.
        '''
        for i in ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
                  'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4',
                  'PAY_AMT5', 'PAY_AMT6']:
            filename = f'Pickles Files/quantile99_{i}.sav'
            with open(filename, 'rb') as f:
                q99 = pickle.load(f)
            self.df.loc[self.df[i] > q99, i] = q99

    def feature_engineering(self):
        '''
        Descripción: Crea nuevas características a partir de los datos transaccionales del cliente.
                     Genera agregaciónes a partir de cada nueva variable además de las variables
                     PAY_AMT y BILL_AMT, del mes 6 en adelante agrupa los atributos iniciando con 2
                     y agregado uno más cada vez hasta tener todos los meses juntos.
                     Las agregaciones son ["mean", "min", "max", "median", "sum", "std"].
        '''
        for i in range(1, 7):
            self.df[f"RATE_REMINDER_LIMIT-BILL_LIMIT{i}"] = (self.df["LIMIT_BAL"] - self.df[f"BILL_AMT{i}"]) / self.df[
                "LIMIT_BAL"]
            self.df[f"RATE_REMINDER_LIMIT-PAY_LIMIT{i}"] = (self.df["LIMIT_BAL"] - self.df[f"PAY_AMT{i}"]) / self.df[
                "LIMIT_BAL"]
            self.df[f'RATE_BILL_LIMIT{i}'] = self.df[f'BILL_AMT{i}'] / self.df['LIMIT_BAL']
            self.df[f"RATE_PAYAMT_LIMIT{i}"] = self.df[f"PAY_AMT{i}"] / self.df["LIMIT_BAL"]
            self.df[f"RATE_PAY_BILL_AMT{i}"] = (self.df[f"PAY_AMT{i}"] / self.df[f"BILL_AMT{i}"]).replace(
                [np.inf, -np.inf, np.nan], 0)
            self.df[f"DIF_BILL_PAY{i}"] = self.df[f"BILL_AMT{i}"] - self.df[f"PAY_AMT{i}"]

        for i in ['PAY_AMT', 'BILL_AMT', 'RATE_REMINDER_LIMIT-BILL_LIMIT', 'RATE_REMINDER_LIMIT-PAY_LIMIT',
                  'RATE_BILL_LIMIT', 'RATE_PAY_BILL_AMT', 'DIF_BILL_PAY', 'RATE_PAYAMT_LIMIT']:
            for agg_f in ["mean", "min", "max", "median", "sum", "std"]:
                for j in range(6, 0, -1):
                    if (j == 1):
                        continue
                    self.df[f"{i}_{agg_f}_1_{j}"] = self.df[[i + f'{x}' for x in range(6, j - 2, -1)]].aggregate(agg_f,
                                                                                                                 axis=1)

    def predict(self):
        """
        Descripción: Genera la predicción del modelo.
        val_pred: Es el valor o valores de la prediccón.
        return:
            numpy array si tiene más de una prediccón.
            Int si tiene una única predicción.
        """
        val_pred = self.xgb.predict(pd.DataFrame(self.scaler.transform(self.df[self.cols]), columns=self.cols))
        if(len(val_pred) == 1):
            return val_pred[0]
        elif(len(val_pred) > 1):
            return val_pred