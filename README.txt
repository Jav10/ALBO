# ALBO
ALBO Test

Modelo Default

Para poder usar el modelo se requiere un dataframe con al menos 1 registro,
con los siguientes datos del cliente:

['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 
 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 
 'PAY_AMT5', 'PAY_AMT6', 'SEX', 'EDUCATION', 'MARRIAGE','PAY_0', 'PAY_2',
 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

Importamos el módulo default e instanciamos la clase DefaultModel pasando como parámetro el
dataframe y hacemos la predicción, se puede guardar en una variable y utilizar 
como sea necesario.

Ejemplo:
    from default import DefaultModel

    obj1 = DefaultModel(dataframe)
    predict = obj1.predict()
