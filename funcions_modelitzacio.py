from sklearn.ensemble import RandomForestClassifier #type:ignore
from sklearn.model_selection import train_test_split #type:ignore
import pandas as pd #type:ignore


def split_variables_and_answer(index_primera_variable:int,index_darrera_variable:int,index_reposta:int,mida_model_perc:float,dades_csv:str)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Retorna 4 DataFrames (Variables to Train, Variables to Check, Answers to Train, Answers to Check)
    
    :param: index_primera_variable, index_darrera_variable, index_resposta: serveixen per seleccionar les columnes de les 
    variables de dades de la manera seguënt dades[index_primera_variable,index_darrera_variable+1] son les columnes de les variables sense la resposta ni columnes que no contribuixen a predirla. dades[index_reposta] selecciona la variables resposta

    :param: mida_model_perc: El percentage de les filas de dades que s'utilitzaran per el model, la part restant servirán per un check del model
    """
    dades_var:pd.DataFrame=pd.read_csv("dataset.csv", usecols=range(index_primera_variable,index_darrera_variable+1))
    dades_resp:pd.DataFrame=pd.read_csv("dataset.csv", usecols = [index_reposta])
    return train_test_split(dades_var, dades_resp, test_size = 1-mida_model_perc, random_state = 1)

def build_model(num_estimadors:int,x_train:pd.DataFrame,y_train:pd.DataFrame,x_test:pd.DataFrame,y_test:pd.DataFrame)->tuple[RandomForestClassifier,float]:
    """Crea i entrena un model de tipus RandomForestClassifier a partir de x_train i y_train, desprès calcula la seva performance amb x_test i y_test.
    
    Retorna el model creat i el score de performance
    
    :param: num_estimadors: es tracta del nombre de arbres al model de RandomForestClassifier"""
    model_rf:RandomForestClassifier = RandomForestClassifier(n_estimators=num_estimadors)
    model_rf.fit(x_train,y_train)
    performance_score:float = model_rf.score(x_test,y_test)
    return (model_rf,performance_score)