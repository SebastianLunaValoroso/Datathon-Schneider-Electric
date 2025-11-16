def warn(*args, **kwargs): #per evitar warnings molestos de sklearn
    pass
import warnings
warnings.warn = warn
from funcions_modelitzacio import pd, RandomForestClassifier
import numpy as np
import shap #type:ignore



def create_tree_model_explainer(tree_model:RandomForestClassifier,x_train:pd.DataFrame)->shap.TreeExplainer:
    """Crea un Tree Model Explainer i el retorna"""
    return shap.TreeExplainer(tree_model, x_train)


def get_shap_values(tree_explainer:shap.TreeExplainer,x_train:pd.DataFrame)->shap.Explanation:
    """Retorna els shap values de tree_explaine amb x_train
    
    :param: x_train: han de ser les dades amb les que s'ha entrenat el model
    """
    return tree_explainer(x_train,check_additivity=False)

def plot_beeswarm(shap_values:shap.Explanation,maxim_display:int=16)->None:
    """Dibuixa un beeswarm plot dels shap_values amb max_display
    
    :param: x_train: han de ser les dades amb les que s'ha entrenat el model
    """
    shap.plots.beeswarm(shap_values[:,:,1], max_display=maxim_display)
