'''
Archivo: statistic.py
Descripción: Archivo para realizar un análisis estadístico
    sobre las variables del dataset 'penguins_size.csv'.
Autora: Andrea Medina Rico
'''
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns
import numpy as np

class Visualization:
    def __init__(self):
        pass

    def correlation_matrix(self, data):
        corr_matrix = data.corr()
        plt.figure(figsize = (10, 8))
        sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', fmt = ".2f")
        plt.title('Correelation Matrix')
        plt.show()

    def histogram(self, data, col):
        plt.figure(figsize=(10, 6))
        sns.histplot(data[col], bins=30, kde=True)
        plt.title(f'{col} histogram')
        plt.xlabel(col)
        plt.ylabel('Frecuency')
        plt.show()