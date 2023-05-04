import numpy as np

class Softmax:
  def __init__(self, var_pesos, nodos):
    #Igual que antes, dividimos para que los pesos no sean muy grandes ni muy pequeños al inicio
    self.pesos = np.random.randn(var_pesos, nodos) / var_pesos  
    self.biases = np.zeros(nodos)

  def pasada(self, entrada):
    '''
    Hace una pasada y devuelve los porcentajes de que sea cada numero
    '''
    entrada = entrada.flatten()

    totales = np.dot(entrada, self.pesos) + self.biases
    exp = np.exp(totales)
    return exp / np.sum(exp, axis=0)