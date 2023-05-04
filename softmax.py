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
    self.forma_ultima_entrada = entrada.shape

    entrada = entrada.flatten()
    self.ultima_entrada = entrada


    totales = np.dot(entrada, self.pesos) + self.biases
    self.ultimos_totales = totales

    exp = np.exp(totales)
    return exp / np.sum(exp, axis=0)
  
  def pasada_atras(self, perdida_capa, ratio_aprendizaje):
    '''
    Hace una pasada hacia atras y devuelve la perdida que tiene la entrada de esta capa
    '''
    for i, perd in enumerate(perdida_capa):
      if perd == 0:
        continue

      t_exp = np.exp(self.ultimos_totales)

      S = np.sum(t_exp)

      d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
      d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

      d_t_d_w = self.ultima_entrada
      d_t_d_b = 1
      d_t_d_inputs = self.pesos
    
      d_L_d_t = perd * d_out_d_t
      
      d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
      d_L_d_b = d_L_d_t * d_t_d_b
      d_L_d_inputs = d_t_d_inputs @ d_L_d_t

      #Actualizamos pesos y desviaciones
      self.pesos -= ratio_aprendizaje * d_L_d_w
      self.biases -= ratio_aprendizaje * d_L_d_b
      return d_L_d_inputs.reshape(self.last_input_shape)