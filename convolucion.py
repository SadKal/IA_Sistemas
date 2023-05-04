import numpy as np
import mnist as mn

class Convolucion:

  def __init__(self, num_filtros):
    self.num_filtros = num_filtros

    #Con el numpy random llenamos un array de 3 dimensiones con filtros de numeros aleatorios
    self.filtros = np.random.randn(num_filtros, 3, 3) / 9

  def iterar_regiones_imagen(self, imagen):
    '''
    Crea toda las combinaciones de 3x3 de los filtros
    '''
    #h es el alto de la imagen y w el ancho
    h, w = imagen.shape

    for i in range(h - 2):
      for j in range(w - 2):
        im_region = imagen[i:(i + 3), j:(j + 3)]
        yield im_region, i, j

  def pasada(self, input):
    '''
    Hace una pasada a la red neuronal
    '''
    h, w = input.shape
    salida = np.zeros((h - 2, w - 2, self.num_filtros))

    #Por cada región que genere la funcion anterior
    for im_region, i, j in self.iterar_regiones_imagen(input):
      #Genera la convolución de los píxeles que se encuentren en esa región
      salida[i, j] = np.sum(im_region * self.filtros, axis=(1, 2))

    return salida