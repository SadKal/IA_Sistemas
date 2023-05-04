import numpy as np
import mnist as mn

class MaxPool:
  #Funcion para hacer pooling

  def iterar_regiones_imagen(self, imagen):
    h, w, _ = imagen.shape
    new_h = h // 2
    new_w = w // 2

    for i in range(new_h):
      for j in range(new_w):  
        im_region = imagen[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
        yield im_region, i, j

  def pasada(self, input):
    '''
    Devuelve un array 3d con la mitad de tamaño del original
    '''
    h, w, num_filtros = input.shape
    salida = np.zeros((h // 2, w // 2, num_filtros))

    for im_region, i, j in self.iterar_regiones_imagen(input):
      salida[i, j] = np.amax(im_region, axis=(0, 1))

    return salida