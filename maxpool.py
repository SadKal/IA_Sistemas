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

  def pasada(self, entrada):
    '''
    Devuelve un array 3d con la mitad de tamaño del original
    '''
    self.ultima_entrada = entrada

    h, w, num_filtros = entrada.shape
    salida = np.zeros((h // 2, w // 2, num_filtros))

    for im_region, i, j in self.iterar_regiones_imagen(entrada):
      salida[i, j] = np.amax(im_region, axis=(0, 1))

    return salida
  
  def  pasada_atras(self, d_L_d_out):

    d_L_d_input = np.zeros(self.ultima_entrada.shape)

    for im_region, i, j in self.iterar_regiones_imagen(self.ultima_entrada):
      h, w, f = im_region.shape
      amax = np.amax(im_region, axis=(0, 1))

      for i2 in range(h):
        for j2 in range(w):
          for f2 in range(f):
            
            if im_region[i2, j2, f2] == amax[f2]:
              d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

    return d_L_d_input