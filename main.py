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


imagenes_entrenamiento = mn.train_images()
etiquetas_entrenamiento = mn.train_labels()

conv = Convolucion(8)
output = conv.pasada(imagenes_entrenamiento[0])
print(output.shape)

train_images = mn.train_images()
train_labels = mn.train_labels()

conv = Convolucion(8)
pool = MaxPool()

output = conv.pasada(train_images[0])
output = pool.pasada(output)
print(output.shape) # (13, 13, 8)