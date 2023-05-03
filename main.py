import numpy as np
import mnist as mn
from softmax import Softmax 

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

def pasada(imagen, etiqueta):
  '''
  Hace una pasada y saca los porcentajes
  '''
  # Se pasa la imagen de 0-255 a -0.5 - 0.5. Por lo que se ve es costumbre hacerlo
  salida = conv.pasada((imagen / 255) - 0.5)
  salida = pool.pasada(salida)
  salida = softmax.pasada(salida)

  # Calcula la perdida y la precision
  perdida = -np.log(salida[etiqueta])
  if (np.argmax(salida) == etiqueta):
    precision=1
  else:
    precision=0

  return salida, perdida, precision

conv = Convolucion(8)                  
pool = MaxPool()                  
softmax = Softmax(13 * 13 * 8, 10) 

imagenes = mn.test_images()[:1000]
etiquetas = mn.test_labels()[:1000]

perdida = 0
num_correctos = 0
#Enumerate hace que los valores de i sean los de un array
#Zip hace que dos arrays se combinen---> [1,2] y [a,b] pasarian a ser [(1,a), (2,b)]
for i, (im, etiqueta) in enumerate(zip(imagenes, etiquetas)):
  _, l, precision = pasada(im, etiqueta)
  perdida += l
  num_correctos += precision

  # Print stats every 100 steps.
  if i % 100 == 99:
    print('[Paso %d] Ultimos 100 pasos: Perdida media: %.3f | Precision: %d%%' % (i + 1, perdida / 100, num_correctos))
    perdida = 0
    num_correctos = 0