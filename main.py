import numpy as np
import mnist as mn
from softmax import Softmax 
from maxpool import MaxPool
from convolucion import Convolucion

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

  if i % 100 == 99:
    print('[Paso %d] Ultimos 100 pasos: Perdida media: %.3f | Precision: %d%%' % (i + 1, perdida / 100, num_correctos))
    perdida = 0
    num_correctos = 0
