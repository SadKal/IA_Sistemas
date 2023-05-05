import numpy as np
import mnist as mn
from softmax import Softmax 
from maxpool import MaxPool
from convolucion import Convolucion

#Para probar nosotros los numeros
from PIL import Image
import cv2  
import matplotlib.pyplot as plt

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

def entrenar(im, etiqueta, lr=.005):
  '''
  Hace un entrenamiento con la imagen y etiqueta que se le pase
  '''
  
  salida, perdida, precision = pasada(im, etiqueta)

  
  gradient = np.zeros(10)
  gradient[etiqueta] = -1 / salida[etiqueta]

  # Pasada hacia atras
  gradient = softmax.pasada_atras(gradient, lr)
  gradient = pool.pasada_atras(gradient)
  gradient = conv.pasada_atras(gradient, lr)

  return perdida, precision

conv = Convolucion(8)                  
pool = MaxPool()                  
softmax = Softmax(13 * 13 * 8, 10) 

imagenes_entrenar = mn.train_images()[:1000]
etiquetas_entrenar = mn.train_labels()[:1000]
imagenes_prueba = mn.test_images()[:1000]
etiquetas_prueba = mn.test_labels()[:1000]




for epoca in range(3):
# Seleccionamos imagenes aleatorias para entrenar
  permutation = np.random.permutation(len(imagenes_entrenar))
  imagenes_entrenar = imagenes_entrenar[permutation]
  etiquetas_entrenar = etiquetas_entrenar[permutation]
  perdida = 0
  correctas = 0
  print('EPOCA: ', epoca)
  for i, (im, etiqueta) in enumerate(zip(imagenes_entrenar, etiquetas_entrenar)):
    if i > 0 and i % 100 == 99:     
      print(
        '[Paso %d] Ultimos 100 pasos: Perdida media %.3f | Precision: %d%%' % (i + 1, perdida / 100,  correctas)
      )
      perdida = 0
      correctas = 0

    per, precision = entrenar(im, etiqueta)
    perdida += per
    correctas += precision

"""
perdida = 0
correctas = 0
for im, etiqueta in zip(imagenes_prueba, etiquetas_prueba):
  _, l, precision = pasada(im, etiqueta)
  perdida += l
  correctas += precision

  if(etiqueta==3):
    print(im)
  for elem in enumerate(_):
    print(elem)
  print(precision)
  print(etiqueta)

num_tests = len(imagenes_prueba)
print('Perdida:', perdida / num_tests)
print('Precision:', correctas / num_tests)
"""


#Probar con una imagen nuestra
img = r'C:\Users\huert\Desktop\DAW\Sistemas\red_neuronal\IA_Sistemas\img2.png'
test_image= cv2.imread(img, cv2.IMREAD_GRAYSCALE)

img_resized=cv2.resize(test_image, (28,28), interpolation=cv2.INTER_LINEAR)
img_resized=cv2.bitwise_not(img_resized)
#plt.imshow(img_resized,cmap='gray')
#plt.show()


salida, l, precision = pasada(img_resized, 2)
for elem in enumerate(salida):
  print(elem)
print('El numero es probablemente: ', np.argmax(salida))

