import cv2
import os 
import numpy as np

dataPath = 'C:/Users/lbpf2/Desktop/Proyectos/sistema-asistencias_JG/Data'
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)
labels = []
facesData = []
label = 0

# tamaño objetivo (coincide con 150x150 usado en el script de captura))
target_size = (150, 150)

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo Imagenes de:', nameDir)

    for fileName in os.listdir(personPath):
        img_path = personPath + '/' + fileName
        image = cv2.imread(img_path, 0)
        if image is None:
            print('WARNING: no se pudo leer:', img_path)
            continue

        # redimensionar si es necesario
        if image.shape != (target_size[1], target_size[0]):
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)

        print('Rostro:', nameDir + '/' + fileName, '->', image.shape)
        facesData.append(image)
        labels.append(label)

    label += 1

if len(facesData) == 0:
    raise SystemExit('No hay imágenes válidas para entrenar. Verifica la carpeta Data.')

# crea el recognizer (asegúrate de tener opencv-contrib-python instalado)
try:
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
except AttributeError:
    raise SystemExit("cv2.face no disponible. Instala opencv-contrib-python:\npython -m pip install opencv-contrib-python")

# Entrenamiento
print('Entrenando... (imagenes:', len(facesData), 'labels:', len(labels), ')')
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido - El Archivo creado sacarlo de la carpeta al momento de actualizar el Git
face_recognizer.write('modeloEigenFace.xml')
print('Modelo Almacenado......')