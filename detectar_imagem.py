import cv2
import numpy as np

cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
foto_grupo = cv2.imread("test/img/foto-grupo-pessoas-teste.jpg")
foto_unica = cv2.imread("test/img/foto-teste.jpg")
foto = foto_grupo

foto_cinza = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)

faces = cascade.detectMultiScale(foto_cinza)

for (x, y, w, h) in faces:
    centro_x, centro_y = x + w // 2, y + h // 2

    distancia = int(max(w, h) * 0.25)  

    pontos = np.array([
        (centro_x, y - distancia),
        (x + w + distancia, centro_y),
        (centro_x, y + h + distancia),
        (x - distancia, centro_y)
    ], np.int32)

    cv2.polylines(foto, [pontos], isClosed=True, color=(255, 255, 255), thickness=2)

cv2.imshow("Faces", foto)
cv2.waitKey(0)
cv2.destroyAllWindows()
