import cv2

cascade = cv2.CascadeClassifier("haarcascades\haarcascade_frontalface_default.xml")
foto_grupo = cv2.imread("test/img/foto-grupo-pessoas-teste.jpg")
foto_unica = cv2.imread("test/img/foto-teste.jpg")

foto_cinza = cv2.cvtColor(foto_unica, cv2.COLOR_BGR2GRAY)
faces = cascade.detectMultiScale(foto_cinza)

cv2.imshow("Faces", foto_unica) # Mostrando a foto na tela
cv2.waitKey()