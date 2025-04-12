import cv2
import face_recognition


img1=cv2.imread("imgs/bhaai.jpg")
rbg_img=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img_encode=face_recognition.face_encodings(rbg_img)[0]

img2=cv2.imread("imgs/allu-arjun.jpg")
rbg_img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
img_encode2=face_recognition.face_encodings(rbg_img2)[0]




result=face_recognition.compare_faces([img_encode],img_encode2)
print("Result:",result)

cv2.imshow("modi",img1)
cv2.imshow("modi2",img2)
cv2.waitKey(0)