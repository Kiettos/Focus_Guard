import cv2
cap = cv2.VideoCapture(r"DAiSEE\DataSet\Validation\4000221001.avi")
print("Opened:", cap.isOpened())
ret, frame = cap.read()
print("Read frame:", ret)