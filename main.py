"""匯入套件"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""讀取辨識物件名稱"""
classes = None
with open('Yolo/coco.names', 'r') as f:
 classes = [line.strip() for line in f.readlines()]

"""讀取預先訓練好的官方模型"""
net = cv2.dnn.readNet('Yolo/yolov3-tiny.weights', 'Yolo/yolov3-tiny.cfg')
#net = cv2.dnn.readNet('Yolo/yolov3-608.weights', 'Yolo/yolov3-608.cfg')

"""開始偵測"""
cap = cv2.VideoCapture(0)
while True:
  # 取得圖片
  ret , image = cap.read()
  #image = plt.imread('data/a.jpg')

  # 辨識
  net.setInput(cv2.dnn.blobFromImage(image, 0.00412, (320,320), (0,0,0), True, crop=False))
  #net.setInput(cv2.dnn.blobFromImage(image, 0.00412, (608,608), (0,0,0), True, crop=False))
  layer_names = net.getLayerNames()
  output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
  outs = net.forward(output_layers)

  class_ids = []
  confidences = []
  boxes = []
  Width = image.shape[1]
  Height = image.shape[0]
  for out in outs:
      for detection in out:
          scores = detection[5:]
          class_id = np.argmax(scores)
          confidence = scores[class_id]
          if confidence > 0.1:
              center_x = int(detection[0] * Width)
              center_y = int(detection[1] * Height)
              w = int(detection[2] * Width)
              h = int(detection[3] * Height)
              x = center_x - w / 2
              y = center_y - h / 2
              class_ids.append(class_id)
              confidences.append(float(confidence))
              boxes.append([x, y, w, h])
          pass
      pass
  pass

  indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
  # 辨識到的結果畫出來
  for i in indices:
      i = i[0]
      box = boxes[i]
      if class_ids[i]==0:
          label = str(classes[class_id]) 
          cv2.rectangle(image, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (255, 0, 0), 2)
          cv2.putText(image, label, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
  
  # 顯示
  cv2.imshow("video",image)
  
  # 如果按q就離開
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  #plt.imshow(image)
  
pass