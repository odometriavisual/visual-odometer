import cv2
import time

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

cap = cv2.VideoCapture('http://10.42.0.95:7123/stream.mjpg')

while True:
  ret, frame = cap.read()
  new_frame_time = time.time()
  cv2.imshow('Video', frame)
  fps = 1 / (new_frame_time - prev_frame_time)
  prev_frame_time = new_frame_time
  print(fps)
  if cv2.waitKey(1) == 27:
      exit(0)

