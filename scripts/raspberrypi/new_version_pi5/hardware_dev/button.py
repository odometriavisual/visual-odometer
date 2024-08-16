import time
import RPi.GPIO as GPIO


  
GPIO.setmode(GPIO.BOARD)
  

GPIO.setup(13, GPIO.IN)
GPIO.setup(18, GPIO.IN)


  
while(1):
    
    if GPIO.input(18) == True:
        print("1")
        time.sleep(0.5)
    if GPIO.input(13) == True:
        print("2")
        time.sleep(0.5)
    
    