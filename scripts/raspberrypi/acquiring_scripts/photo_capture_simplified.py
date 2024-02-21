from picamera2 import Picamera2
picam2 = Picamera2()
picam2.set_controls({"ExposureTime": 500, "AnalogueGain": 1.0})
picam2.start_and_capture_files("images/image{:d}.jpg", initial_delay=3, delay=0.5, num_files=300)
