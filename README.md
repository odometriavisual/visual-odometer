# Visual Odometer

## Usage

### Sequencial usage

```python
import numpy as np
import cv2
from PIL import Image, ImageOps
from visual_odometer import VisualOdometer

def cv2_to_nparray_grayscale(frame):
    cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_array_rgb = Image.fromarray(cv2_img)
    img_grayscale = ImageOps.grayscale(img_array_rgb)
    img_array = np.asarray(img_grayscale)
    return img_array

odometer = VisualOdometer()
odometer.calibrate(new_xres=1.0, new_yres=1.0)
odometer.save_config('./odometer-config.json')

vid = cv2.VideoCapture('https://example/stream.mjpg')

_, frame = vid.read()
frame = cv2_to_nparray_grayscale(frame)

odometer.feed_image(frame)

x = y = 0

while True:
    _, frame = vid.read()
    frame = cv2_to_nparray_grayscale(frame)
    
    odometer.feed_image(frame)
    dx, dy = odometer.estimate_last_displacement()
    
    x += dx
    y += dy

    print(f'Total displacement estimate: x = {x}, y = {y}')

```

### Single shot

```python
import numpy as np
from visual_odometer import VisualOdometer

odometer = VisualOdometer()
odometer.calibrate(new_xres=1.0, new_yres=1.0)
odometer.save_config('./odometer-config.json')

img0 = np.load('./img0.jpg') # image at t = t₀
img1 = np.load('./img1.jpg') # image at t = t₀ + Δt

dx, dy = odometer.estimate_displacement_between(img0, img1)

print(f'Displacement estimate: x = {dx}, y = {dy}')
```
