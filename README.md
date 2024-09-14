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


vid = cv2.VideoCapture('https://example/stream.mjpg')
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH),
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

odometer = VisualOdometer(img_size=(width, height))
odometer.calibrate(new_xres=1.0, new_yres=1.0)
odometer.save_config('./')

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
from visual_odometer import VisualOdometer

def load(filename):
    from PIL import Image, ImageOps
    import numpy as np

    img_array_rgb = Image.open(filename)
    img_grayscale = ImageOps.grayscale(img_array_rgb)
    img_array = np.asarray(img_grayscale)

    return img_array

img0 = load('./img.png') # image at t = t₀
img1 = load('./img_translated.png') # image at t = t₀ + Δt

odometer = VisualOdometer(img_size=img0.shape)
odometer.save_config('./')

odometer.calibrate(new_xres=1.0, new_yres=1.0)
dx, dy = odometer.estimate_displacement_between(img0, img1)

print(f'Displacement estimate: x = {dx}, y = {dy}')
```
