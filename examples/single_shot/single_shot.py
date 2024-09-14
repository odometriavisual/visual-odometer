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
