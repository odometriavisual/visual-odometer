import numpy as np
from virtualencoder.visualodometry.score_focus import score_teng
from virtualencoder.visualodometry.utils import get_imgs

score = 0
n = 100
imgs = get_imgs(n=n, data_root="data/DATASETS-21.02.2024/-9/LONG/2024-02-21_11-13-59")
score_teng_list = []

for i, img in enumerate(imgs):
    score_teng_list.append(score_teng(img))

print(np.average(score_teng_list))