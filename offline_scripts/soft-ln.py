import os

from tqdm import tqdm

mode = 'train'
# mode = 'val'
sPath = '/ssd_datasets/shuyu/videos/Kinetics/frames_tvl1/'
tPath = '/home/shuyu/data/Kinetics/frames_tvl1/'

p1 = os.path.join(sPath, mode)
cates = os.listdir(p1)

for c in cates:
    p2 = os.path.join(p1, c)
    videos = os.listdir(p2)
    p3 = os.path.join(tPath, c)
    if not os.path.exists(p3):
        os.makedirs(p3)
    for v in tqdm(videos):
        s = os.path.join(p2, v)
        t = os.path.join(p3, v)
        cmd = 'ln -s {} {}'.format(s, t)
        try:
            # print(cmd)
            os.system(cmd)
        except PermissionError:
            print('No permission!')


