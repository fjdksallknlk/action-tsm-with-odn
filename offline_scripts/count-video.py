import os

from tqdm import tqdm

p1 = '/home/shuyu/data/Kinetics/frames_tvl1/'

cates = os.listdir(p1)

num = 0
with open('./filelist-Kinetics.txt', 'w') as fh:
    for c in cates:
        p2 = os.path.join(p1, c)
        videos = os.listdir(p2)
        num += len(videos)
        for v in tqdm(videos):
            fh.write("{}/{}\n".format(c, v))
print(num)


