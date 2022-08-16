import os
from tqdm import tqdm

with open('../tools/kinetics_label_map.txt', 'r') as f:
    categories = f.readlines()
    categories = [c.strip().replace(' ', '_').replace('"', '').replace('(', '').replace(')', '').replace("'", '') for c in categories]
assert len(set(categories)) == 400
dict_categories = {}
with open('./kclassInd.txt', 'w') as fh:
    for i, category in enumerate(categories):
        dict_categories[category] = i
        fh.write("{} {}\n".format(category, i))

dataPath = '/ssd_datasets/shuyu/videos/Kinetics/frames_tvl1/all'
newDataPath = '/home/shuyu/data/Kinetics/frames_tvl1'

dir_names = os.listdir(dataPath)

for d in tqdm(dir_names):
    nd = d.strip().replace(' ', '_').replace('"', '').replace('(', '').replace(')', '').replace("'", '')
    if not os.path.exists(os.path.join(newDataPath, nd)):
        os.makedirs(os.path.join(newDataPath, nd))
    videos = os.listdir(os.path.join(dataPath, d))
    for v in tqdm(videos):
        nv = '{}_{}'.format(v, dict_categories[nd])
        s = os.path.join(dataPath, d, v)
        t = os.path.join(newDataPath, nd, nv)
        try:
            os.symlink(s, t)
        except PermissionError:
            print('No permission!')