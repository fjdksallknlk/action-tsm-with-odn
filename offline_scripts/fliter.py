from tqdm import tqdm

mode = 'train'
# mode = 'test'

infile = '/ssd_datasets/shuyu/videos/Kinetics/prepared_list/{}_filelist_tsm'.format(mode)

data = []
with open('./filelist-Kinetics.txt', 'r') as fh:
    for ln in tqdm(fh):
        s = ln.strip()
        data.append(s)

with open('{}_list_tsm'.format(mode), 'w') as sfh:
    with open(infile, 'r') as fh:
        for ln in tqdm(fh):
            ln = ln.strip()
            name,_,_ = ln.split(' ')
            if name in data:
                sfh.write(ln + "\n")
    
