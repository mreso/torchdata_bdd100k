import os
import json
import time
import torch
from torchdata.datapipes.iter import IterableWrapper, StreamReader
from torch.utils.data import DataLoader
from torchvision.io import decode_jpeg

CONNECTION_STRING=os.environ['CONNECTION_STRING']

def decode(d):
    # return decode_jpeg(torch.frombuffer(d[1], dtype=torch.uint8))
    # torch.frombuffer(d[1], dtype=torch.uint8).clone()
    return torch.zeros([3,10,10])

with open('file_list.json', 'r') as f:
    files = [f'abfs://{a}' for a in json.load(f)]

storage_options={'connection_string': CONNECTION_STRING}
# dp = IterableWrapper(['abfs://bdd100k/images/100k/train']).list_files_by_fsspec(**storage_options).sharding_filter().open_files_by_fsspec(mode='rb',**storage_options)

dp = IterableWrapper(files[:5000]).sharding_filter().open_files_by_fsspec(mode='rb',**storage_options)
dp = dp.read_from_stream().map(fn=decode)

dl = DataLoader(dp, batch_size=8, num_workers=16, prefetch_factor=4)

batch_times=[]

lb = time.time()
for c, _ in enumerate(dl):
    batch_times += [time.time() - lb]
    lb = time.time()
    if c == 0:
        st = time.time()
        print(f'Benchmark started')
    elif c % 100 == 0:
        print(f'Samples: {8*c} / 100000')
print(f'Time elapsed: {time.time() - st} seconds')

with open('batch_times.txt', 'w') as f:
    f.writelines('\n'.join([f'{t}' for t in batch_times]))

