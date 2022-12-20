import os
import json
import time
import torch
import numpy as np
from torchdata.datapipes.iter import IterableWrapper, StreamReader
from torch.utils.data import DataLoader
from torchvision.io import decode_jpeg

CONNECTION_STRING=os.environ['CONNECTION_STRING']

def decode(d):
    byte_tensor = torch.frombuffer(d[1], dtype=torch.uint8)
    return decode_jpeg(byte_tensor)
    # torch.frombuffer(d[1], dtype=torch.uint8).clone()
    # return torch.zeros([3,10,10])


def run(batch_size, num_workers):
    with open('file_list.json', 'r') as f:
        files = [f'abfs://{a}' for a in json.load(f)]

    storage_options={'connection_string': CONNECTION_STRING}
    # dp = IterableWrapper(['abfs://bdd100k/images/100k/train']).list_files_by_fsspec(**storage_options).sharding_filter().open_files_by_fsspec(mode='rb',**storage_options)

    dp = IterableWrapper(files).sharding_filter().open_files_by_fsspec(mode='rb',**storage_options)
    dp = dp.read_from_stream().map(fn=decode)

    batch_size = 64
    dl = DataLoader(dp, batch_size=batch_size, num_workers=num_workers, prefetch_factor=4)

    batch_times=[]

    lb = time.time()
    st = time.time()
    for c, _ in enumerate(dl):
        batch_times += [time.time() - lb]
        lb = time.time()
        if c % 100 == 0:
            print(f'Samples: {batch_size*c} / {len(files)}')

    images_per_second = len(files) / (time.time() - st)
    print(f'Time elapsed: {time.time() - st} seconds')
    print(f'Throughput: {images_per_second} images / seconds')
    return images_per_second

def main(batch_size=64, num_workers=(24,48, 64, 128), num_runs=5):
    results = {}
    for nw in num_workers:
        results[nw] = []
        for _ in range(num_runs):
            results[nw].append(run(batch_size, nw))
    print(results)
    print({k: np.mean(v) for k,v in results.items()})

if __name__ == '__main__':
    main()
