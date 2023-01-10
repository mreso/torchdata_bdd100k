import asyncio
import fsspec
import json
import os
import time
import torch
import numpy as np
from torchdata.datapipes.iter import IterableWrapper, StreamReader
from torch.utils.data import DataLoader
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torchvision.io import decode_jpeg

CONNECTION_STRING=os.environ['CONNECTION_STRING']
storage_options={'connection_string': CONNECTION_STRING}

torch.multiprocessing.set_sharing_strategy('file_system')

def open_read(batch):
    fs = fsspec.filesystem("abfs", **storage_options)
    fh = fs.cat(batch)

    return fh

async def create_tensor(buffer):
    return torch.frombuffer(buffer, dtype=torch.uint8).clone()

async def do_work(buffer):
    return decode_jpeg(torch.frombuffer(buffer, dtype=torch.uint8))
    # return torch.frombuffer(buffer, dtype=torch.uint8).clone()

async def decode_async(batch):
    images = [await do_work(b) for _, b in batch.items()]
    return images

def decode(batch):
    images = asyncio.run(decode_async(batch))
    return images
    # return [torch.zeros([3,10,10]) for _ in range(len(batch))]

def run(batch_size, num_workers):
    with open('file_list.json', 'r') as f:
        files = [f'abfs://{a}' for a in json.load(f)]

    storage_options={'connection_string': CONNECTION_STRING}
    dp = IterableWrapper(files).sharding_filter().batch(batch_size).map(fn=open_read)
    
    dp = dp.map(fn=decode).collate()

    dl = DataLoader2(datapipe=dp, reading_service=MultiProcessingReadingService(num_workers=num_workers))

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
    main(batch_size=64,  num_workers=[64], num_runs=5)
