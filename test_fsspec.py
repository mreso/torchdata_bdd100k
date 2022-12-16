import os
import fsspec
from torchvision.io import decode_jpeg
import torch
import json
import asyncio

CONNECTION_STRING=os.environ['CONNECTION_STRING']
storage_options={'connection_string': CONNECTION_STRING}

def open_read_decode(h):
    with h.open() as f:
        return decode_jpeg(torch.frombuffer(f.read(), dtype=torch.uint8))

handle = fsspec.open('abfs://bdd100k/images/100k/train/27aecec2-52d5d1d0.jpg', **storage_options)
print(open_read_decode(handle).shape)

with open('file_list.json', 'r') as f:
    files = json.load(f)


def load_batch_sync(my_files):
    fh = fsspec.open_files([f'abfs://{f}' for f in my_files], **storage_options)

    for f in fh:
        print(open_read_decode(f).shape)


def load_batch_async(my_files):
    fs = fsspec.filesystem("abfs", **storage_options)
    # session = await fs.set_session()  # creates client
    res = fs.cat(my_files)  # fetches data concurrently

    # fh = fsspec.open_files([f'abfs://{f}' for f in my_files], **storage_options)

    # tasks = (asyncio.create_task(open_read_decode(f)) for f in fh)
    # res = await asyncio.gather(*tasks)

    for r in res: 
        print(r)

    # with f.open() as ff:
    #     print(decode_jpeg(torch.frombuffer(ff.read(), dtype=torch.uint8)).shape)

load_batch_sync(files[:10])
load_batch_async(files[:10])

# fs = fsspec.open('abfs://bdd100k/images/100k/train/', **storage_options).filesystem
# with handle.open() as f:
#     print(decode_jpeg(torch.frombuffer(f.read(), dtype=torch.uint8)).shape)

# async def work_coroutine(my_files):
#     fs = fsspec.filesystem("abfs", asynchronous=True, **storage_options)
#     # session = await fs.set_session()  # creates client
#     out = await fs.cat(my_files)  # fetches data concurrently
#     # await session.close()  # explicit destructor

# print(asyncio.run(work_coroutine(files[:10])))