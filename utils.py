import glob
import os
import time
import random
import numpy as np
import cv2
from tqdm import tqdm
import torch
from sklearn.utils import shuffle
from torch.utils.data import Dataset
import heapq
from collections import Counter


def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")

def rle_encode(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


class HuffmanNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    freq = Counter(data)
    heap = [HuffmanNode(sym, f) for sym, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(
            heap,
            HuffmanNode(None, a.freq + b.freq, a, b)
        )
    return heap[0]

def build_codebook(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    else:
        build_codebook(node.left, prefix + "0", codebook)
        build_codebook(node.right, prefix + "1", codebook)
    return codebook


"RLE-Huffman encode"
def rle_huffman_encode(mask):
    rle = rle_encode(mask)

    tree = build_huffman_tree(rle)
    codebook = build_codebook(tree)

    bitstream = "".join(codebook[x] for x in rle)

    return {
        "bitstream": bitstream,
        "codebook": codebook,
        "shape": mask.shape
    }
    
"RLE-Huffman dncode"
def rle_huffman_decode(encoded):
    bitstream = encoded["bitstream"]
    codebook = encoded["codebook"]
    shape = encoded["shape"]

    inv_codebook = {v: k for k, v in codebook.items()}

    rle = []
    buffer = ""
    for bit in bitstream:
        buffer += bit
        if buffer in inv_codebook:
            rle.append(inv_codebook[buffer])
            buffer = ""

    return rle_decode(rle, shape)
