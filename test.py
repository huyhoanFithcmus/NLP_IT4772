import numpy as np
import tensorflow as tf
from pyvi import ViTokenizer, ViPosTagger
from load_data import word2idx
import re
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('data/wiki.vi.vec')
model["ngon"]