#-*- coding:utf-8 -*-

import codecs
import os
import collections
from six.moves import cPickle,reduce,map
import numpy as np

BEGIN_CHAR = '^'
END_CHAR = '$'
UNKNOWN_CHAR = '*'
MAX_LENGTH = 100

class TextLoader():

    def __init__(self, batch_size, max_vocabsize=3000, encoding='utf-8'):
        self.batch_size = batch_size
        self.max_vocabsize = max_vocabsize
        self.encoding = encoding

        data_dir = './data'

        input_file = os.path.join(data_dir, "poems.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        def handle_poem(line):
            line = line.replace(' ','')
            if len(line) >= MAX_LENGTH:
                index_end = line.rfind(u'。',0,MAX_LENGTH)
                index_end = index_end if index_end > 0 else MAX_LENGTH
                line = line[:index_end+1]
            return BEGIN_CHAR+line+END_CHAR

        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            lines = list(map(handle_poem,f.read().strip().split('\n')))

        counter = collections.Counter(reduce(lambda data,line: line+data,lines,''))
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        chars, _ = zip(*count_pairs)
        self.vocab_size = min(len(chars),self.max_vocabsize - 1) + 1
        self.chars = chars[:self.vocab_size-1] + (UNKNOWN_CHAR,)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        unknown_char_int = self.vocab.get(UNKNOWN_CHAR)
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        get_int = lambda char: self.vocab.get(char,unknown_char_int)
        lines = sorted(lines,key=lambda line: len(line))
        self.tensor = [ list(map(get_int,line)) for line in lines ]
        with open(tensor_file,'wb') as f:
            cPickle.dump(self.tensor,f)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        with open(tensor_file,'rb') as f:
            self.tensor = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

    def create_batches(self):
        self.num_batches = int(len(self.tensor) / self.batch_size)
        self.tensor = self.tensor[:self.num_batches * self.batch_size]
        unknown_char_int = self.vocab.get(UNKNOWN_CHAR)
        self.x_batches = []
        self.y_batches = []

        for i in range(self.num_batches):
            from_index = i * self.batch_size
            to_index = from_index + self.batch_size
            batches = self.tensor[from_index:to_index]
            seq_length = max(map(len,batches))
            xdata = np.full((self.batch_size,seq_length),unknown_char_int,np.int32)
            for row in range(self.batch_size):
                xdata[row,:len(batches[row])] = batches[row]
            ydata = np.copy(xdata)
            ydata[:,:-1] = xdata[:,1:]
            self.x_batches.append(xdata)
            self.y_batches.append(ydata)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
