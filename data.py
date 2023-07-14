# -*- coding: utf-8 -*-
"""
Contains means to read, generate and handle data.

Created on Tue Oct  3 08:20:52 2017

Copyright (c) 2017, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import numpy as np
import os
import tensorflow
from natsort import natsorted 
from keras.utils import OrderedEnqueuer
import gc
from keras import backend as K
from skimage.measure import block_reduce

class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self,
                 data_path,
                 batch_size=32,
                 shuffle=True,
                 use_multiprocessing=False,
                 num_workers=4,
                 max_queue_size=10
                 ):

        self.data_path = data_path
        self.inputs = ["T1", "FLAIR"]
        self.outputs = ["Background", "WMH", "Other"]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_size = [256, 256]

        if data_path is None:
            raise ValueError('The data path is not defined.')

        if not os.path.isdir(data_path):
            raise ValueError(f'The data path ({repr(data_path)}) is not a directory.')

        self.file_list = []
        for i, ID in enumerate([self.data_path + s for s in
                          os.listdir(self.data_path)]):
            self.file_list.append(ID)

        self.file_idx = 0
        self.indexes = np.arange(len(self.file_list))

        if (self.shuffle):
            self.file_list.sort()
        else:
            self.file_list = natsorted(self.file_list)
        
        self.in_dims = [self.batch_size, self.img_size[0], self.img_size[1], 1]
        self.n_channels = 1

        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.enqueuer = OrderedEnqueuer(self, use_multiprocessing=use_multiprocessing)
        self.data_seq = None

        if (self.shuffle):
            self.on_epoch_end()

    def set_experiment(self, experiment):
        self.experiment = experiment

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.file_list)) / self.batch_size))

    def next_batch(self):
        if (self.data_seq is None):
            self.enqueuer.start(workers=self.num_workers, max_queue_size=self.max_queue_size)
            self.data_seq = self.enqueuer.get()

        return next(self.data_seq)

    def __getitem__(self, index):
        'Generate one batch of data'
        if (index > len(self) - 1):
            raise ValueError("Index is larger than the number of allowed batches")
        
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        self.temp_ID = [self.file_list[k] for k in indexes]
            
        # Generate data
        i, o = self.__data_generation(self.temp_ID)

        # Clear memory
        K.clear_session()
        gc.collect()
        
        return i, o

    def stop(self):
        'Updates indexes after each epoch'

        if (self.data_seq != None):
            self.enqueuer.stop()
            self.data_seq = None

            K.clear_session()
            gc.collect()
            
        self.on_epoch_end()

        
    def on_epoch_end(self):
        if self.shuffle is True:
            np.random.shuffle(self.indexes)
        

    #@threadsafe_generator
    def __data_generation(self, temp_list):
        import tensorflow as tf
        'Generates data containing batch_size samples'
        inputs = np.zeros((self.batch_size, self.img_size[0], self.img_size[1], len(self.inputs)))
        outputs = np.zeros((self.batch_size, self.img_size[0], self.img_size[1], len(self.outputs)))

        if (len(temp_list) != self.batch_size):
            raise ValueError("Batch size is not equal to the number of files in the list")


        for i, ID in enumerate(temp_list):
            with np.load(ID, allow_pickle=True) as npzfile:
                for idx in range(len(self.inputs)):
                    inputs[i, :, :, idx] = npzfile[self.inputs[idx]].astype(np.float32)

                for idx in range(len(self.outputs)):   

                    if (npzfile.files.__contains__(self.outputs[idx])):
                        mask = npzfile[self.outputs[idx]].astype(np.int16)
                        outputs[i, :, :, idx] = mask
                        
                npzfile.close()
        return inputs, outputs

if __name__ == "__main__":
    import doctest
    doctest.testmod()
