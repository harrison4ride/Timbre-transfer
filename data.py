import os
import numpy as np
import librosa
import tensorflow as tf
from config import hop_length,bins_per_octave

def slice_first_dim(array, slice_size):
    n_sections = int(np.floor(array.shape[1]/slice_size))
    has_last_mag = n_sections*slice_size < array.shape[1]

    last_mag = np.zeros(shape=(1, array.shape[0], slice_size, array.shape[2]))
    last_mag[:,:,:array.shape[1]-(n_sections*slice_size),:] = array[:,n_sections*int(slice_size):,:]
    
    if(n_sections > 0):
        array = np.expand_dims(array, axis=0)
        sliced = np.split(array[:,:,0:n_sections*slice_size,:], n_sections, axis=2)
        sliced = np.concatenate(sliced, axis=0)
        if(has_last_mag): # Check for reminder
            sliced = np.concatenate([sliced, last_mag], axis=0)
    else:
        sliced = last_mag
    return sliced

def slice_magnitude(mag, slice_size):
    magnitudes = np.stack([mag], axis=2)
    return slice_first_dim(magnitudes, slice_size)

def join_magnitude_slices(mag_sliced, target_shape):
    mag = np.zeros((mag_sliced.shape[1], mag_sliced.shape[0]*mag_sliced.shape[2]))
    for i in range(mag_sliced.shape[0]):
        mag[:,(i)*mag_sliced.shape[2]:(i+1)*mag_sliced.shape[2]] = mag_sliced[i,:,:,0]
    mag = mag[0:target_shape[0], 0:target_shape[1]]
    return mag

def add_hf(mag, target_shape):
    rec = np.zeros(target_shape)
    rec[0:mag.shape[0],0:mag.shape[1]] = mag
    return rec

def remove_hf(mag):
    return mag[0:int(mag.shape[0]/2), :]

def forward_cqt(audio,sr):
    cqt = librosa.cqt(audio, sr=sr,hop_length=hop_length, n_bins=8*bins_per_octave,bins_per_octave=bins_per_octave)
    mag, phase = librosa.magphase(cqt)
    return mag, phase

def inverse_cqt(mag, phase, sr):
    R = mag * phase
    audio = librosa.icqt(R, sr=sr,hop_length=hop_length,bins_per_octave=bins_per_octave)
    return audio

def forward_transform(audio, nfft=1024, normalize=True, crop_hf=True):
    window = np.hanning(nfft)
    S = librosa.stft(audio, n_fft=nfft, hop_length=int(nfft/2), window=window)
    mag, phase = librosa.magphase(S) #np.abs(S), np.angle(S)
    if(crop_hf):
        mag = remove_hf(mag)
    if(normalize):
        mag = 2 * mag / np.sum(window)
    return mag, phase

def inverse_transform(mag, phase, nfft=1024, normalize=True, crop_hf=True):
    window = np.hanning(nfft)
    if(normalize):
        mag = mag * np.sum(np.hanning(nfft)) / 2
    if(crop_hf):
        mag = add_hf(mag, target_shape=(phase.shape[0], mag.shape[1]))
    R = mag * phase
    audio = librosa.istft(R, hop_length=int(nfft/2), window=window)
    return audio

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, origin, target, base_path, batch_size=1, img_dim=(256,256,1)):
        self.img_dim = img_dim
        self.batch_size = batch_size

        self.base_path = base_path if(type(base_path) is list) else [base_path]

        self.origin = origin
        self.target = target
        self.filenames = self.__get_filenames()
        assert len(self.filenames) > 0, 'Filenames is empty' 

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]  # Generate indexes of the batch
        batch = self.__data_generation(filenames)                                    # Generate data
        return batch

    def get_empty_batch(self):
        batch = np.zeros((self.batch_size, *self.img_dim))
        return batch, batch

    def get_random_batch(self):
        random_idx = np.random.randint(self.__len__())
        return self.__getitem__(random_idx)

    def __data_generation(self, filenames):
        'Generates data containing batch_size samples'                                  
        x = np.empty((self.batch_size, *self.img_dim))
        y = np.empty((self.batch_size, *self.img_dim))
        # Generate data
        for i, filename in enumerate(filenames):
            x[i,] = np.load(filename)
            y[i,] = np.load(filename.replace(self.origin, self.target))
            
            # Now images should be scaled in the range [0,1]. Make them [-1,1]
            x[i,] = x[i,] * 2 - 1
            y[i,] = y[i,] * 2 - 1
        return x,y

    def __get_filenames(self):
        origin_filenames, target_filenames = [],[]
        for base_path in self.base_path:
            origin_temp = [os.path.join(base_path, self.origin, f) for f in os.listdir(os.path.join(base_path, self.origin))]
            target_temp = [os.path.join(base_path, self.target, f) for f in os.listdir(os.path.join(base_path, self.target))]
            origin_filenames += origin_temp
            target_filenames += target_temp
        if(len(origin_filenames) == len(target_filenames)):
            return origin_filenames
        return []

    def shuffle(self):
        np.random.shuffle(self.filenames)