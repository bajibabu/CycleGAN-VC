import math
import numpy as np
import tensorflow as tf
from scipy import signal
import librosa
import librosa.filters
import soundfile


_mel_basis = None
_mel_basis_inv = None


cfg = {
    "num_mels": 80,
    "num_freq": 1025,
    "sample_rate": 16000,
    "frame_length_ms": 50,
    "frame_shift_ms": 12.5,
    "preemphasis": 0.97,
    "min_level_db":-100,
    "ref_level_db": 20,
    
    "griffin_lim_iters": 100,
    "power": 1.2
}

def load_wav(path):
    x, fs = soundfile.read(path)
    assert(fs == cfg['sample_rate'])
    return x


def save_wav(wav, path):
    soundfile.write(path, wav, cfg['sample_rate'])

def preemphasis(x):
    return signal.lfilter([1, -cfg['preemphasis']], [1], x)


def inv_preemphasis(x):
    return signal.lfilter([1], [1, -cfg['preemphasis']], x)


def spectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(np.abs(D)) - cfg['ref_level_db']
    return _normalize(S)


def inv_spectrogram(spectrogram):
    '''Converts spectrogram to waveform using librosa'''
    S = _db_to_amp(_denormalize(spectrogram) +
                   cfg['ref_level_db'])  # Convert back to linear
    # Reconstruct phase
    return inv_preemphasis(_griffin_lim(S ** cfg['power']))


def inv_spectrogram_tensorflow(spectrogram):
    '''Builds computational graph to convert spectrogram to waveform using TensorFlow.

    Unlike inv_spectrogram, this does NOT invert the preemphasis. The caller should call
    inv_preemphasis on the output after running the graph.
    '''
    S = _db_to_amp_tensorflow(_denormalize_tensorflow(
        spectrogram) + cfg['ref_level_db'])
    return _griffin_lim_tensorflow(tf.pow(S, cfg['power']))


def inv_mel_spectrogram(mel_spectrogram):
    global _mel_basis
    MS = _denormalize(mel_spectrogram) + cfg['ref_level_db']
    MS = _db_to_amp(MS)
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    S = _mel_to_linear(MS, _mel_basis)  # convet to linear scale
    y = _griffin_lim(S)
    y = inv_preemphasis(y)
    return y


def melspectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - cfg['ref_level_db']
    return _normalize(S)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
    window_length = int(cfg['sample_rate'] * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = _db_to_amp(threshold_db)
    for x in range(hop_length, len(wav) - window_length, hop_length):
        if np.max(wav[x:x+window_length]) < threshold:
            return x + hop_length
    return len(wav)


def _griffin_lim(S):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(cfg['griffin_lim_iters']):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y


def _griffin_lim_tensorflow(S):
    '''TensorFlow implementation of Griffin-Lim
    Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
    '''
    with tf.variable_scope('griffinlim'):
        # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
        S = tf.expand_dims(S, 0)
        S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
        y = _istft_tensorflow(S_complex)
        for i in range(cfg['griffin_lim_iters']):
            est = _stft_tensorflow(y)
            angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
            y = _istft_tensorflow(S_complex * angles)
        return tf.squeeze(y, 0)


def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_tensorflow(signals):
    n_fft, hop_length, win_length = _stft_parameters()
    return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)


def _istft_tensorflow(stfts):
    n_fft, hop_length, win_length = _stft_parameters()
    return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)


def _stft_parameters():
    n_fft = (cfg['num_freq'] - 1) * 2
    hop_length = int(cfg['frame_shift_ms'] / 1000 * cfg['sample_rate'])
    win_length = int(cfg['frame_length_ms'] / 1000 * cfg['sample_rate'])
    return n_fft, hop_length, win_length


# Conversions:

def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _mel_to_linear(melspec, _mel_basis):
    global _mel_basis_inv
    if _mel_basis_inv is None:
        _mel_basis_inv = np.linalg.pinv(_mel_basis)
        if np.any(np.isnan(_mel_basis_inv)) or np.any(np.isinf(_mel_basis_inv)):
            print('There is a problem with mel inverse')
            raise
    return np.dot(_mel_basis_inv, melspec)


def _build_mel_basis():
    n_fft = (cfg['num_freq'] - 1) * 2
    return librosa.filters.mel(cfg['sample_rate'], n_fft, n_mels=cfg['num_mels'])


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _db_to_amp_tensorflow(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


def _normalize(S):
    return np.clip((S - cfg['min_level_db']) / -cfg['min_level_db'], 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -cfg['min_level_db']) + cfg['min_level_db']


def _denormalize_tensorflow(S):
    return (tf.clip_by_value(S, 0, 1) * -cfg['min_level_db']) + cfg['min_level_db']
