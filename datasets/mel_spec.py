import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio


def build_from_path(opt, tqdm=lambda x: x):
    with open(opt.txt, 'r') as fid:
        file_ids = [l.strip() for l in fid.readlines()]

    executor = ProcessPoolExecutor(max_workers=opt.num_workers)
    futures = []
    index = 1
    for file_id in file_ids:
        task = partial(_process_utterance, opt, file_id)
        futures.append(executor.submit(task))
        index += 1
    results = [future.result() for future in tqdm(futures)]


def _process_utterance(opt, file_id):
    if opt.mode == "analysis":
        # Load the wav file
        wav_path = '%s/%s.wav' % (opt.in_dir, file_id)
        wav = audio.load_wav(wav_path)
        # compute mel-spectrum
        mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
        out_file_path = '%s/%s.npz' % (opt.out_dir, file_id)
        feats = {}
        feats['mel_spec'] = mel_spectrogram.T
        np.savez(out_file_path, **feats)
    elif opt.mode == "synthesis":
        mel_spec_file = '%s/%s.npz' % (opt.in_dir, file_id)
        mel_spec = np.load(mel_spec_file)['mel_spec']
        wav = audio.inv_mel_spectrogram(mel_spec.T)
        wav_path = '%s/%s.wav' % (opt.out_dir, file_id)
        audio.save_wav(wav, wav_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', required=True, type=str,
                        help='text file contain file ids')
    parser.add_argument('--in-dir', required=True, type=str,
                        help='Directory contains the files')
    parser.add_argument('--out-dir', required=True, type=str,
                        help='output directory to save the files')
    parser.add_argument('--num-workers', default=4, type=int,
                        help='number of workers to run in parallel')
    parser.add_argument('--mode', required=True, type=str,
                        help='analysis|synthesis')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_args()

    if not os.path.isdir(opt.out_dir):
        os.makedirs(opt.out_dir)

    build_from_path(opt)

    # minimum sequence length in the nancy data is 57 (at 12.5 ms frameshift)
    # maximum sequence length in the nancy data is 1614 (at 12.5 ms fameshift)
    # number of samples have higher than the sequence length of 700 are 451 samples only (at 12.5 ms frameshift)
    # So we shoud remove the samples which have sequence length higher than 700 to save the
    # computational resources
