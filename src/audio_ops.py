import numpy as np
import librosa
import soundfile as sf
from scipy import signal

def normalize_peak(input_path, output_path, target_db):
    y, sr = librosa.load(input_path, sr=None, mono=False)
    max_val = np.max(np.abs(y))
    if max_val > 0:
        target = 10 ** (target_db / 20.0)
        if max_val > target:
            y = y * (target / max_val)
    sf.write(output_path, y.T, sr)
    return output_path

def align_and_mix(file_paths, weights, sr):
    loaded = []
    min_len = None
    
    # Load all
    for f in file_paths:
        y, _ = librosa.load(f, sr=sr, mono=False)
        if y.ndim == 1: y = np.stack([y, y])
        curr_len = y.shape[-1]
        if min_len is None or curr_len < min_len:
            min_len = curr_len
        loaded.append(y)
    
    # Align based on first file
    ref = loaded[0][..., :min_len]
    mixed = np.zeros_like(ref)
    norm_weights = [w/sum(weights) for w in weights]

    for i, aud in enumerate(loaded):
        tgt = aud[..., :min_len]
        if i == 0:
            algn = tgt
        else:
            # Cross-correlation alignment
            corr = signal.fftconvolve(ref[0], tgt[0][::-1], mode='full')
            lag = np.argmax(corr) - (len(ref[0]) - 1)
            
            if abs(lag) > 2000: algn = tgt
            elif lag > 0: algn = np.hstack((np.zeros((2, lag)), tgt))[..., :min_len]
            elif lag < 0: algn = np.hstack((tgt[..., abs(lag):], np.zeros((2, abs(lag)))))[..., :min_len]
            else: algn = tgt
            
        mixed += algn * norm_weights[i]
        
    return mixed, min_len

def mix_weighted(path1, path2, w1, w2, sr):
    y1, _ = librosa.load(path1, sr=sr, mono=False)
    y2, _ = librosa.load(path2, sr=sr, mono=False)
    min_len = min(y1.shape[-1], y2.shape[-1])
    
    y1 = y1[..., :min_len]
    y2 = y2[..., :min_len]
    
    if y1.ndim == 1: y1 = np.stack([y1, y1])
    if y2.ndim == 1: y2 = np.stack([y2, y2])
    
    return (y1 * w1 + y2 * w2) / (w1 + w2), min_len

def smart_gate(input_path, output_path, sr):
    y, _ = librosa.load(input_path, sr=sr, mono=False)
    y_mono = librosa.to_mono(y) if y.ndim > 1 else y
    
    frame_len, hop_len = 2048, 512
    rms = librosa.feature.rms(y=y_mono, frame_length=frame_len, hop_length=hop_len)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Dynamic Thresholding
    valid_db = rms_db[rms_db > -80]
    noise_floor = np.percentile(valid_db, 10) if len(valid_db) > 0 else -80.0
    thresh_close = max(-70.0, min(-40.0, noise_floor + 4.0))
    thresh_open = max(-60.0, min(-30.0, thresh_close + 8.0))
    
    mask = np.zeros_like(rms_db)
    state = 0
    for i in range(len(rms_db)):
        if state == 0:
            if rms_db[i] > thresh_open: state = 1; mask[i] = 1.0
        else:
            if rms_db[i] < thresh_close: state = 0; mask[i] = 0.0
            else: mask[i] = 1.0
            
    # Smoothing
    smoothed = np.zeros_like(mask)
    curr = 0.0
    att = 1.0 / max(1, int(0.01 * sr / hop_len))
    rel = 1.0 / max(1, int(0.60 * sr / hop_len))
    
    for i in range(len(mask)):
        if mask[i] > curr: curr += (mask[i] - curr) * att
        else: curr += (mask[i] - curr) * rel
        smoothed[i] = curr
        
    t_f = librosa.frames_to_time(np.arange(len(smoothed)), sr=sr, hop_length=hop_len)
    t_s = librosa.samples_to_time(np.arange(y.shape[-1]), sr=sr)
    final_mask = np.interp(t_s, t_f, smoothed)
    
    y_clean = y * (final_mask * 0.95 + 0.05)
    sf.write(output_path, y_clean.T, sr)