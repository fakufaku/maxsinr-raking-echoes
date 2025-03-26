from pathlib import Path
import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse

from tqdm import tqdm

from scipy.io import wavfile
from mir_eval.separation import bss_eval_images

from scipy.signal import fftconvolve
from einops import rearrange, repeat

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pb_bss.distribution import CACGMMTrainer
from pb_bss.permutation_alignment import DHTVPermutationAlignment

from geo_utils import get_wall_order_from_images, distance
from max_sinr_beamforming import compute_gain

from beamformers import (
    mvdr_weights, 
    delay_and_sum_weights, 
    souden_weights, 
    max_sinr_weights, 
    lcmv_weights,
    rake_weights,
    compute_steering_vector
)

import pandas as pd

N_IMAGES_PERFORMANCE = 30
VAD_GUARD = 1024
NFFT = 2048
SNR_AWGN = 30
MIC_TYPE = "pyramic"

noise_loc_choices = ["none", "noise_doa", "speech_doa"]
SIR_choices = [-5, 0, 5, 10, 15, 20, 25]
RT60_choices = [0.3, 0.6, 0.9]
bf_choices = [
    "ds", "mvdr", "lcmv", "rake",
    "souden", "matched", "mpdr", "mvdr_iso",
    "max_sinr",
]
mask_choice = ["oracle_vad", "oracle_ibm", "oracle_wiener", "cacgmm", "oracle_scm"]

parser = argparse.ArgumentParser()
parser.add_argument("--SIR", type=int, default=20, choices=SIR_choices)
parser.add_argument("--RT60", type=float, default=0.6, choices=RT60_choices)
parser.add_argument("--noise_loc", type=str, default="noise_doa", choices=noise_loc_choices)
parser.add_argument("--bf", type=str, default="max_sinr", choices=bf_choices)
parser.add_argument("--mask", type=str, default="oracle_ibm", choices=mask_choice)
parser.add_argument("--plot", action="store_true")
parser.add_argument("--all", action="store_true", help="Run all the experiments defined in the choices")


# Set folders
experiment_folder = "./datanet/projects/otohikari/robin/measurements/20171207"
file_pattern = os.path.join(experiment_folder, "segmented/{}_{}_SIR_{}_dB.wav")
file_speech_ref = os.path.join(experiment_folder, "segmented/{}_speech_ref.wav")
protocol_file = os.path.join(experiment_folder, "protocol.json")

# Load the protocol
with open(protocol_file, "r") as f:
    protocol = json.load(f)

# Set the microphone geometry
mic_choices = {"olympus": "camera_audio", "pyramic": "pyramic_audio"}
mics_geom = {
    "pyramic": np.array(protocol["geometry"]["microphones"]["pyramic"]["locations"]),
    "olympus": np.array(protocol["geometry"]["microphones"]["olympus"]["locations"]),
}   


def main(SIR, RT60, noise_loc_desc, bf, mask_type, plot=False):
    
    filename = f"./results/exp1_results_SIR-{SIR}_RT60-{RT60}_bf-{bf}_noise-{noise_loc_desc}_mask-{mask_type}"
    
    if os.path.exists(filename + "_per_image.csv"):
        print(f"File {filename} already exists, skipping")
        return
    
    #%% initialize results dict
    results_dict_image = {
        'SIR': SIR,
        'bf': bf,
        'RT60': RT60,
        'noise_loc': noise_loc_desc,
        'mask' : mask_type,
    }
    results_dict_doa = results_dict_image.copy()
    

    #%% Get the locations of the microphones
    mics_loc = np.array(protocol["geometry"]["microphones"][MIC_TYPE]["reference"])
    noise_loc = protocol["geometry"]["speakers"]["locations"][0]
    speech_loc = protocol["geometry"]["speakers"]["locations"][1]
    room_dim = protocol["geometry"]["room"]

    if MIC_TYPE == "pyramic":
        I = list(range(8, 16)) + list(range(24, 32)) + list(range(40, 48))  # flat part
        mics_positions = mics_geom["pyramic"][I].copy()
        # place in room 2-806
        mics_positions -= np.mean(mics_positions, axis=0)[None, :]
        mics_positions[:, 2] -= np.max(mics_positions[:, 2])
        mics_positions += mics_loc

    elif MIC_TYPE == "olympus":
        mics_positions = mics_geom["olympus"].copy() + mics_loc
        
    print("Room dimensions: ", room_dim)
    
    #%% Read the mix signals
    # read in the ref signals
    fs_n, noise_ref = wavfile.read(file_pattern.format(mic_choices[MIC_TYPE], "noise_ref", 20))
    fs_s, speech_ref = wavfile.read(file_speech_ref.format(mic_choices[MIC_TYPE]))
    assert fs_n == fs_s
    fs = fs_n
    
    # remove DC component
    speech_ref = speech_ref - np.mean(speech_ref, axis=0)
    noise_ref = noise_ref - np.mean(noise_ref, axis=0)

    # high pass filter
    speech_ref = pra.highpass(speech_ref.astype(float), fs, fc=80)
    noise_ref = pra.highpass(noise_ref.astype(float), fs, fc=80)

    speech_ref = speech_ref[:,0]
    noise_ref = noise_ref[:,0]
    
    speech_ref = speech_ref / np.std(speech_ref)
    noise_ref = noise_ref / np.std(noise_ref)
    
    # scale according to the desired SIR
    speech_ref *= 10**(SIR/20)
    
    #%% STFT and other stuff
    engine = pra.transform.stft.STFT(
        NFFT, NFFT // 2, pra.hann(NFFT), channels=mics_positions.shape[0]
    )
    freqs = np.fft.rfftfreq(NFFT, 1/fs)

    def analysis(x):
        engine.analysis(x)
        return engine.X

    def synthesis(X):
        engine.X = X
        return engine.synthesis()
    
    engine_1chan = pra.transform.stft.STFT(
        NFFT, NFFT // 2, pra.hann(NFFT), channels=1
    )

    def analysis_1chan(x):
        engine_1chan.analysis(x)
        return engine_1chan.X

    def synthesis_1chan(X):
        engine_1chan.X = X
        return engine_1chan.synthesis()

    
    #%% Create the room and plot it        
    e_absorption, max_order = pra.inverse_sabine(RT60, room_dim)

    print("absorption = ", e_absorption)
    print("max_order = ", max_order)
    
    results_dict_doa['absorption'] = e_absorption
    results_dict_doa['max_order'] = max_order
    results_dict_image['absorption'] = e_absorption
    results_dict_image['max_order'] = max_order

    room = pra.ShoeBox(room_dim, fs=fs_n, materials=pra.Material(e_absorption), max_order=max_order)
    room.add_source(speech_loc, signal=speech_ref) # speech
    room.sources[-1].name = 'target'
    if noise_loc_desc == 'speech_doa':
        noise_loc = speech_loc - np.array([0.8, 0.01, 0]) # between the ref source and the mic
    elif noise_loc_desc == 'noise_doa':
        noise_loc = noise_loc   # the original noise location
    elif noise_loc_desc in ['none']:
        noise_loc = noise_loc   # place the noise it original location, but silence it
        noise_ref = np.zeros_like(speech_ref)
        raise NotImplementedError("Not implemented yet")
    else:
        raise ValueError("Unknown noise_loc, got {}".format(noise_loc_desc))
        
    room.add_source(noise_loc, signal=noise_ref) # noise
    room.sources[-1].name = 'interf'
    room.add_microphone_array(pra.MicrophoneArray(mics_positions.T, fs=room.fs))

    if plot:
        room.plot()
        ax = plt.gca()
        ax.text(*speech_loc, "Speech", fontsize=12, color="blue")
        ax.text(*noise_loc, "Noise", fontsize=12, color="red")
        ax.text(*mics_loc, "Microphones", fontsize=12, color="green")
        ax.set_xlim([-.1, 10])
        ax.set_ylim([-.1, 8])
        ax.set_zlim([-.1, 2.5])
        # rotate the view
        ax.view_init(azim=-90-45)
        plt.savefig("fa2025_room_synth.png")
        plt.show()
        
    #%% Compute the RIR and images
    room.compute_rir()

    rirs = room.rir
    # make rir in a matrix
    L = max([max([len(rir) for rir in rir_i]) for rir_i in rirs])
    L = max(L, NFFT)
    rir_matrix = np.zeros((L, len(rirs), len(rirs[0])))
    for i, rir_i in enumerate(rirs):
        for j, rir_ij in enumerate(rir_i):
            rir_matrix[:len(rir_ij), i, j] = rir_ij
    rirs = rir_matrix # nSamples, nSrc, nChan
    print("RIR shape: ", rirs.shape)

    X_rirs = np.stack([analysis(rirs[:,:,0]), analysis(rirs[:,:,1])], axis=0)
    X_rirs.shape

    # compute sources' image information
    source_echoes = []
    n_images = 200
    for s, source in enumerate(tqdm(room.sources)):

        print(f"Source {s}")

        name = source.name
        src_images_pos =  room.sources[s].images
        src_images_order = room.sources[s].orders
        src_images_dampings = room.sources[s].damping.squeeze(0)
        src_images_dist = np.linalg.norm(src_images_pos - mics_loc[:,None], axis=0)

        # sort accoding to distance
        idx = np.argsort(src_images_dist)
        
        src_images_pos = src_images_pos[:,idx]
        src_images_dist = src_images_dist[idx]
        src_images_order = src_images_order[idx]
        src_images_dampings = src_images_dampings[idx]
        
        # keep first 100 accoding to distance    
        src_images_pos = src_images_pos[:,:n_images]
        src_images_dist = src_images_dist[:n_images]
        src_images_dampings = src_images_dampings[:n_images]
        src_images_order = src_images_order[:n_images]
        
        # prune the doas whose energy is below a threshold
        coeff = src_images_dampings / src_images_dist
        idx = coeff > 0.1 * coeff[0]
        src_images_pos = src_images_pos[:,idx]
        src_images_dist = src_images_dist[idx]
        src_images_dampings = src_images_dampings[idx]
        src_images_order = src_images_order[idx]
        coeff = src_images_dampings / src_images_dist
        
        # get the wall list
        src_images_walls = get_wall_order_from_images(src_images_pos, mics_loc, room_dim)
        
        images_names = [f"{i}_{src_images_walls[i]}" for i in range(len(src_images_walls))]

        # angle between image and reference point
        unit_vect = src_images_pos - mics_loc[:,None]
        doas_images = np.arctan2(unit_vect[1], unit_vect[0])
        doas_images = np.mod(doas_images, 2*np.pi)
        toas_images = src_images_dist / room.c
        
        # flag the early reflection
        # early reflection = less than 80 ms from the direct path (C80) and energy is above 15 dB from the direct path
        tdoa = np.abs(toas_images - toas_images[0])
        relative_amplitude_dB = 20*np.log10(coeff) - 20*np.log10(coeff[0])
        early_reflection_flag = (tdoa < 80e-3) & (relative_amplitude_dB > -10)
        
        # compute acoustic images
        atfs = []
        for d in range(min(N_IMAGES_PERFORMANCE, src_images_pos.shape[1])):
            atf = coeff[d] * compute_steering_vector(src_images_pos[:,[d]], room.mic_array.R, room.c, freqs, ref_mic_idx=None, mode="near", delay_sec=40/fs)
            atfs.append(atf)
        atfs = np.concatenate(atfs, axis=1)
        incrememntal_atf = np.cumsum(atfs, axis=1)
        
        assert np.allclose(atfs[:,0], incrememntal_atf[:,0])
        
        source_echoes.append({
        "name" : name,
        "coeffs" : coeff,
        "doas" : doas_images,
        "toas" : toas_images,
        "images" : src_images_pos,
        "walls" : src_images_walls,
        "order" : src_images_order,
        "images_names" : images_names,
        "early_reflection_flag" : early_reflection_flag,
        "atfs_fji" : atfs,
        "incremental_atf_fji" : incrememntal_atf,
    })
        
    
    results_dict_image['image_early_reflection_flag'] = source_echoes[0]['early_reflection_flag'][:N_IMAGES_PERFORMANCE]
    results_dict_image['image_walls'] = source_echoes[0]['walls'][:N_IMAGES_PERFORMANCE]
    results_dict_image['image_order'] = source_echoes[0]['order'][:N_IMAGES_PERFORMANCE]
    results_dict_image['image_toas'] = source_echoes[0]['toas'][:N_IMAGES_PERFORMANCE]
    results_dict_image['image_doas_rad'] = source_echoes[0]['doas'][:N_IMAGES_PERFORMANCE]
    results_dict_image['image_coeffs'] = source_echoes[0]['coeffs'][:N_IMAGES_PERFORMANCE]
    
    #%% Simulate the mix
    premix = room.simulate(return_premix=True)
    mix = room.mic_array.signals
    L = min(mix.shape[1], speech_ref.shape[0], premix.shape[-1])
    mix = mix[:, :L]
    premix = premix[..., :L]
    speech_ref = speech_ref[:L]
    noise_ref = noise_ref[:L]
        
    # add tiny noise to the mix
    noise = np.random.randn(*mix.shape)
    noise = (noise / np.std(noise)) * np.std(mix) / 10**(SNR_AWGN/20)
    mix = mix + noise
    
    #%% STFT of the mix
    mix_stft = analysis(mix.T)

    print("Mix shape: ", mix.shape)
    print("Mix STFT shape: ", mix_stft.shape)
    
    premix_stft = np.stack(
        [analysis(premix[0,:,:].T), 
            analysis(premix[1,:,:].T)
    ], axis=0)
    
    print("Premix shape: ", premix.shape)
    print("Premix STFT shape: ", premix_stft.shape)

    if plot:
        plt.figure(figsize=(12, 3))
        plt.plot(mix[0])
        plt.title("Mix signal")
        plt.show()
    
        fig, ax = plt.subplots(1, 2, figsize=(12, 3), sharex=True, sharey=True)
        ax[0].plot(premix[0,0,:])
        ax[0].set_title("Premix signal")
        ax[1].plot(premix[1,0,:])
        ax[1].set_title("Premix signal")
        plt.show()

    #%% # Compute oracle VAD mask
    vad_snd = np.abs(speech_ref) > 0.1*np.max(np.abs(speech_ref))

    vad_guard = VAD_GUARD
    vad_guarded = vad_snd.copy()
    if vad_guard is not None:
        for i, v in enumerate(vad_snd):
            if np.any(vad_snd[i - vad_guard : i + vad_guard]):
                vad_guarded[i] = True
    speech_mask = vad_guarded.copy()

    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(2, 1, 1)
        plt.plot(speech_ref)
        plt.plot(speech_mask * np.max(speech_ref))
        plt.subplot(2, 1, 2)
        plt.plot(speech_mask)
        plt.title("Speech mask")
        plt.show()
    
    #%% Compute reference performances
    ref_mic = 0
    ref = np.vstack([speech_ref, noise_ref])

    print("Reference shape: ", ref.shape)
    print("Mix shape: ", mix.shape)

    metric = bss_eval_images(
        ref[:, :, None], 
        np.stack([mix[ref_mic,:, None]]*2, axis=0)
    )

    SDR_in = metric[0][0]
    SIR_in = metric[2][0]
    print('SDR_in', SDR_in)
    print('SIR_in', SIR_in)

    results_dict_image['SDR_in'] = SDR_in
    results_dict_image['SIR_in'] = SIR_in
    results_dict_doa['SDR_in'] = SDR_in
    results_dict_doa['SIR_in'] = SIR_in 
    
    #%% START Beamforming
    # compute mask
    X_mix = analysis(mix.T) # [T x F x M]
    if mask_type == 'oracle_vad':
        oracle_mask_speech = analysis(np.ones_like(mix.T) * vad_guarded[:, None])
        X_speech = analysis(mix.T * vad_guarded[:, None])
        X_noise = analysis(mix.T * (1 - vad_guarded[:, None]))
        oracle_mask_speech = oracle_mask_speech
        oracle_mask_noise = 1 - oracle_mask_speech
        
    elif mask_type == 'oracle_scm':
        X_speech = premix_stft[0]
        X_noise = X_mix - X_speech
        
    elif mask_type == 'oracle_ibm':
        oracle_mask = 20 * np.log10(abs(S_ref[...,None])) < 30
        oracle_mask_noise = oracle_mask
        oracle_mask_speech = (1 - oracle_mask_noise)
        X_speech = X_mix * oracle_mask_speech
        X_noise = X_mix - X_speech
        
    elif mask == "oracle-wiener":
        oracle_mask_speech = (np.abs(S_ref)**2 / (np.abs(S_ref)**2 + np.abs(N_ref)**2))[...,None]
        oracle_mask_noise = (np.abs(N_ref)**2 / (np.abs(S_ref)**2 + np.abs(N_ref)**2))[...,None]
        oracle_mask = oracle_mask_noise
        X_speech = X_mix * oracle_mask_speech
        X_noise = X_mix - X_speech
        
    elif mask == 'cacgmm':
        X_mix_ftm = rearrange(X_mix, 't f m -> f t m')
        M = 10 # number of mics
        X_speech_ftm = rearrange(
            analysis(mix * vad_guarded[:, None]), 
            't f m -> f t m'
        )
        source_activity_mask = np.abs(X_speech_ftm)[...,:1] > 0.1 # [nFreq, nTime, nSources]
        source_activity_mask = rearrange(source_activity_mask, 'f t k -> f k t')
        source_activity_mask = np.concatenate([
            source_activity_mask, 
            np.logical_not(source_activity_mask)
        ], axis=1)
        model = CACGMMTrainer().fit(
            X_mix_ftm[...,:M],
            source_activity_mask=source_activity_mask,
            num_classes=2, # number of sources
            iterations=40, 
            covariance_norm='trace',
        )
        affiliation = model.predict(X_mix_ftm[...,:M])
        pa = DHTVPermutationAlignment.from_stft_size(X_mix_ftm.shape[0]-1)
        mapping = pa.calculate_mapping(
            rearrange(affiliation, 'f k t -> k f t')
        )
        affiliation_pa = pa.apply_mapping(
            rearrange(affiliation, 'f k t -> k f t'),
            mapping,
        )
        masks = rearrange(affiliation_pa, 's f t -> s t f')
        X_speech = masks[0,...,None] * X_mix
        X_noise = X_mix - X_speech
        
    else:
        raise NotImplementedError(f"Unknown mask type, got {mask_type}")
    print('X_speech', X_speech.shape)
    print('X_noise', X_noise.shape)
    print('X_mix', X_mix.shape)

    nTime, nFreq, nChan = X_mix.shape
    
    #%% Compute spatial covariance functions
    
    # covariance matrices from noisy signal
    Rx = np.einsum("i...j,i...k->...jk", X_mix, np.conj(X_mix)) / X_mix.shape[0]
    Rs = np.einsum("i...j,i...k->...jk", X_speech, np.conj(X_speech)) / X_speech.shape[0]
    Rsh = np.einsum("i...j,i...k->...jk", X_rirs[0], np.conj(X_rirs[0])) / X_rirs[0].shape[0]
    Rn = np.einsum("i...j,i...k->...jk", X_noise, np.conj(X_noise)) / X_noise.shape[0]
    Rnh = np.einsum("i...j,i...k->...jk", X_rirs[1], np.conj(X_rirs[1])) / X_rirs[1].shape[0]
    Rn += 1e-5 * np.eye(Rn.shape[1]) * np.trace(Rn) / Rn.shape[-1]
    eI = 1e-5 * np.trace(Rx) *  np.stack([np.eye(nChan) for _ in range(nFreq)], axis=0) / nChan

    assert Rn.shape == (nFreq, nChan, nChan)
    assert Rx.shape == Rs.shape == Rn.shape, f'Rx : {Rx.shape}, Rs : {Rs.shape}, Rn : {Rn.shape}'
    print('Rx', Rx.shape)
    print('Rs', Rs.shape)
    print('Rsh', Rsh.shape)
    print('Rn', Rn.shape)
    print('Rnh', Rnh.shape)
    print('eI', eI.shape)
    
    #%% Compute the beamformer weights
    if bf == 'max_sinr':
        Rnum = Rx
        Rden = Rn
        w = max_sinr_weights(Rnum[1:], Rden[1:])
        # Post processing of the weights
        nw = np.linalg.norm(w, axis=1)
        w[nw > 1e-10, :] /= nw[nw > 1e-10, None]
        w = np.concatenate([np.ones((1, nChan)), w], axis=0)
        # normalize with respect to input signal (see Warsitz and Haeb-Umbach, 2007)
        z = []
        for i in range(nChan):
            z.append(compute_gain(w, X_speech, X_speech[...,i]))
        normalization_factor = np.stack(z, -1)
        w_r = np.einsum('fi,fr->fir', w, normalization_factor)
        
    elif bf == 'ds':
        w_r = []
        for i in range(nChan):
            w = delay_and_sum_weights(room.sources[0].position[:,None], room.mic_array.R, room.c, freqs[1:], None)  
            w = np.concatenate([np.zeros((1, w.shape[1])), w], axis=0)
            w_r.append(w)
        w_r = np.stack(w_r, axis=-1)
        
    elif bf == 'mvdr':
        w_r = []
        for i in range(nChan):
            w = mvdr_weights(room.sources[0].position[:,None], room.mic_array.R, room.c, freqs[1:], Rn[1:], i, True)
            w = np.concatenate([np.zeros((1, w.shape[1])), w], axis=0)
            w_r.append(w)
        w_r = np.stack(w_r, axis=-1)
        
    elif bf == 'mpdr':
        w_r = []
        for i in range(nChan):
            w = mvdr_weights(room.sources[0].position[:,None], room.mic_array.R, room.c, freqs[1:], Rx[1:], i, True)
            w = np.concatenate([np.zeros((1, w.shape[1])), w], axis=0)
            w_r.append(w)
        w_r = np.stack(w_r, axis=-1)
        
    elif bf == 'mvdr_iso':
        mic_pos = room.mic_array.R # [3 x I]
        ell_ii = np.linalg.norm(mic_pos[:,None,:] - mic_pos[:,:,None], axis=0) # [I x I]
        Gamma = np.sinc(2 * np.pi * freqs[:,None,None] * ell_ii[None] / room.c) # [F x I x I]
        w_r = []
        for i in range(nChan):
            w = mvdr_weights(room.sources[0].position[:,None], room.mic_array.R, room.c, freqs[1:], Gamma[1:], i, True)
            w = np.concatenate([np.zeros((1, w.shape[1])), w], axis=0)
            w_r.append(w)
        w_r = np.stack(w_r, axis=-1)
    
    elif bf == 'lcmv':
        w_r = []
        for i in range(nChan):
            w = lcmv_weights(
                room.sources[0].position[:,None], # target
                room.sources[1].position[:,None], # interf
                room.mic_array.R, room.c, freqs[1:], Rn[1:], i, True)
            w = np.concatenate([np.zeros((1, w.shape[1])), w], axis=0)
            w_r.append(w)
        w_r = np.stack(w_r, axis=-1)
    
    elif bf == 'rake':
        w_r = []
        for i in range(nChan):
            w = rake_weights(
                source_echoes[0]["images"][:,:5], 
                room.mic_array.R, room.c, freqs[1:], Rn[1:], i, 0)
            w = np.concatenate([np.zeros((1, w.shape[1])), w], axis=0)
            w_r.append(w)
        w_r = np.stack(w_r, axis=-1)
        
    elif bf == 'matched':
        w_r = []
        for i in range(nChan):
            w = souden_weights(Rn=Rn[1:], Rs=Rsh[1:], ref_mic_idx=i)
            w = np.concatenate([np.zeros((1, w.shape[1])), w], axis=0)
            w_r.append(w)
        w_r = np.stack(w_r, axis=-1)
        
    elif bf == 'souden':
        w_r = []
        for i in range(nChan):
            w = souden_weights(Rn[1:], Rs[1:], 0, clip_gain=args.clip_gain)
            w = np.concatenate([np.zeros((1, w.shape[1])), w], axis=0)
            w_r.append(w)
        w_r = np.stack(w_r, axis=-1)
    
    else:
        raise ValueError(f"Unknown beamforming criterion {bf}")

    # find the best reference microphone
    snr_per_chan = np.einsum('fir,fij,fjr->r', w_r.conj(), Rs, w_r).real / np.einsum('fir,fij,fjr->r', w_r.conj(), Rn, w_r).real
    snr_per_chan = snr_per_chan / nFreq

    if plot:
        plt.figure(figsize=(6, 2))
        plt.plot(snr_per_chan)
        plt.title("SNR per channel")
        plt.show()

    ref_mic_idx = np.argmax(snr_per_chan)
    print(f"Best reference microphone is {ref_mic_idx}")
    bf_weights = w_r[:,:,ref_mic_idx]

    results_dict_doa['ref_mic_idx'] = ref_mic_idx
    results_dict_image['ref_mic_idx'] = ref_mic_idx
    
    print("Apply beamformer")

    # 2D beamformer
    mic_array = pra.Beamformer(
        mics_positions.T[:2], fs=fs, N=NFFT, hop=NFFT, zpb=NFFT # it s in 2D so we can plot the response
    )
    mic_array.signals = mix
    mic_array.weights = bf_weights.T

    out = mic_array.process()

    if plot:
        plt.figure(figsize=(12,3))
        plt.plot(mix[ref_mic], label="Mix")
        plt.plot(out, label="Beamformer")
        plt.legend()
        plt.title("Mix and beamformer output")
        plt.show()
        
    #%% Signal alignment step
    ref = np.vstack([speech_ref, noise_ref])
    # Not sure why the delay is sometimes negative here... Need to check more
    delay = np.abs(int(pra.tdoa(out, speech_ref.astype(float), phat=True)))
    if delay > 0:
        out_trunc = out[delay : delay + ref.shape[1]]
        noise_eval = mix.T[: ref.shape[1], ref_mic_idx] - out_trunc
    else:
        out_trunc = np.concatenate((np.zeros(-delay), out[: ref.shape[1] + delay]))
        noise_eval = mix.T[: ref.shape[1], ref_mic_idx] - out_trunc
    sig_eval = np.vstack([out_trunc, noise_eval])

    # We use the BSS eval toolbox
    metric = bss_eval_images(ref[:, :, None], sig_eval[:, :, None])

    # we are only interested in SDR and SIR for the speech source
    SDR_out = metric[0][0]
    SIR_out = metric[2][0]

    print(f'Metric: SDR_out {SDR_out:.2f}  SIR_out {SIR_out:.2f}')
    iSDR = SDR_out - SDR_in
    iSIR = SIR_out - SIR_in
    print(f'Improvement: iSDR {iSDR:.2f} iSIR {iSIR:.2f}')

    results_dict_image['SDR_out'] = SDR_out
    results_dict_image['SIR_out'] = SIR_out
    results_dict_image['iSDR'] = iSDR
    results_dict_image['iSIR'] = iSIR
    results_dict_doa['SDR_out'] = SDR_out
    results_dict_doa['SIR_out'] = SIR_out
    results_dict_doa['iSDR'] = iSDR
    results_dict_doa['iSIR'] = iSIR
    
    #%% Apply the beamformer to the each image
    mic_images = []
    inc_mic_images = []
    bf_mic_images = []
    bf_inc_mic_images = []

    S_ref = analysis_1chan(speech_ref.T)
    N_ref = analysis_1chan(noise_ref.T)

    sources = [speech_ref, noise_ref]

    nFreq, nImg, nChan = source_echoes[0]['atfs_fji'].shape

    for j in tqdm(range(len(room.sources)), desc='Partial AFT for source'):
        
        name = source_echoes[j]['name']
        atfs_fki = source_echoes[j]['atfs_fji']
        inc_atf_fki = source_echoes[j]['incremental_atf_fji']
        
        airs = np.fft.irfft(atfs_fki, NFFT, axis=0)
        inc_airs = np.fft.irfft(inc_atf_fki, NFFT, axis=0)
        
        # go in the time domain
        sig = sources[j]
        mic_image = []
        inc_mic_image = []

        L = airs.shape[0] + sig.shape[0] - 1
        
        mic_image = np.zeros((L, nImg, nChan))
        inc_mic_image = np.zeros((L, nImg, nChan))

        for k in range(nImg):
            for i in np.arange(nChan):
                conv = fftconvolve(airs[:,k,i], sig)
                mic_image[:len(conv),k,i] = conv
                conv = fftconvolve(inc_airs[:,k,i], sig)
                inc_mic_image[:len(conv),k,i] = conv
        
        
        mic_images.append(mic_image)
        inc_mic_images.append(inc_mic_image)
        
        bf_images = []
        bf_inc_images = []
        for k in range(nImg):
            mic_array.signals = mic_images[j][:,k,:].T
            mic_array.weights = bf_weights.T
            bf_images.append(mic_array.process())
            
            mic_array.signals = inc_mic_images[j][:,k,:].T
            mic_array.weights = bf_weights.T
            bf_inc_images.append(mic_array.process())
            
        bf_mic_images.append(np.stack(bf_images, axis=1))
        bf_inc_mic_images.append(np.stack(bf_inc_images, axis=1))


    mic_images = np.stack(mic_images, axis=0)                 # [nSrc x T x nDoas x nChan]
    inc_mic_images = np.stack(inc_mic_images, axis=0)         # [nSrc x T x nDoas x nChan]
    bf_mic_images = np.stack(bf_mic_images, axis=0)           # [nSrc x T x nDoas]
    bf_inc_mic_images = np.stack(bf_inc_mic_images, axis=0)   # [nSrc x T x nDoas]

    print('mic_images', mic_images.shape)
    print('inc_mic_images', inc_mic_images.shape)
    print('bf_mic_images', bf_mic_images.shape)
    print('bf_inc_mic_images', bf_inc_mic_images.shape)
    
    #%% # Get the full image with RIR and apply beamforming
    L = rirs.shape[0] + sig.shape[0] - 1
    full_images = np.zeros((2, L, nChan))
    bf_full_images = np.zeros((2, L + NFFT - 1))


    for j in range(len(room.sources)):
        for i in range(nChan):
            conv = fftconvolve(rirs[:,i,j].T, sources[j])
            full_images[j, : len(conv), i] = conv

        mic_array.signals = full_images[0].T
        mic_array.weights = bf_weights.T
        out = mic_array.process()
        bf_full_images[j, :len(out)] = out

    print('full_image', full_images.shape)
    print('bf_full_image', bf_full_images.shape)
    
    # Apply beamforming to the noise
    mic_array.signals = noise
    mic_array.weights = bf_weights.T
    bf_noise = mic_array.process()
    
    L = min(mix.shape[1], full_images.shape[1], noise.shape[1])
    # first we need to check that the images correctly provide the mix signal
    mix_from_images = full_images[:, : L, ref_mic].sum(0) + noise[ref_mic,: L]
    mix_minus_contrib = mix[ref_mic,: L] - mix_from_images
    assert np.allclose(mix_minus_contrib, 0)
    
    #%% Compute all the metrics per image
    OIR_k_out = np.zeros((nImg))
    IP_k_out = np.zeros((nImg))
    SP_k_out = np.zeros((nImg))
    SNR_k_out = np.zeros((nImg))
    Rakeness_k_out = np.zeros((nImg))

    for k in range(nImg):
        
        # computation of the OIR
        other_bf_images = bf_mic_images[j].sum(-1) - bf_mic_images[j,:,k]
        bf_interf = bf_mic_images[1].sum(-1)
        L = min(other_bf_images.shape[0], bf_interf.shape[0], bf_noise.shape[0])
        all_rest = other_bf_images[: L] + bf_noise[: L] + bf_interf[: L]
        noise_power_out = np.std(all_rest)**2
        signal_power_out = np.std(bf_mic_images[0,:L,k])**2
        OIR_k_out[k] = 10 * (np.log10(signal_power_out) - np.log10(noise_power_out))
        
        # computation of the IP        
        bf_out_power = np.std(bf_mic_images[0,:,k])**2
        ref_sig_power = np.std(mic_images[0,:, k])**2
        IP_k_out[k] = 10 * (np.log10(bf_out_power) - np.log10(ref_sig_power))
        
        # computation of the SP
        bf_out_power = np.std(bf_mic_images[0,:,k])**2
        ref_sig_power = np.std(mic_images[0,:, 0])**2
        SP_k_out[k] = 10 * (np.log10(bf_out_power) - np.log10(ref_sig_power))
        
        # computation of the SNR
        L = min(bf_noise.shape[0], bf_mic_images[0,:,k].shape[0], bf_interf.shape[0])
        bf_distortion = bf_noise[: L] + bf_interf[: L]
        bf_distortion_power = np.std(bf_distortion)**2
        bf_out_power = np.std(bf_mic_images[0,:L,k])**2
        SNR_k_out[k] = 10 * (np.log10(bf_out_power) - np.log10(bf_distortion_power))
        
        # computation of the Rakeness
        L = min(bf_mic_images[0,:,k].shape[0], bf_full_images.shape[1])
        image_power = np.std(bf_mic_images[0,:L,k])**2
        all_images_power = np.sum(np.std(bf_mic_images[0,:L,:], axis=0)**2)
        Rakeness_k_out[k] = 10 * (np.log10(image_power) - np.log10(all_images_power))

    results_dict_image['OIR_k_out'] = OIR_k_out.tolist()
    results_dict_image['IP_k_out'] = IP_k_out.tolist()
    results_dict_image['SP_k_out'] = SP_k_out.tolist()
    results_dict_image['SNR_k_out'] = SNR_k_out.tolist()
    results_dict_image['Rakeness_k_out'] = Rakeness_k_out.tolist()
    
    # Correlation per image
    images_atfs = source_echoes[0]["atfs_fji"]
    nFreq, nImg, nChan = images_atfs.shape
    bf_gain = np.abs(np.einsum('fi,fji->fj', bf_weights.conj(), images_atfs))**2
    bf_gain = np.mean(bf_gain, axis=0)
    
    results_dict_image['bf_gain'] = bf_gain.tolist()
    
    # Correlation per image
    images_atfs = source_echoes[0]["atfs_fji"]
    nFreq, nImg, nChan = images_atfs.shape

    corr_with_dp = np.abs(np.einsum('fi,fji->fj', images_atfs[1:,0,:].conj(), images_atfs[1:]))
    corr_with_dp /= np.linalg.norm(images_atfs[1:,0,:], axis=-1)[:,None] * np.linalg.norm(images_atfs[1:], axis=-1)
    corr_pooled_wtr_dp = np.mean(corr_with_dp, axis=0)

    corr_with_bf = np.abs(np.einsum('fi,fji->fj', bf_weights[1:].conj(), images_atfs[1:]))
    corr_with_bf /= np.linalg.norm(bf_weights, axis=-1)[1:,None] * np.linalg.norm(images_atfs[1:], axis=-1)
    corr_pooled_wrt_bf = np.mean(corr_with_bf, axis=0)
    
    results_dict_image['corr_pooled_wtr_dp'] = corr_pooled_wtr_dp.tolist()
    results_dict_image['corr_pooled_wrt_bf'] = corr_pooled_wrt_bf.tolist()
    
    # Compute directivity index
    src_pos = room.sources[0].position
    mic_pos = room.mic_array.R
    arr_pos = room.mic_array.center
    src_arr_dist = np.linalg.norm(src_pos - arr_pos)
    print("Source to array distance: ", src_arr_dist)
    
    # create a grid of DOAs
    nDoas = 360
    grid_circ = pra.doa.grid.GridCircle(n_points=360)
    cart_pts_circ = grid_circ.cartesian
    cart_pts_circ *= src_arr_dist
    cart_pts_circ += arr_pos
    azimuths = grid_circ.azimuth
    print(cart_pts_circ.shape)

    beampatter_circ = np.zeros((nDoas, nFreq))    
    for i in range(nDoas):
        steering_vector = compute_steering_vector(cart_pts_circ[:,[i]], mic_pos, room.c, freqs, ref_mic_idx=ref_mic_idx, mode="near")
        beampatter_circ[i,:] = np.abs(np.einsum('fi,fji->fj', bf_weights.conj(), steering_vector)[:,0])**2

    # compute directivity index
    DI = 1 / beampatter_circ.sum(-1).mean()
    results_dict_image['DI_circ'] = DI
    
    results_dict_doa['azimuths'] = azimuths.tolist()
    results_dict_doa['beampatter_circ'] = beampatter_circ.tolist()

    # Directivity on the sphere    
    nDoas_sph = 1000
    grid_sph = pra.doa.grid.GridSphere(n_points=nDoas_sph)
    cart_pts_sph = grid_sph.cartesian
    cart_pts_sph *= src_arr_dist
    cart_pts_sph += arr_pos
    colatitude = grid_sph.colatitude

    beampatter_sph = np.zeros((nDoas_sph, nFreq))
    for i in range(nDoas_sph):
        steering_vector = compute_steering_vector(cart_pts_sph[:,[i]], mic_pos, room.c, freqs, ref_mic_idx=ref_mic_idx, mode="near")
        beampatter_sph[i,:] = np.abs(np.einsum('fi,fji->fj', bf_weights.conj(), steering_vector)[:,0])**2

    # compute directivity index
    P = beampatter_sph.sum(-1)
    Power_full_sph = np.sum(P * np.sin(colatitude), axis=0) / (4 * np.pi)
    DI_sph = np.max(P) / Power_full_sph
    results_dict_image['DI_sph'] = DI_sph
    
    # compute all pair distances
    distances = np.linalg.norm(mic_pos[:,:,None] - mic_pos[:,None,:], axis=0)
    # compute the isotropic coherence matrix
    noise_coherence = np.sinc(2 * freqs[:,None,None] * distances[None] / room.c)
    # compute the directivity index
    DI = 1 / np.einsum('fi,fij,fj->f', bf_weights.conj(), noise_coherence, bf_weights).real[1:]
    print("Directivity index with diffuse noise: ", DI.sum())

    results_dict_image["DI_diff"] = DI.sum()
    results_dict_doa["DI_diff"] = DI.sum()
    
    # Directivity per image
    DI_per_k = np.zeros((nImg))
    for k in range(mic_images.shape[2]):
        atf = source_echoes[0]["atfs_fji"][:,k,:]
        svect = compute_steering_vector(source_echoes[0]['images'][:,[k]], room.mic_array.R, room.c, freqs, ref_mic_idx=None, mode="near", delay_sec=40/fs)
        power_k = np.sum(np.abs(np.einsum('fi,fji->fj', bf_weights.conj(), svect))**2)
        DI_per_k[k] = power_k / Power_full_sph
        
    DI_per_k = np.array(DI_per_k)

    results_dict_image['DI_k'] = DI_per_k.tolist()
    
    #%% Save the results
    df_results_image = pd.DataFrame(results_dict_image)
    df_results_image.to_csv(filename + "_per_image.csv", index=False)
    df_results_azimuth = pd.DataFrame(results_dict_doa)
    df_results_azimuth.to_csv(filename + "_per_azimuth.csv", index=False)
    print("Results saved in ", filename + "_per_image.csv")
    print("Results saved in ", filename + "_per_azimuth.csv")

if __name__ == "__main__":
    args = parser.parse_args()
    
    SIR = args.SIR
    RT60 = args.RT60
    noise_loc = args.noise_loc
    bf = args.bf
    mask = args.mask
    plot = args.plot
    main(SIR, RT60, noise_loc, bf, mask, plot)
    
    # main(20, 0.6, "noise_doa", "ds")
    print("Done. :*")