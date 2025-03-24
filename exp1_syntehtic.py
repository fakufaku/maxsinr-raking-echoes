from pathlib import Path
import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
import json
import os

from tqdm import tqdm

from scipy.io import wavfile
from mir_eval.separation import bss_eval_images

from einops import rearrange, repeat

import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

noise_loc_choices = ["none", "noise_doa", "speech_loc"]
SIR_choices = [-5, 0, 5, 10, 15, 20, 25]
RT60_choices = [0.3, 0.6, 0.9]
bf_criterion_choices = [
    "ds", "mvdr", "lcmv", "rake",
    "souden", "matched",
    "max_sinr",
]


# Set folders
experiment_folder = "../datanet/projects/otohikari/robin/measurements/20171207"
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


def main(SIR, RT60, noise_loc_desc, bf_criterion, plot=False):
    
    #%% initialize results dict
    results_dict_image = {
        'SIR': SIR,
        'bf_criterion': bf_criterion,
        'RT60': RT60,
        'noise_loc': noise_loc_desc,
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
    speech_ref = pra.highpass(speech_ref.astype(float), r, fc=80)
    noise_ref = pra.highpass(noise_ref.astype(float), r, fc=80)

    speech_ref = speech_ref[:,0]
    noise_ref = noise_ref[:,0]
    
    speech_ref = speech_ref / np.std(speech_ref)
    noise_ref = noise_ref / np.std(noise_ref)
    
    # scale according to the desired SIR
    speech_ref *= 10**(SIR/20)
    
    #%% Create the room and plot it
    e_absorption, max_order = pra.inverse_sabine(RT60, room_dim)

    # max_order = 1

    print("absorption = ", e_absorption)
    print("max_order = ", max_order)

    room = pra.ShoeBox(room_dim, fs=fs_n, materials=pra.Material(e_absorption), max_order=max_order)
    room.add_source(speech_loc, signal=speech_ref) # speech
    room.sources[-1].name = 'target'
    if noise_loc_desc == 'doa_speech':
        noise_loc = speech_loc - np.array([0.8, 0, 0]) # between the ref source and the mic
    elif noise_loc_desc == 'doa_noise':
        noise_loc = noise_loc   # the original noise location
    elif noise_loc_desc == 'none':
        noise_loc = noise_loc   # place the noise it original location, but silence it
        noise_ref = np.zeros_like(speech_ref)
    else:
        raise ValueError("Unknown noise_loc, got {}".format(noise_loc))
        
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
    
    #%% Simulate the mix
    
    premix = room.simulate(return_premix=True)
    mix = room.mic_array.signals

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

    L = min(mix.shape[1], speech_ref.shape[0])
    ref = ref[:, :L]
    mix = mix[:, :L]

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
    X_mix = analysis(mix.T) # [T x F x M]
    oracle_mask_speech = analysis(np.ones_like(mix.T) * vad_guarded[:, None])
    X_speech = analysis(mix.T * vad_guarded[:, None])
    X_noise = analysis(mix.T * (1 - vad_guarded[:, None]))
    oracle_mask_speech = oracle_mask_speech
    oracle_mask_noise = 1 - oracle_mask_speech
    oracle_mask = oracle_mask_noise
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
    
    
if __name__ == "__main__":
    main(20, 0.6, "noise_doa", "ds", plot=True)