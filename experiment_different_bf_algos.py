"""
This file applies a max SINR approach using the VAD information
from the LED and the two channels from the camera

Author: Robin Scheibler
Created: 2017/12/01
Modified: 2022/07/29 by Diego Di Carlo
"""

import argparse, os, json, sys
import numpy as np
import scipy.linalg as la
from scipy.io import wavfile
import pyroomacoustics as pra
import librosa as lr
from mir_eval.separation import bss_eval_images
from pprint import pprint
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from max_sinr_beamforming import compute_gain
from geo_utils import get_wall_order_from_images

experiment_folder = "datanet/projects/otohikari/robin/measurements/20171207"
file_pattern = os.path.join(experiment_folder, "segmented/{}_{}_SIR_{}_dB.wav")
file_speech_ref = os.path.join(experiment_folder, "segmented/{}_speech_ref.wav")
protocol_file = os.path.join(experiment_folder, "protocol.json")

with open(protocol_file, "r") as f:
    protocol = json.load(f)

thresh_opt = {5: 650, 10: 600, 15: 540, 20: 425, 25: 340}  # 2017/12/07
# thresh_opt = {"5": 660, "10": 600, "15": 540, "20": 450, "25": 340}

sir_choices = [5, 10, 15, 20, 25]
mic_choices = {"olympus": "camera_audio", "pyramic": "pyramic_audio"}
mics_geom = {
    "pyramic": np.array(protocol["geometry"]["microphones"]["pyramic"]["locations"]),
    "olympus": np.array(protocol["geometry"]["microphones"]["olympus"]["locations"]),
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "SIR",
    nargs="?",
    type=int,
    choices=sir_choices,
    help="The SIR between speech and noise",
)
parser.add_argument(
    "mic",
    nargs="?",
    type=str,
    choices=mic_choices.keys(),
    help="Which input device to use",
)
parser.add_argument("--thresh", "-t", type=float, help="The threshold for VAD")
parser.add_argument(
    "--nfft", type=int, default=1024, help="The FFT size to use for STFT"
)
parser.add_argument(
    "--no_norm", action="store_true", help="Disable matching of output to channel 1"
)
parser.add_argument("--clip_gain", type=float, help="Clip the maximum gain")
parser.add_argument(
    "--vad_guard", type=int, help="Value by which to extend VAD boundaries"
)
parser.add_argument(
    "--synth_mix",
    action="store_true",
    help="Works on artifical mix of signals and compute objective evaluation.",
)
parser.add_argument(
    "--save_sample",
    type=Path,
    help="Save samples of input and output signals, argument is the directory",
)
parser.add_argument("--plot", action="store_true", help="Display all the figures")
parser.add_argument("--all", action="store_true", help="Process all the samples")


def process_experiment_max_sinr(SIR, mic, args):

    nfft = args.nfft
    vad_guard = args.vad_guard
    if args.thresh is None:
        vad_thresh = thresh_opt[SIR]
    else:
        vad_thresh = args.thresh

    # read_in the mix signals
    fs_led, leds = wavfile.read(
        file_pattern.format("camera_leds_zero_hold", "mix", SIR)
    )
    fs_snd, mix = wavfile.read(file_pattern.format(mic_choices[mic], "mix", SIR))
    assert fs_led == fs_snd

    # read in the ref signals
    r, noise_ref = wavfile.read(file_pattern.format(mic_choices[mic], "noise_ref", SIR))
    assert r == fs_snd
    r, speech_ref = wavfile.read(file_speech_ref.format(mic_choices[mic]))
    assert r == fs_snd
    r, leds_ref = wavfile.read(file_speech_ref.format("camera_leds_zero_hold"))
    assert r == fs_snd
    fs = fs_snd # shorthand
    
    # In case of objective evaluation, we do an artificial mix
    if args.synth_mix:
        mix = noise_ref + speech_ref

    # get the geometry information to get nice plots.
    mics_loc = np.array(protocol["geometry"]["microphones"][mic]["reference"])
    noise_loc = protocol["geometry"]["speakers"]["locations"][0]
    speech_loc = protocol["geometry"]["speakers"]["locations"][1]
    room_dim = protocol["geometry"]["room"]
    
    # the directions of arrival
    theta_speech = 0
    p0 = speech_loc - mics_loc
    p1 = noise_loc - mics_loc
    theta_noise = np.arccos(np.inner(p0, p1) / la.norm(p0) / la.norm(p1))
    print("Source separation", theta_noise / np.pi * 180)

    if mic == "pyramic":
        I = list(range(8, 16)) + list(range(24, 32)) + list(range(40, 48))  # flat part
        mix = mix[:, I]
        noise_ref = noise_ref[:, I].copy()
        speech_ref = speech_ref[:, I].copy()
        mics_positions = mics_geom["pyramic"][I].copy()
        # place in room 2-806
        mics_positions -= np.mean(mics_positions, axis=0)[None, :]
        mics_positions[:, 2] -= np.max(mics_positions[:, 2])
        mics_positions += mics_loc

    elif mic == "olympus":
        mics_positions = mics_geom["olympus"].copy() + mics_loc

    n_samples = mix.shape[0]  # shorthand
    n_channels = mix.shape[1]

    # perform VAD
    vad_snd = leds > vad_thresh

    # Now we want to make sure no speech speech goes in estimation of the noise covariance matrix.
    # For that we will remove frames neighbouring the detected speech
    vad_guarded = vad_snd.copy()
    if vad_guard is not None:
        for i, v in enumerate(vad_snd):
            if np.any(vad_snd[i - vad_guard : i + vad_guard]):
                vad_guarded[i] = True
                
    
    #######################
    ## REFERENCE METRICS ##
    #######################
    
    # Signal alignment step
    ref = np.vstack([speech_ref[:, 0], noise_ref[:, 0]])

    print(ref.shape)
    print(mix.shape)
    ref_mic = 0

    metric = bss_eval_images(
        ref[:, :, None], 
        np.stack([mix.T[ref_mic,:, None]]*2, axis=0))
    SDR_in = metric[0][0]
    SIR_in = metric[2][0]
    print('SDR_in', SDR_in)
    print('SIR_in', SIR_in)

    ##############################
    ## STFT and frame-level VAD ##
    ##############################

    print("STFT and stuff")
    sys.stdout.flush()

    engine = pra.transform.stft.STFT(
        nfft, nfft // 2, pra.hann(nfft), channels=mix.shape[1]
    )
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    

    def analysis(x):
        engine.analysis(x)
        return engine.X

    # Now compute the STFT of the microphone input
    print('mix', mix.shape)
    X = analysis(mix)
    X_time = np.arange(1, X.shape[0] + 1) * (nfft / 2) / fs_snd
    print(X.shape)

    X_mix = analysis(mix)
    X_speech = analysis(mix * vad_guarded[:, None])
    X_noise = analysis(mix * (1 - vad_guarded[:, None]))
    
    # Signal alignment step
    delay = np.abs(int(pra.tdoa(speech_ref[:, 0].astype(float), mix[:,0], phat=True)))
    speech_ref_ = np.concatenate((np.zeros(delay), speech_ref[:mix.shape[0] - delay,0]))
    noise_ref_ = np.concatenate((np.zeros(delay), noise_ref[:mix.shape[0] - delay,0]))
    
    # plot all the signals
    if args.plot:
        fig, axarr = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
        axarr[0].plot(speech_ref_)
        axarr[0].set_title(f"(Delayed) Speech reference -- delay : {delay} samples")
        axarr[1].plot(noise_ref_)
        axarr[1].set_title(f"(Delayed) Noise reference -- delay : {delay} samples")
        axarr[2].plot(mix[:, 0])
        axarr[2].set_title("Mix")
        plt.tight_layout()
        
    S_ref = pra.transform.stft.STFT(nfft, nfft // 2, pra.hann(nfft)).analysis(speech_ref_)
    N_ref = pra.transform.stft.STFT(nfft, nfft // 2, pra.hann(nfft)).analysis(noise_ref_)
    oracle_mask = np.abs(S_ref) > 10 * np.mean(np.abs(S_ref)[:30,:], axis=0)
    oracle_mask = np.abs(S_ref) / (np.abs(S_ref) + np.abs(N_ref))
    X_speech_oracle = X_mix * oracle_mask[:,:,None]

    # X_speech = X_speech_oracle
    # X_noise = X_mix - X_speech_oracle
    
    if args.plot:
        fig, ax = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
        img = lr.display.specshow(lr.amplitude_to_db(np.abs(S_ref.T), ref=np.max), y_axis='log', x_axis='time', ax=ax[0], sr=fs)
        ax[0].set_title("Speech reference")
        img = lr.display.specshow(lr.amplitude_to_db(np.abs(N_ref.T), ref=np.max), y_axis='log', x_axis='time', ax=ax[1], sr=fs)
        ax[1].set_title('Noise reference')
        img = lr.display.specshow(np.abs(oracle_mask.T), y_axis='log', x_axis='time', ax=ax[2], sr=fs)
        ax[2].set_title('Speech mask reference')
        img = lr.display.specshow(lr.amplitude_to_db(np.abs(X_speech_oracle[:,:,0].T), ref=np.max), y_axis='log', x_axis='time', ax=ax[3], sr=fs)
        ax[3].set_title('Masked mix')
        plt.tight_layout()

    ###############
    ## MAKE ROOM ##
    ###############
    
    rt60 = 0.300  # seconds. A dummy value for now
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    print("absorption = ", e_absorption)
    print("max_order = ", max_order)

    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)
    room.add_source(speech_loc) # speech
    room.add_source(noise_loc) # noise
    room.add_microphone_array(pra.MicrophoneArray(mics_positions.T, fs=room.fs))
    
    room.compute_rir()
    
    # compute sources' image information
    source_echoes = []
    n_images = 2000
    for s, source in enumerate(room.sources):
    
        src_images_pos =  room.sources[s].images
        src_images_order = room.sources[s].orders
        src_images_dampings = room.sources[s].damping.squeeze()
        src_images_dist = np.linalg.norm(src_images_pos - mics_loc[:,None], axis=0)

        # sort accoding to distance
        idx = np.argsort(src_images_dist)
            
        src_images_pos = src_images_pos[:,idx]
        src_images_dist = src_images_dist[idx]
        src_images_order = src_images_order[idx]
        src_images_dampings = src_images_dampings[idx]
        
        # # remove the images that are not on the x-y plane
        # src_images_z = src_images_pos[2,:]
        # idx = np.where(np.abs(src_images_z - src_images_z[0]) < 1e-6 )[0]
        
        # src_images_pos = src_images_pos[:,idx]
        # src_images_dist = src_images_dist[idx]
        # src_images_dampings = src_images_dampings[idx]
        # src_images_order = src_images_order[idx]
        
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

        # angle between image and reference point
        unit_vect = src_images_pos - mics_loc[:,None]
        doas_images = np.arctan2(unit_vect[1], unit_vect[0])
        doas_images = np.mod(doas_images, 2*np.pi)
        toas_images = src_images_dist / room.c

        source_echoes.append({
            "coeffs" : coeff,
            "doas" : doas_images,
            "toas" : toas_images,
            "images" : src_images_pos,
            "walls" : src_images_walls,
            "order" : src_images_order,
        })
        print(f"Source {s}")
        

    ##########################
    ## MAX SINR BEAMFORMING ##
    ##########################

    print("Max SINR beamformer computation")
    sys.stdout.flush()

    # covariance matrices from noisy signal
    Rs = np.einsum("i...j,i...k->...jk", X_speech, np.conj(X_speech)) / X_speech.shape[-1]
    Rn = np.einsum("i...j,i...k->...jk", X_noise, np.conj(X_noise)) / X_noise.shape[-1]
    # Rs = Rs - Rn

    # compute the MaxSINR beamformer
    w = [
        la.eigh(
            rs,
            b=rn,
            eigvals=(n_channels - 1, n_channels - 1),
        )[1]
        for rs, rn in zip(Rs[1:], Rn[1:])
    ]
    w = np.squeeze(np.array(w))
    print(w.shape)
    nw = la.norm(w, axis=1)
    w[nw > 1e-10, :] /= nw[nw > 1e-10, None]
    w = np.concatenate([np.ones((1, n_channels)), w], axis=0)

    if not args.no_norm:
        # normalize with respect to input signal
        z = compute_gain(w, X_speech, X_speech[:, :, 0], clip_up=args.clip_gain)
        w *= z[:, None]
    bf_weights = w

    ###########
    ## APPLY ##
    ###########

    print("Apply beamformer")
    sys.stdout.flush()

    # 2D beamformer
    mic_array = pra.Beamformer(
        mics_positions[:, :2].T, fs=fs_snd, N=nfft, hop=nfft, zpb=nfft
    )
    mic_array.signals = mix.T
    mic_array.weights = w.T

    out = mic_array.process()

    # Signal alignment step
    ref = np.vstack([speech_ref[:, 0], noise_ref[:, 0]])
    # Not sure why the delay is sometimes negative here... Need to check more
    delay = np.abs(int(pra.tdoa(out, speech_ref[:, 0].astype(float), phat=True)))
    if delay > 0:
        out_trunc = out[delay : delay + ref.shape[1]]
        noise_eval = mix[: ref.shape[1], 0] - out_trunc
    else:
        out_trunc = np.concatenate((np.zeros(-delay), out[: ref.shape[1] + delay]))
        noise_eval = mix[: ref.shape[1], 0] - out_trunc
    sig_eval = np.vstack([out_trunc, noise_eval])

    # We use the BSS eval toolbox
    metric = bss_eval_images(ref[:, :, None], sig_eval[:, :, None])

    # we are only interested in SDR and SIR for the speech source
    SDR_out = metric[0][0]
    SIR_out = metric[2][0]
    
    iSDR = SDR_out - SDR_in
    iSIR = SIR_out - SIR_in

    print(f'SDR {SDR_in:.2f} --> {SDR_out:.2f} ==> improv: {iSDR:.2f}')
    print(f'SIR {SIR_in:.2f} --> {SIR_out:.2f} ==> improv: {iSIR:.2f}')

    ##################
    ## SAVE SAMPLES ##
    ##################

    if args.save_sample is not None:
        
        if not args.save_sample.exists():
            args.save_sample.mkdir()

        # for informal listening tests, we need to high pass and normalize the
        # amplitude.
        upper = np.maximum(mix[:, 0].max(), out.max())
        sig_in = pra.highpass(mix[:, 0].astype(float) / upper, fs_snd, fc=150)
        sig_out = pra.highpass(out / upper, fs_snd, fc=150)

        f1 = args.save_sample / "{}_ch0_SIR_{}_dB.wav".format(mic, SIR)
        wavfile.write(f1, fs_snd, sig_in)
        f2 = args.save_sample / "{}_out_SIR_{}_dB.wav".format(mic, SIR)
        wavfile.write(f2, fs_snd, sig_out)

    ##########
    ## PLOT ##
    ##########

    if args.plot:
        
        if True: # added only to filter some plots
       
            # MICROPHONE
            plt.figure(figsize=(12,4))
            plt.suptitle('Microphone positions')
            plt.subplot(1,3,1)
            plt.scatter(mics_positions[:,0], mics_positions[:,1], c='r')
            plt.subplot(1,3,2)
            plt.scatter(mics_positions[:,0], mics_positions[:,2], c='r')
            plt.subplot(1,3,3)
            plt.scatter(mics_positions[:,1], mics_positions[:,2], c='r')
            plt.tight_layout()

            # SIGNAL
            plt.figure()
            plt.plot(out_trunc)
            plt.plot(speech_ref[:, 0])
            plt.legend(["output", "reference"])

            # time axis for plotting
            led_time = np.arange(leds.shape[0]) / fs_led + 1 / (2 * fs_led)
            mix_time = np.arange(n_samples) / fs_snd

            plt.figure()
            plt.plot(led_time, leds, "r")
            plt.title("LED signal")

            # match the scales of VAD and light to sound before plotting
            q_vad = np.max(mix)
            q_led = np.max(mix) / np.max(leds)

            plt.figure()
            plt.plot(mix_time, mix[:, 0], "b")
            plt.plot(led_time, leds * q_led, "r")
            plt.plot(mix_time, vad_snd * q_vad, "g")
            plt.plot(mix_time, vad_guarded * q_vad, "g--")
            plt.legend(["mix", "VAD"])
            plt.title("LED and mix signals")

            plt.figure()
            a_time = np.arange(mix.shape[0]) / fs_snd
            plt.plot(a_time, mix[:, 0])
            plt.plot(a_time, out_trunc)
            plt.legend(["channel 0", "beamformer output", "speech reference"])
            
            # SIGNAL IN FREQ DOMAIN
            n_frames_n = np.sum(np.mean(np.abs(X_noise)[...,0], axis=1) > 1e-6 ) # number of frame where noise is active
            n_frames_s = np.sum(np.mean(np.abs(X_speech)[...,0], axis=1) > 1e-6 )
            X_n_freq = np.sum(np.abs(X_noise[...,0])**2, axis=0) / n_frames_n
            X_s_freq = np.sum(np.abs(X_speech[...,0])**2, axis=0) / n_frames_s
            
            freqs_to_plot = [125.0, 218.75, 406.25, 500.0, 718.75, 1218.75] # Hz, manual
            plt.figure()
            plt.semilogy(freqs, X_n_freq, label='noise')
            plt.semilogy(freqs, X_s_freq, label='speech')
            for freq in freqs_to_plot:
                plt.axvline(x=freq, color='r', linestyle='--')
            plt.xlim([0, 2000])
            plt.legend()        
            plt.title('Spectrum of the signals, channel 0, salient frequencies')

            plt.figure()
            mic_array.plot_beam_response()
            plt.vlines(
                [180 + np.degrees(theta_speech), 180 - np.degrees(theta_noise)],
                0,
                nfft // 2,
            )
                    
            # plot beamformer weights
            theta = np.deg2rad(np.arange(-180, 180, 1))
            vect_doa_src = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
            vect_mics = room.mic_array.R
            toas_far_free = vect_doa_src.T @ vect_mics / room.c # nDoas x nChan
            svects = np.exp(- 1j * 2 * np.pi * freqs[:,None,None] * (toas_far_free[None,...])) #  nFreq x nDoas x nChan
            
            bf_freq_abs = np.abs(np.einsum('fm,fsm->fs', w, svects))**2
            bf_freq_abs /= np.max(bf_freq_abs)
            
            fig_pol, axarr_pol = plt.subplots(1, 2, figsize=(8,4), sharey=True, subplot_kw={'projection': 'polar'})
            fig_lin, axarr_lin = plt.subplots(1, 2, figsize=(8,4), sharey=True)

            mean_bf_abs = np.mean(bf_freq_abs, axis=0)
            mean_bf_abs /= np.max(mean_bf_abs)

            for src_idx, src_name in zip([0, 1], ['target', 'interf']):

                doas_ = source_echoes[src_idx]['doas']
                coeff_ = source_echoes[src_idx]['coeffs']
                coeff_ /= np.max(np.abs(coeff_))
                walls_ = source_echoes[src_idx]['walls']
                order_ = source_echoes[src_idx]['order']

                axarr_pol[src_idx].plot(theta, mean_bf_abs, label='BF')
                axarr_lin[src_idx].plot(theta, mean_bf_abs, label='BF')
                for d, (doa, coeff, wall) in enumerate(zip(doas_, coeff_, walls_)):
                    
                    # skip ceiling and floor for viz
                    if wall in ['ceiling', 'floor']:
                        continue
                    
                    doa = doa if doa < np.pi else doa - 2*np.pi
                    axarr_pol[src_idx].arrow(doa, 0, 0, coeff, width = 0.015, edgecolor = 'black', facecolor = 'C0', lw = 1, zorder = 5)
                    axarr_pol[src_idx].text(doa, coeff + 0.1, f"{d} - {wall}", fontsize=8, ha='center', va='center')
                    
                    # add text on the tip of the arrow
                    axarr_lin[src_idx].arrow(doa, 0, 0, coeff, width = 0.015, edgecolor = 'black', facecolor = 'C0', lw = 1, zorder = 5)
                    axarr_lin[src_idx].text(doa, coeff + 0.1, f"{d} - {wall}", fontsize=8, ha='center', va='center')

                axarr_lin[src_idx].set_title('{} DOAs'.format(src_name))
                axarr_pol[src_idx].set_title('{} DOAs'.format(src_name))

            fig_pol.suptitle("Beamforming directivity pattern vs sources' DOAs")
            fig_lin.suptitle("Beamforming directivity pattern vs sources' DOAs")
            fig_lin.legend()
            fig_pol.legend()
            fig_lin.tight_layout()
            fig_pol.tight_layout()
            
            def generate_axes(fig):
                gridspec = fig.add_gridspec(nrows=24, ncols=12)
                axes = {}
                axes['spec'] = fig.add_subplot(gridspec[0:24, 0:2])
                
                axes['lin_doas6'] = fig.add_subplot(gridspec[0:4, 2:8])
                axes['pol_doas6'] = fig.add_subplot(gridspec[0:4, 8:10], projection='polar')
                axes['pol_doas6b'] = fig.add_subplot(gridspec[0:4, 10:12], projection='polar')
                axes['lin_doas5'] = fig.add_subplot(gridspec[4:8, 2:8])
                axes['pol_doas5'] = fig.add_subplot(gridspec[4:8, 8:10], projection='polar')
                axes['pol_doas5b'] = fig.add_subplot(gridspec[4:8, 10:12], projection='polar')
                axes['lin_doas4'] = fig.add_subplot(gridspec[8:12, 2:8])
                axes['pol_doas4'] = fig.add_subplot(gridspec[8:12, 8:10], projection='polar')
                axes['pol_doas4b'] = fig.add_subplot(gridspec[8:12, 10:12], projection='polar')
                axes['lin_doas3'] = fig.add_subplot(gridspec[12:16, 2:8])
                axes['pol_doas3'] = fig.add_subplot(gridspec[12:16, 8:10], projection='polar')
                axes['pol_doas3b'] = fig.add_subplot(gridspec[12:16, 10:12], projection='polar')
                axes['lin_doas2'] = fig.add_subplot(gridspec[16:20, 2:8])
                axes['pol_doas2'] = fig.add_subplot(gridspec[16:20, 8:10], projection='polar')
                axes['pol_doas2b'] = fig.add_subplot(gridspec[16:20, 10:12], projection='polar')
                axes['lin_doas1'] = fig.add_subplot(gridspec[20:24, 2:8])
                axes['pol_doas1'] = fig.add_subplot(gridspec[20:24, 8:10], projection='polar')
                axes['pol_doas1b'] = fig.add_subplot(gridspec[20:24, 10:12], projection='polar')
                return axes

            fig = plt.figure(figsize=(16, 12))
            axes = generate_axes(fig)
            axes['spec'].semilogx(X_n_freq, freqs, label='Noise PSD')
            axes['spec'].semilogx(X_s_freq, freqs, label='Speech PSD')
            axes['spec'].invert_xaxis()
            axes['spec'].set_ylim([100, 2000])
            axes['spec'].legend(loc="upper right")

            for f, freq in enumerate(freqs_to_plot):
                freq_idx = np.argmin(np.abs(freqs - freq))
                curr_bf = bf_freq_abs[freq_idx]
                curr_bf_normalized = curr_bf / np.max(curr_bf)
                axes[f'lin_doas{f+1}'].plot(np.rad2deg(theta), curr_bf, 'C0', label=f'BF at {freq:.0f} Hz')
                axes[f'lin_doas{f+1}'].plot(np.rad2deg(theta), curr_bf_normalized, 'C0--', label=f'BF normalized', alpha=0.5)
                
                axes[f'pol_doas{f+1}'].plot(theta, curr_bf, 'C0', label=f'{freq:.0f} Hz')
                axes[f'pol_doas{f+1}'].plot(theta, curr_bf_normalized, 'C0--', label=f'{freq:.0f} Hz, normalized', alpha=0.5)
                axes[f'pol_doas{f+1}b'].plot(theta, curr_bf, 'C0', label=f'{freq:.0f} Hz')
                axes[f'pol_doas{f+1}b'].plot(theta, curr_bf_normalized, 'C0--', label=f'{freq:.0f} Hz, normalized', alpha=0.5)
                
                axes[f'pol_doas{f+1}'].set_xticks([])
                axes[f'pol_doas{f+1}'].set_yticks([])
                axes[f'pol_doas{f+1}b'].set_xticks([])
                axes[f'pol_doas{f+1}b'].set_yticks([])
                
                for s, src_name, color, plot in zip([0, 1], ['target', 'interf'], ['C0', 'C1'], ['', 'b']):
                    doas_ = source_echoes[s]['doas']
                    doas_[doas_ > np.pi] -= 2*np.pi
                    coeff_ = source_echoes[s]['coeffs']
                    coeff_ /= np.max(coeff_)
                    walls_ = source_echoes[s]['walls']
                    for i, (doa, coeff, wall) in enumerate(zip(doas_, coeff_, walls_)):
                        # skip ceiling and floor for viz
                        if wall in ['ceiling', 'floor']:
                            continue
                        if i == 0:
                            axes[f'pol_doas{f+1}{plot}'].arrow(doa, 0, 0, coeff, width = 0.015, edgecolor = color, facecolor = color, lw = 1, zorder = 5, label=src_name)
                        else:
                            axes[f'pol_doas{f+1}{plot}'].arrow(doa, 0, 0, coeff, width = 0.015, edgecolor = color, facecolor = color, lw = 1, zorder = 5)
                    
                    axes[f'lin_doas{f+1}'].stem(np.rad2deg(doas_), coeff_, 
                                                markerfmt=f'{color}^', 
                                                linefmt=f'{color}',
                                                basefmt=' ', label=src_name)
                            
                axes[f'lin_doas{f+1}'].legend(loc='upper center', ncols=3)
                
                if f == len(freqs_to_plot) - 1:
                    axes[f'pol_doas{f+1}'].set_title(f"{freq:.0f} Hz - target's DOAs")
                    axes[f'pol_doas{f+1}b'].set_title(f"{freq:.0f} Hz - interf's DOAs")
                else:
                    axes[f'pol_doas{f+1}'].set_title(f"{freq:.0f} Hz")
                    axes[f'pol_doas{f+1}b'].set_title(f"{freq:.0f} Hz")
                
                axes['spec'].axhline(freq, color='k', linestyle='--')
                axes['spec'].text(0.04, freq, f"{freq:.0f} Hz", ha='center', va='bottom')

            axes[f'pol_doas6'].set_title(f"{freq:.0f} Hz - {src_name}'s DOAs")
            axes[f'pol_doas6b'].set_title(f"{freq:.0f} Hz - {src_name}'s DOAs")

            plt.suptitle(f'Beamforming directivity pattern vs sources DOAs at {SIR} dB')
            plt.tight_layout()            
            
            ## PLOT 2D ROOM WITH IMAGES
            fig, ax = plt.subplots()
            plt.scatter(room.mic_array.center[0], room.mic_array.center[1], c='C0', marker='x', label='pyr')
            
            source_color = ['C1', 'C3']
            source_name = ['target', 'interf']
            for j in range(len(room.sources)):
                plt.scatter(room.sources[j].position[0], room.sources[j].position[1], c=source_color[j], label=source_name[j])
                for n in range(len(source_echoes[j]['images'][0])):
                    if n == 0:
                        plt.scatter(source_echoes[j]['images'][0,n], source_echoes[j]['images'][1,n], c=source_color[j], alpha=0.5, label='target')
                    else:
                        plt.scatter(source_echoes[j]['images'][0,n], source_echoes[j]['images'][1,n], c=source_color[j], alpha=0.5)
                    # for each image, add the name of the wall as text from the list source_echoes[j]['walls'][n]
                    plt.text(source_echoes[j]['images'][0,n], source_echoes[j]['images'][1,n], source_echoes[j]['walls'][n], fontsize=8)
                    
            rect = patches.Rectangle((0, 0), room_dim[0], room_dim[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.legend()

            # spectrograms
            fig, ax = plt.subplots(figsize=(12, 4))
            img = lr.display.specshow(lr.amplitude_to_db(np.abs(X_speech[...,0].T), ref=np.max), y_axis='log', x_axis='time', ax=ax, sr=fs)
            ax.set_title('X_speech')
            fig.colorbar(img, ax=ax, format="%+2.0f dB")

            fig, ax = plt.subplots(figsize=(12, 4))
            img = lr.display.specshow(lr.amplitude_to_db(np.abs(X_noise[...,0].T), ref=np.max), y_axis='log', x_axis='time', ax=ax, sr=fs)
            ax.set_title('X_noise')
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
        
            ## PLOT 3D BEAMFORMER RADIATION PATTERN
            azimuth = np.linspace(0, 2 * np.pi, 100)
            elevation = np.linspace(0, np.pi, 50)
            # Create a grid for azimuth and elevation
            azimuth, elevation = np.meshgrid(azimuth, elevation)
            mesh_shape = azimuth.shape
            azimuth = azimuth.flatten()
            elevation = elevation.flatten()
            vect_doa_src = np.stack([np.cos(azimuth) * np.sin(elevation), np.sin(azimuth) * np.sin(elevation), np.cos(elevation)])
            vect_mics = room.mic_array.R
            toas_far_free = vect_doa_src.T @ vect_mics / room.c # nDoas x nChan
            svects = np.exp(- 1j * 2 * np.pi * freqs[:,None,None] * (toas_far_free[None,...])) #  nFreq x nDoas x nChan
            
            # Assuming a unit sphere for visualization
            r = np.abs(np.einsum('fm,fsm->fs', bf_weights, svects))**2
            r = np.mean(r, axis=0)
            r /= np.max(r)
                    
            # Convert spherical coordinates to Cartesian coordinates
            X = r * np.sin(elevation) * np.cos(azimuth)
            Y = r * np.sin(elevation) * np.sin(azimuth)
            Z = r * np.cos(elevation)
            
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(111, projection='3d')

            # Plot the surface
            ax.plot_surface(
                X.reshape(*mesh_shape),
                Y.reshape(*mesh_shape),
                Z.reshape(*mesh_shape),
                facecolors=plt.cm.viridis(r.reshape(*mesh_shape)), 
                rstride=1, cstride=1, alpha=0.8
            )
            
            N = np.sqrt(X**2 + Y**2 + Z**2)
            Rmax = np.max(N)
            axes_length = 0.65
            ax.plot([0, axes_length*Rmax], [0, 0], [0, 0], linewidth=2, color='red')
            ax.plot([0, 0], [0, axes_length*Rmax], [0, 0], linewidth=2, color='green')
            ax.plot([0, 0], [0, 0], [0, axes_length*Rmax], linewidth=2, color='blue')

            # Customize the plot
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Beamforming Weights')
        
        ## Correlation plots
        fig, axarr = plt.subplots(2, 1, figsize=(12, 8))
        
        fig.suptitle('Correlation between steering vectors and weights for given angles')
        source_name = ['target', 'interf']
        for j in range(len(room.sources)):
            images_ = source_echoes[j]['images']
            doas_ = source_echoes[j]['doas']
            coeff_ = source_echoes[j]['coeffs']
            walls_ = source_echoes[j]['walls']
            order_ = source_echoes[j]['order']
            
            N = images_.shape[1]
            
            vect_doas = images_ / np.linalg.norm(images_, axis=0)
            vect_mics = room.mic_array.R
            toas_far_free = vect_doas.T @ vect_mics / room.c # nDoas x nChan
            svects = np.exp(- 1j * 2 * np.pi * freqs[:,None,None] * (toas_far_free[None,...])) #  nFreq x nDoas x nChan
            
            correlation = np.abs(np.einsum('fm,fsm->fs', bf_weights, svects))**2 / (np.linalg.norm(bf_weights, axis=1)[:,None] * np.linalg.norm(svects, axis=2)) # [nFreq x nDoas]
            
            img = axarr[j].imshow(correlation, aspect='auto', origin='lower', interpolation='nearest')
            plt.colorbar(img, ax=axarr[j])
            
            axarr[j].set_xlabel('DOAs "wall - order"')
            axarr[j].set_xticks(np.arange(N))
            axarr[j].set_xticklabels([f"{walls_[i]} - {order_[i]}" for i in range(N)], rotation=45, ha='right')
            
            axarr[j].set_ylabel('Freqs index')
            axarr[j].set_title(source_name[j])
        fig.tight_layout()
        
        # plot all the plots
        plt.show()        

        
    # Return SDR and SIR
    return SDR_out, SIR_out


if __name__ == "__main__":

    args = parser.parse_args()

    if args.all:
        results = dict()
        for mic in mic_choices.keys():
            results[mic] = {"SIR_in": [], "SIR_out": [], "SDR_out": []}
            for SIR in sir_choices:
                results[mic]["SIR_in"].append(SIR)
                sdr_o, sir_o = process_experiment_max_sinr(SIR, mic, args)
                results[mic]["SIR_out"].append(sir_o)
                results[mic]["SDR_out"].append(sdr_o)

        import datetime

        now = datetime.datetime.now()
        date_str = datetime.datetime.strftime(now, "%Y%m%d-%H%M%S")
        filename = "figures/{}_results_experiment_sir.json".format(date_str)

        parameters = dict(
            nfft=args.nfft,
            vad_guard=args.vad_guard,
            clip_gain=args.clip_gain,
            thresh=args.thresh,
            no_norm=args.no_norm,
            synth_mix=args.synth_mix,
        )

        record = dict(
            parameters=parameters,
            results=results,
        )

        with open(filename, "w") as f:
            json.dump(record, f)

    else:
        try:
            SIR = args.SIR
            mic = args.mic
        except:
            raise ValueError(
                "When the keyword --all is not used, SIR and mic are required arguments"
            )

        SDR_out, SIR_out = process_experiment_max_sinr(SIR, mic, args)
