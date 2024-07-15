"""
This file applies a max SINR approach using the VAD information
from the LED and the two channels from the camera

Author: Robin Scheibler
Created: 2017/12/01
"""

import argparse, os, json, sys
import numpy as np
import scipy.linalg as la
from scipy.io import wavfile
import pyroomacoustics as pra
import sounddevice as sd
from mir_eval.separation import bss_eval_images

import matplotlib.pyplot as plt

# import seaborn as sns
# sns.set_palette('pastel')

import numpy.fft as fft

from max_sinr_beamforming import compute_variances, compute_gain

experiment_folder = "datanet/projects/otohikari/robin/measurements/20171207"
file_pattern = os.path.join(experiment_folder, "segmented/{}_{}_SIR_{}_dB.wav")
file_speech_ref = os.path.join(experiment_folder, "segmented/{}_speech_ref.wav")
protocol_file = os.path.join(experiment_folder, "protocol.json")

with open(protocol_file, "r") as f:
    protocol = json.load(f)

# thresh_opt = { 5 : 530, 10 : 460, 15 : 370, 20 : 260, 25 : 220 }  # 2017/11/29
thresh_opt = {5: 650, 10: 600, 15: 540, 20: 425, 25: 340}  # 2017/12/07


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
    type=str,
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
    fs_snd, audio = wavfile.read(file_pattern.format(mic_choices[mic], "mix", SIR))
    assert fs_led == fs_snd

    # read in the ref signals
    r, noise_ref = wavfile.read(file_pattern.format(mic_choices[mic], "noise_ref", SIR))
    assert r == fs_snd
    r, speech_ref = wavfile.read(file_speech_ref.format(mic_choices[mic]))
    assert r == fs_snd
    r, leds_ref = wavfile.read(file_speech_ref.format("camera_leds_zero_hold"))
    assert r == fs_snd

    # In case of objective evaluation, we do an artificial mix
    if args.synth_mix:
        audio = noise_ref + speech_ref

    # get the geometry information to get nice plots.
    mics_loc = np.array(protocol["geometry"]["microphones"][mic]["reference"])
    noise_loc = protocol["geometry"]["speakers"]["locations"][0]
    speech_loc = protocol["geometry"]["speakers"]["locations"][1]

    # the directions of arrival
    theta_speech = 0
    p0 = speech_loc - mics_loc
    p1 = noise_loc - mics_loc
    theta_noise = np.arccos(np.inner(p0, p1) / la.norm(p0) / la.norm(p1))
    print("Source separation", theta_noise / np.pi * 180)

    if mic == "pyramic":
        I = list(range(8, 16)) + list(range(24, 32)) + list(range(40, 48))  # flat part
        # I = list(range(24,32)) + list(range(40,48)) # flat part
        # I = list(range(8,16))
        # I = list(range(48))
        audio = audio[:, I]
        noise_ref = noise_ref[:, I].copy()
        speech_ref = speech_ref[:, I].copy()
        mics_positions = mics_geom["pyramic"][I].copy()
        # place in room 2-806
        mics_positions -= np.mean(mics_positions, axis=0)[None, :]
        mics_positions[:, 2] -= np.max(mics_positions[:, 2])
        mics_positions += mics_loc

    elif mic == "olympus":
        mics_positions = mics_geom["olympus"].copy() + mics_loc

    n_samples = audio.shape[0]  # shorthand
    n_channels = audio.shape[1]

    # perform VAD
    vad_snd = leds > vad_thresh

    # Now we want to make sure no speech speech goes in estimation of the noise covariance matrix.
    # For that we will remove frames neighbouring the detected speech
    vad_guarded = vad_snd.copy()
    if vad_guard is not None:
        for i, v in enumerate(vad_snd):
            if np.any(vad_snd[i - vad_guard : i + vad_guard]):
                vad_guarded[i] = True

    ##############################
    ## STFT and frame-level VAD ##
    ##############################

    print("STFT and stuff")
    sys.stdout.flush()

    engine = pra.transform.stft.STFT(
        nfft, nfft // 2, pra.hann(nfft), channels=audio.shape[1]
    )

    def analysis(x):
        engine.analysis(x)
        return engine.X

    # Now compute the STFT of the microphone input
    print('audio', audio.shape)
    X = analysis(audio)
    X_time = np.arange(1, X.shape[0] + 1) * (nfft / 2) / fs_snd
    print(X.shape)

    X_speech = analysis(audio * vad_guarded[:, None])
    X_noise = analysis(audio * (1 - vad_guarded[:, None]))

    S_ref = analysis(speech_ref)
    N_ref = analysis(noise_ref)

    ##########################
    ## MAX SINR BEAMFORMING ##
    ##########################

    print("Max SINR beamformer computation")
    sys.stdout.flush()

    # covariance matrices from noisy signal
    Rs = np.einsum("i...j,i...k->...jk", X_speech, np.conj(X_speech)) / X_speech.shape[-1]
    Rn = np.einsum("i...j,i...k->...jk", X_noise, np.conj(X_noise)) / X_noise.shape[-1]

    # compute covariances with reference signals to check everything is working correctly
    # Rs = np.einsum('i...j,i...k->...jk', S_ref, np.conj(S_ref))
    # Rn = np.einsum('i...j,i...k->...jk', N_ref, np.conj(N_ref))

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
        import ipdb; ipdb.set_trace()
        z = compute_gain(w, X_speech, X_speech[:, :, 0], clip_up=args.clip_gain)
        w *= z[:, None]

    ###########
    ## APPLY ##
    ###########

    print("Apply beamformer")
    sys.stdout.flush()

    # 2D beamformer
    mic_array = pra.Beamformer(
        mics_positions[:, :2].T, fs=fs_snd, N=nfft, hop=nfft, zpb=nfft
    )
    mic_array.signals = audio.T
    mic_array.weights = w.T

    out = mic_array.process()

    # Signal alignment step
    ref = np.vstack([speech_ref[:, 0], noise_ref[:, 0]])
    # Not sure why the delay is sometimes negative here... Need to check more
    delay = np.abs(int(pra.tdoa(out, speech_ref[:, 0].astype(float), phat=True)))
    if delay > 0:
        out_trunc = out[delay : delay + ref.shape[1]]
        noise_eval = audio[: ref.shape[1], 0] - out_trunc
    else:
        out_trunc = np.concatenate((np.zeros(-delay), out[: ref.shape[1] + delay]))
        noise_eval = audio[: ref.shape[1], 0] - out_trunc
    sig_eval = np.vstack([out_trunc, noise_eval])

    # We use the BSS eval toolbox
    metric = bss_eval_images(ref[:, :, None], sig_eval[:, :, None])

    # we are only interested in SDR and SIR for the speech source
    SDR_out = metric[0][0]
    SIR_out = metric[2][0]

    ##################
    ## SAVE SAMPLES ##
    ##################

    if args.save_sample is not None:

        # for informal listening tests, we need to high pass and normalize the
        # amplitude.
        upper = np.maximum(audio[:, 0].max(), out.max())
        sig_in = pra.highpass(audio[:, 0].astype(np.float) / upper, fs_snd, fc=150)
        sig_out = pra.highpass(out / upper, fs_snd, fc=150)

        f1 = os.path.join(args.save_sample, "{}_ch0_SIR_{}_dB.wav".format(mic, SIR))
        wavfile.write(f1, fs_snd, sig_in)
        f2 = os.path.join(args.save_sample, "{}_out_SIR_{}_dB.wav".format(mic, SIR))
        wavfile.write(f2, fs_snd, sig_out)

    ##########
    ## PLOT ##
    ##########

    if args.plot:

        plt.figure()
        plt.plot(out_trunc)
        plt.plot(speech_ref[:, 0])
        plt.legend(["output", "reference"])

        # time axis for plotting
        led_time = np.arange(leds.shape[0]) / fs_led + 1 / (2 * fs_led)
        audio_time = np.arange(n_samples) / fs_snd

        plt.figure()
        plt.plot(led_time, leds, "r")
        plt.title("LED signal")

        # match the scales of VAD and light to sound before plotting
        q_vad = np.max(audio)
        q_led = np.max(audio) / np.max(leds)

        plt.figure()
        plt.plot(audio_time, audio[:, 0], "b")
        plt.plot(led_time, leds * q_led, "r")
        plt.plot(audio_time, vad_snd * q_vad, "g")
        plt.plot(audio_time, vad_guarded * q_vad, "g--")
        plt.legend(["audio", "VAD"])
        plt.title("LED and audio signals")

        plt.figure()
        a_time = np.arange(audio.shape[0]) / fs_snd
        plt.plot(a_time, audio[:, 0])
        plt.plot(a_time, out_trunc)
        # plt.plot(a_time, speech_ref[:,0])
        plt.legend(["channel 0", "beamformer output", "speech reference"])

        plt.figure()
        mic_array.plot_beam_response()
        plt.vlines(
            [180 + np.degrees(theta_speech), 180 - np.degrees(theta_noise)],
            0,
            nfft // 2,
        )

        room = pra.ShoeBox(protocol["geometry"]["room"][:2], fs=16000, max_order=1)

        room.add_source(noise_loc[:2])  # noise
        room.add_source(speech_loc[:2])  # speech
        room.add_source(protocol["geometry"]["speakers"]["locations"][1][:2])  # signal

        room.add_microphone_array(mic_array)
        room.plot(img_order=1, freq=[800, 1000, 1200, 1400, 1600, 2500, 4000])

        plt.figure()
        mic_array.plot()

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

        print("SDR={} SIR={}".format(SDR_out, SIR_out))
