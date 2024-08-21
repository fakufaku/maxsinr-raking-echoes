import numpy as np
from scipy import linalg as la
from max_sinr_beamforming import compute_gain
from geo_utils import distance

def compute_steering_vector(src_pos, mic_pos, c, freqs, ref_mic_idx=None, mode="far"):
    assert len(src_pos.shape) ==  len(mic_pos.shape) == 2
    assert src_pos.shape[0] ==  mic_pos.shape[0] == 3
    if mode == "near":
        toas = distance(src_pos, mic_pos) / c
    elif mode == "far":
        mic_center = np.mean(mic_pos, axis=1, keepdims=True)
        uvect_to_src = src_pos - mic_center
        uvect_to_src = uvect_to_src / np.linalg.norm(uvect_to_src, axis=0)
        uvect_to_mic = mic_pos - mic_center
        toas = -1. * uvect_to_src.T @ uvect_to_mic / c # [1 x nMics]
        toas -= np.min(toas)
    # compute steeering vectors
    a1 = np.exp(- 1j * 2 * np.pi * freqs[:,None,None] * toas[None,:,:]) # [nFreq x nMic]
    a1 = a1 / a1[:,:,[ref_mic_idx]]
    return a1

def mvdr_weights(image_pos, mic_pos, c, freqs, Rn, ref_mic_idx, reg=0.):
    a1 = compute_steering_vector(image_pos, mic_pos, c, freqs, ref_mic_idx)
    assert len(a1.shape) == 3
    assert a1.shape[1] == 1
    a1 = np.sum(a1, axis=1)
    # compute optimal weights
    invRn = np.linalg.inv(Rn + reg * np.eye(Rn.shape[1]))
    invRn_a1 = np.einsum("fij,fj->fi", invRn, a1)
    a1H_invRn_a1 = np.einsum("fi,fi->f", np.conj(a1), invRn_a1)
    w = invRn_a1 / a1H_invRn_a1[:, None]
    
    # Rs = np.einsum("fi,fj->fij", a1, a1.conj())
    # invRn = np.linalg.inv(Rn)
    # num = np.einsum("fij,fjk->fik", invRn, Rs)
    # denom = np.trace(num, axis1=1, axis2=2)
    # w = (num / denom[:,None,None])[:,:,ref_mic]
    
    # # check beamforming condition
    assert np.allclose(np.einsum("fi,fi->f", w.conj(), a1), np.ones(w.shape[0]) + 1j*0)
    return w

def lcmv_weights(image_pos_good, image_pos_bad, mic_pos, c, freqs, Rn, ref_mic_idx, reg=1e-6):
    a1_good = compute_steering_vector(image_pos_good, mic_pos, c, freqs, ref_mic_idx)
    assert len(a1_good.shape) == 3
    assert a1_good.shape[1] == 1
    a1_good = np.sum(a1_good, axis=1)
    
    a1_bad = compute_steering_vector(image_pos_bad, mic_pos, c, freqs, ref_mic_idx)
    assert len(a1_bad.shape) == 3
    assert a1_bad.shape[1] == 1
    a1_bad = np.sum(a1_bad, axis=1)
    
    A = np.stack([a1_good, a1_bad], axis=-1)
    q = np.array([1, 0])
    # compute optimal weights
    invRn = np.linalg.inv(Rn + reg * np.eye(Rn.shape[-1]))
    invRn_A = np.einsum("fij,fjk->fik", invRn, A)
    AH_invRn_A = np.einsum("fik,fiK->fkK", np.conj(A), invRn_A)
    inv_AH_invRn_A = np.linalg.inv(AH_invRn_A + reg * np.eye(AH_invRn_A.shape[-1]))
    w = np.einsum("fik,fkK->fiK", invRn_A, inv_AH_invRn_A)
    w = np.einsum("fik,k->fi", w, q)
    
    # Rs = np.einsum("fi,fj->fij", a1, a1.conj())
    # invRn = np.linalg.inv(Rn)
    # num = np.einsum("fij,fjk->fik", invRn, Rs)
    # denom = np.trace(num, axis1=1, axis2=2)
    # w = (num / denom[:,None,None])[:,:,ref_mic]
    
    # # check beamforming condition
    assert np.allclose(np.einsum("fi,fi->f", w.conj(), a1_good), np.ones(w.shape[0]) + 1j*0)
    assert np.allclose(np.einsum("fi,fi->f", w.conj(), a1_bad), np.zeros(w.shape[0]) + 1j*0)
    return w

def rake_weights(image_pos, mic_pos, c, freqs, Rn, ref_mic_idx, reg=0.):
    a1 = compute_steering_vector(image_pos, mic_pos, c, freqs, ref_mic_idx)
    assert len(a1.shape) == 3
    a1 = np.sum(a1, axis=1)
    
    # compute optimal weights
    invRn = np.linalg.inv(Rn + reg * np.eye(Rn.shape[1]))
    invRn_a1 = np.einsum("fij,fj->fi", invRn, a1)
    a1H_invRn_a1 = np.einsum("fi,fi->f", np.conj(a1), invRn_a1)
    w = invRn_a1 / a1H_invRn_a1[:, None]
    
    # Rs = np.einsum("fi,fj->fij", a1, a1.conj())
    # invRn = np.linalg.inv(Rn)
    # num = np.einsum("fij,fjk->fik", invRn, Rs)
    # denom = np.trace(num, axis1=1, axis2=2)
    # w = (num / denom[:,None,None])[:,:,ref_mic]
    
    # # check beamforming condition
    assert np.allclose(np.einsum("fi,fi->f", w.conj(), a1), np.ones(w.shape[0]) + 1j*0)
    return w

def delay_and_sum_weights(image_pos, mic_pos, c, freqs, ref_mic_idx):
    a1 = compute_steering_vector(image_pos, mic_pos, c, freqs, ref_mic_idx)
    assert len(a1.shape) == 3
    assert a1.shape[1] == 1
    a1 = np.sum(a1, axis=1)
    # compute optimal weights
    w = a1 / np.einsum("fi,fi->f", a1.conj(), a1)[:,None]
    assert np.allclose(np.einsum("fi,fi->f", a1.conj(), w), np.ones(w.shape[0]) + 1j*0)
    return w

def souden_weights(Rn, Rs, X_speech, ref_chan_idx, no_norm, clip_gain):
    n_channels = Rs.shape[-1]
    # compute optimal weights
    invRn = np.linalg.inv(Rn[1:])
    num = np.einsum("fij,fjk->fik", invRn, Rs[1:])
    denom = np.trace(num, axis1=1, axis2=2)
    w = (num / denom[:,None,None])[:,:,ref_chan_idx]
    # Post processing of the weights
    nw = la.norm(w, axis=1)
    w[nw > 1e-10, :] /= nw[nw > 1e-10, None]
    w = np.concatenate([np.ones((1, n_channels)), w], axis=0)
    
    if not no_norm:
        # normalize with respect to input signal
        z = compute_gain(w, X_speech, X_speech[:, :, ref_chan_idx], clip_up=clip_gain)
        w *= z[:, None]
    return w

def max_sinr_weights(Rs, Rn):
    n_channels = Rs.shape[-1]
    w = [
        la.eigh(
            rs,
            b=rn,
            eigvals=(n_channels - 1, n_channels - 1),
        )[1]
        for rs, rn in zip(Rs, Rn)
    ]
    w = np.squeeze(np.array(w))
    return w