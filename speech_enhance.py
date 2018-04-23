# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 13:24:10 2015

@author: lwang
"""

#import wave
import scipy
#import struct
import numpy as np
import pylab as pl
import mne
from figure2pdf import figure2pdf
from speech_procfuns import *
import time

start_time = time.time()

# Main function ##
audioroot = 'C:/Users/lwang/My Work/EEG_Experiments/Experiment_Scripts/Ex012_STHLAlgrth/quicksin_ANL/original_stimuli/separated_lists/'
timingroot = 'C:/Users/lwang/My Work/EEG_Experiments/Experiment_Scripts/Ex012_STHLAlgrth/quicksin_ANL/timings/separated_lists/'

pitch_range = np.asarray([50, 500])
clip_thr = 0.7
energy_thr = 0.4 
winsize = 0.06
shiftsize = 0.015

alpha = 0.9  # comb filter coefficient

for i in range(1,2):

    fname = 'list_' + str(i).zfill(2) +'.wav'
    print 'Processing ' + fname
    
    T, nbits, data, nframes, nchannels, sampling_frequency = read_wavfile(audioroot+fname)
    
    a = np.asarray(data)
    
    y = np.zeros((nframes, nchannels))
    
    
    normfactor = 2**(nbits*8-1)
    
    y[:,0] = 1.0*a[0::nchannels]/normfactor
    y[:,1] = 1.0*a[1::nchannels]/normfactor
    
    timing_file = open(timingroot+fname[:-4]+'.txt','r')
    line1 = timing_file.readline()
    words = line1.split()
    s1_start = int(float(words[0])*sampling_frequency)
    s1_end = int(float(words[1])*sampling_frequency)
    s1_inds = np.asarray(range(s1_start+int((s1_end-s1_start)/3.0), s1_end-int((s1_end-s1_start)/3.0)))
    
    #s1_inds = np.asarray(range(544355, 675403))
    s1_inds = np.asarray(range(181452, 372984))
    y = y[s1_inds,:]
    
    nframes = y.shape[0]
    yout = np.zeros((nframes, nchannels))
    
#    ##### Speech in pink noise demo  ####
#    noise = np.random.normal(loc=0, scale=1, size=(nframes,))
#    Noise = np.fft.rfft(noise)
#    f = np.fft.fftfreq(nframes, d=1./sampling_frequency)[:len(Noise)]
#    f[-1]=np.abs(f[-1])
#    alpha = 0.9
#    gain = np.ones_like(f)
#    gain = np.abs((f**(-alpha)))
#    gain[gain>1] = 1
#    Noise_pink = Noise * gain
#    noise_pink = np.fft.irfft(Noise_pink)    
#    noise_pink = noise_pink/rms(noise_pink)*rms(y)
#    
#    
#    
#    SIN_demo = noise_pink+y[:,0]*0.2
#    SIN_demo = SIN_demo/np.abs(SIN_demo).max()
    
    
#    #### Speech-shaped noise
#    Y = np.fft.fft(y[:,1])
#    ssNoise = abs(Y)*np.exp(np.random.uniform(low=0,high=1,size=y[:,1].shape)*1j*np.pi*2)
#    ssnoise = np.real((np.fft.ifft(ssNoise)))
#    ssnoise = ssnoise/rms(ssnoise)*rms(y[s1_inds,0])
    
    ####  babble noise
    babble = y[:,1]
    babble = babble/rms(babble)*rms(y[:,0])
    
    
    
    #### envelope enhancement  ####
    #ymix = y[:,0]+y[:,1]
    #
    #cutoff_freq = [150, 550, 1550, 3550, 8000]
    #k = 1
    #ymix_enh = np.zeros((nframes,))
    #ymix_lp = mne.filter.low_pass_filter(ymix, sampling_frequency, cutoff_freq[0])
    #ymix_hp = mne.filter.high_pass_filter(ymix, sampling_frequency, cutoff_freq[-1])
    #ymix_enh = ymix_lp + ymix_hp
    #for i in range(len(cutoff_freq)-1):
    #    ymix_bp = mne.filter.band_pass_filter(ymix, sampling_frequency, cutoff_freq[i], cutoff_freq[i+1])
    #    ymix_hilt = np.abs(scipy.signal.hilbert(ymix_bp))
    #    ymix_env = mne.filter.low_pass_filter(ymix_hilt, sampling_frequency, 16)
    #    
    #    ymix_enh_bp = (ymix_env**k)*ymix_bp
    #    ymix_enh_bp = mne.filter.band_pass_filter(ymix_enh_bp, sampling_frequency, cutoff_freq[i], cutoff_freq[i+1])
    #    ymix_enh_bp = ymix_enh_bp/rms(ymix_enh_bp)*rms(ymix_bp)
    #    ymix_enh = ymix_enh + ymix_enh_bp
    #    
    #    
    #stft_mix = stft(ymix, sampling_frequency, framesz, hop)
    #pxx_mix = db(np.square(scipy.absolute(stft_mix)))
    #stft_mix_enh = stft(ymix_enh, sampling_frequency, framesz, hop)
    #pxx_mix_enh = db(np.square(scipy.absolute(stft_mix_enh)))
    #
    ## Plot the magnitude spectrogram.
    #pl.figure()
    #pl.imshow(pxx_mix[:,0:int(pxx_mix.shape[1]/2)+1].T, origin='lower', aspect='auto', extent=[0, T, 0, sampling_frequency/2],
    #             interpolation='nearest')
    #pl.xlabel('Time')
    #pl.ylabel('Frequency')
    #pl.show()
    #
    #pl.figure()
    #pl.imshow(pxx_mix_enh[:,0:int(pxx_mix_enh.shape[1]/2)+1].T, origin='lower', aspect='auto', extent=[0, T, 0, sampling_frequency/2],
    #             interpolation='nearest')
    #pl.xlabel('Time')
    #pl.ylabel('Frequency')
    #pl.show()
    
    #ymix_enh = ymix_enh/np.abs(ymix_enh).max()
    
    
    
    
    #### pitch extraction ####

    
#    yclean = y[:,0]    
#    
#    yclean_mbenh = multband_pitch_enhance(yclean, alpha, sampling_frequency, winsize, shiftsize, pitch_range, clip_thr, energy_thr)
#    
#    yout = np.zeros(y.shape)
#    
#    yout[:,0] = yclean/np.max(abs(yclean))
#    yout[:,1] = yout[:,0]
#    write_wavfile('yclean_orig.wav', yout*0.95, nframes, nchannels, sampling_frequency, nbits)
#    
#    yout[:,0] = yclean_mbenh/np.max(abs(yclean_mbenh))
#    yout[:,1] = yout[:,0]
#    write_wavfile('yclean_mb_enh.wav', yout*0.95, nframes, nchannels, sampling_frequency, nbits)
        
    for SNR in range(-5, 5, 5):
        ymix = y[:,0] + babble*db2mag(-SNR)
        ymix[ymix>1] = 1
        ymix[ymix<-1] = -1
        
        ymix_mbenh = multband_pitch_enhance(ymix, alpha, sampling_frequency, winsize, shiftsize, pitch_range, clip_thr, energy_thr)
        
        yout[:,0] = ymix #/np.max(abs(ymix))
        yout[:,1] = yout[:,0]
        write_wavfile(fname[:-4]+'_'+str(SNR)+'dB_orig.wav', yout, nframes, nchannels, sampling_frequency, nbits)   
        
        yout[:,0] = ymix_mbenh #/np.max(abs(ymix_mbenh))
        yout[:,1] = yout[:,0]
        write_wavfile(fname[:-4]+'_'+str(SNR)+'dB_mbenh2.wav', yout*0.95, nframes, nchannels, sampling_frequency, nbits)    
        
    end_time = time.time()
    
    print end_time-start_time, ' seconds'
    
    #### save the processed waveform in wav file #####
    #yout = np.zeros((len(y_mix[s1_inds]),2))
    #yout[:,0] = y_s1mix_enh/np.max(abs(y_s1mix_enh))
    #yout[:,1] = yout[:,0]
    #write_wavfile('y_s1mix_enh.wav', yout*0.95, nframes, nchannels, sampling_frequency, nbits)