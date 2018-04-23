# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 10:35:08 2017

@author: lwang
"""

#import wave
import scipy
#import struct
import numpy as np
import pylab as pl
import time
import os
from skimage import measure
from matplotlib.colors import LinearSegmentedColormap

from speech_procfuns import *
from peakdetect import peakdetect
#from figure2pdf import figure2pdf

#resultfolder = 'C:/Users/lwang/My Work/EEG_Experiments/Analysis_Codes/Python/Ex012 (STHL Algorithm)/Algorithm/'
#T, nbits, data, nframes, nchannels, sampling_frequency = read_wavfile(resultfolder+"binary_105_noise.wav")
#a = np.asarray(data)
#y = np.zeros((nframes, nchannels))
#normfactor = 2**(nbits*8-1)
#n = 1.0*a[0::nchannels]/normfactor

segment_x1_allrep = list()
segment_x2_allrep = list()
segment_x3_allrep = list()
segment_x4_allrep = list()
segment_x5_allrep = list()

for rep in range(1000):
    print '#'+str(rep+1) + ' of 20 repetitions'
    nframes = 10000
    sampling_frequency = 16000.0
    noise = np.random.normal(loc=0, scale=1, size=(nframes,))
    Noise = np.fft.rfft(noise)
    f = np.fft.fftfreq(nframes, d=1./sampling_frequency)[:len(Noise)]
    f[-1]=np.abs(f[-1])
    alpha = 0.09
    gain = np.ones_like(f)
    gain[1:] = np.abs((f[1:]**(-alpha)))
    gain[gain>1] = 1
    Noise_pink = Noise * gain
    noise_pink = np.fft.irfft(Noise_pink)    
    n = noise_pink
                
    #### STFT analysis  ####
    
    T = nframes/sampling_frequency
    framesz = 0.02  # with a frame size of 60 milliseconds
    hop = 0.001 # 0.0003125      # and hop size of 15 milliseconds.
    t = scipy.linspace(0, T, nframes, endpoint=False)
    cf_cutoff = 5000
    
    N, N_reassigned = tf_reassignment(n, sampling_frequency, framesz, hop, sigma=5)
    
#    pl.figure()
#    pl.subplot(121)
#    pl.imshow(db(np.abs(N)).T,origin='lower',aspect='auto', extent=[0, T, 0, sampling_frequency], cmap='jet',interpolation='nearest')
#    pl.xlabel('Time')
#    pl.ylabel('Frequency')            
#    pl.ylim((0,cf_cutoff))
#    pl.subplot(122)            
#    pl.imshow(db(np.abs(N_reassigned)).T,origin='lower',aspect='auto', extent=[0, T, 0, sampling_frequency], cmap='jet',interpolation='nearest')
#    pl.xlabel('Time')
#    pl.ylabel('Frequency')            
#    pl.ylim((0,cf_cutoff))
    
    # vowel extraction
    sigma = [4,5,6]
    
    gabor_orig_allsigma, consensus_allsigma = ridge_detection(n, sampling_frequency, framesz, hop,sigma,prct_thr=95, cf_cutoff=cf_cutoff,prune=False,plot=False)
    freq = np.linspace(0,sampling_frequency,num=int(sampling_frequency*framesz),endpoint=False)
    tspan_fft = np.linspace(0,T,gabor_orig_allsigma[0].shape[0])
    
    final_consensus = np.asarray(consensus_allsigma).sum(axis=0)
#    pl.figure()
#    pl.subplot(121)
#    pl.imshow(db(np.abs(gabor_orig_allsigma[1])).T,origin='lower',aspect='auto', extent=[0, T, 0, sampling_frequency], cmap='jet',interpolation='nearest')
#    pl.xlabel('Time')
#    pl.ylabel('Frequency')            
#    pl.ylim((0,cf_cutoff))
#    pl.subplot(122)            
#    pl.imshow((final_consensus>1).T,origin='lower',aspect='auto', extent=[0, T, 0, cf_cutoff], cmap='binary',interpolation='nearest')
#    pl.xlabel('Time')
#    pl.ylabel('Frequency')
    
    ridge_mask = final_consensus>1
    tlen,flen = ridge_mask.shape
    
    lowfspan=freq[freq<=cf_cutoff]
    framesize = 10
    n_frames = np.floor(tlen/framesize).astype(int)
    pitch_range = np.arange(80,300,1).astype(float)
    pitch_est = np.zeros((n_frames,))
    pitch_str = np.zeros((n_frames,))
    num_harmonics_thresh = 1
    pitch_jump_thresh = 10
    harm_freq_margin = 0.1
    pitch_cand = {}
    pitch_cand_str = {}
    segments_str = np.zeros(ridge_mask.shape+(n_frames,))
    segments_to_keep = np.zeros(ridge_mask.shape)
    harmonics_to_keep = np.zeros(ridge_mask.shape)
    segments_harm_num = np.zeros(ridge_mask.shape)*np.nan
    segments_slope = np.zeros(ridge_mask.shape)*np.nan
    segments_harm_dist = np.zeros(ridge_mask.shape)*np.nan
    segments_parallelism = np.zeros(ridge_mask.shape)*np.nan
    
    for i_tbin in np.arange(0,n_frames*framesize,framesize):
        CC = measure.regionprops(measure.label(ridge_mask[i_tbin:i_tbin+framesize,:]))
        if len(CC) != 0:
            f_segments = list()
            inds = list()
            slope_segments = list()
            for i in range(len(CC)):
                fmean = freq[CC[i].coords[:,1]].mean()
                if fmean>pitch_range[0]:
                    f_segments.append(fmean)
                    inds.append(CC[i].coords)  
                    x = CC[i].coords[:,0]
                    y = CC[i].coords[:,1]
                    if len(np.unique(x)) > 1:
                        m,b=np.polyfit(x,y,1)
                    else:
                        m = 0
                    slope_segments.append(m)
                    
                        
            f_segments = np.array(f_segments)
            slope_segments = np.array(slope_segments)
            num_harmonics = f_segments[:,None]/pitch_range[None,:]
            resid = np.abs(num_harmonics - np.round(num_harmonics))
            num_harmonics = np.sum(resid<harm_freq_margin,axis=0)
            pitch_cand_inds = num_harmonics>=num_harmonics_thresh
            
            segment_cand_str = np.sum(resid[:,pitch_cand_inds]<harm_freq_margin, axis=1)
            
            for i, ind in enumerate(inds):
                segments_str[ind[:,0]+i_tbin,ind[:,1],(i_tbin/framesize).astype(int)] = segment_cand_str[i]
            
            if np.sum(pitch_cand_inds) > 0:
                pitch_cand[i_tbin] = pitch_range[pitch_cand_inds]
                pitch_cand_str[i_tbin] = num_harmonics[pitch_cand_inds]
                pitch_est[i_tbin/framesize] = pitch_cand[i_tbin][pitch_cand_str[i_tbin]==pitch_cand_str[i_tbin].max()].max()
                pitch_str[i_tbin/framesize] = pitch_cand_str[i_tbin][pitch_cand[i_tbin]==pitch_est[i_tbin/framesize]]
                
                for i, ind in enumerate(inds):
                    x = ind[:,0]
                    y = ind[:,1]
                    segments_harm_num[x+i_tbin,y] = np.round(f_segments[i]/pitch_est[i_tbin/framesize])
                    segments_slope[x+i_tbin,y] = slope_segments[i]
                    segments_harm_dist[x+i_tbin,y] = resid[i, pitch_range==pitch_est[i_tbin/framesize]]
                    segments_parallelism[x+i_tbin,y] = (slope_segments[i]-slope_segments.mean())/(slope_segments.std()+np.finfo(np.float32).eps)
                    
                harmf_est = pitch_est[i_tbin/framesize]*np.arange(1,np.floor(cf_cutoff/pitch_est[i_tbin/framesize]))                    
                harmonics_to_keep[i_tbin:i_tbin+framesize, np.round(harmf_est/freq[1]).astype(int)] += 1
                
    #            segment_ind_to_keep = resid[:, pitch_range==pitch_est[i_tbin/framesize]]<harm_freq_margin
    #            inds2, _ = np.nonzero(segment_ind_to_keep)
    #            if len(inds2) > 0:
    #                for ind2 in inds2:
    #                    segments_to_keep[inds[ind2][:,0]+i_tbin,inds[ind2][:,1]] += 1                                
                        
    segments_str_max = np.max(segments_str,axis=2)
    
    segments_combined_feature = np.zeros(ridge_mask.shape)*np.nan
    temp = np.zeros(ridge_mask.shape)
    CC = measure.regionprops(measure.label(ridge_mask))
    segment_x1 = np.zeros((len(CC),))
    segment_x2 = np.zeros((len(CC),))
    segment_x3 = np.zeros((len(CC),))
    segment_x4 = np.zeros((len(CC),))
    segment_x5 = np.zeros((len(CC),))
    for i, segment in enumerate(CC):
        x = segment.coords[:,0]
        y = segment.coords[:,1]
        segment_x1[i] = segments_harm_dist[x,y].mean()
        segment_x2[i] = segments_harm_dist[x,y].std()
        segment_x3[i] = segments_slope[x,y].std()
        segment_x4[i] = np.abs(segments_parallelism[x,y]).mean()
        segment_x5[i] = segments_harm_num[x,y].std()
        
        segments_combined_feature[x,y] = (segment_x1[i]<0.1)&(segment_x1[i]<0.1)&(segment_x3[i]<1.5)&(segment_x4[i]<1.2)&(segment_x5[i]<1)
        if np.isnan(segment_x1[i]):
            temp[x,y] = 2
        else:
            temp[x,y] = 1
    
    segment_x1_allrep = segment_x1_allrep + list(segment_x1)
    segment_x2_allrep = segment_x2_allrep + list(segment_x2)
    segment_x3_allrep = segment_x3_allrep + list(segment_x3)
    segment_x4_allrep = segment_x4_allrep + list(segment_x4)
    segment_x5_allrep = segment_x5_allrep + list(segment_x5)
    
#    segments_to_keep_pruned = np.zeros(ridge_mask.shape)   
#    segments_to_keep_pruned[segments_combined_feature==1] = 1
#    
#    pl.figure()
#    pl.scatter(np.arange(0,n_frames*framesize,framesize)/1000., pitch_est)
#    pl.xlabel('Time (s)')
#    pl.ylabel('Pitch (Hz)')
#    
#    pl.figure()
#    pl.subplot(121)
#    pl.imshow((temp).T,origin='lower',aspect='auto', extent=[0, T, 0, cf_cutoff], cmap='binary',interpolation='nearest')
#    pl.xlabel('Time')
#    pl.ylabel('Frequency')
#    pl.ylim((0,cf_cutoff))            
#    #            pl.colorbar()
#    pl.subplot(122)
#    pl.imshow(segments_to_keep_pruned.T,origin='lower',aspect='auto', extent=[0, T, 0, cf_cutoff], cmap='binary',interpolation='nearest')
#    pl.xlabel('Time')
#    pl.ylabel('Frequency')
    
#    pl.figure()
#    pl.subplot(221)
#    pl.imshow(segments_harm_dist.T,origin='lower',aspect='auto', extent=[0, T, 0, cf_cutoff], cmap='jet',interpolation='nearest')
#    pl.xlabel('Time')
#    pl.ylabel('Frequency')
#    pl.title('harmonic dist.')
#    pl.colorbar()
#    pl.subplot(222)
#    pl.imshow(segments_slope.T,origin='lower',aspect='auto', extent=[0, T, 0, cf_cutoff], cmap='jet',interpolation='nearest')
#    pl.xlabel('Time')
#    pl.ylabel('Frequency')
#    pl.title('segment slope')
#    pl.colorbar()
#    pl.subplot(223)
#    pl.imshow(np.abs(segments_parallelism).T,origin='lower',aspect='auto', extent=[0, T, 0, cf_cutoff], cmap='jet',interpolation='nearest')
#    pl.xlabel('Time')
#    pl.ylabel('Frequency')
#    pl.title('segment parallelness')
#    pl.colorbar()
#    pl.subplot(224)
#    pl.imshow(segments_harm_num.T,origin='lower',aspect='auto', extent=[0, T, 0, cf_cutoff], cmap='jet',interpolation='nearest')
#    pl.xlabel('Time')
#    pl.ylabel('Frequency')
#    pl.title('harmonic number')
#    pl.colorbar()
    
#    pl.figure()
#    pl.subplot(221)
#    counts, bins = pl.histogram(segment_x1[~np.isnan(segment_x1)], bins=20)
#    pl.bar(bins[:-1], counts,width=0.03*(bins[-1]-bins[0]))
#    pl.title('harmonic dist.')
#    pl.subplot(222)
#    counts, bins = pl.histogram(segment_x2[~np.isnan(segment_x2)], bins=20)
#    pl.bar(bins[:-1], counts,width=0.03*(bins[-1]-bins[0]))
#    pl.title('segment slope')
#    pl.subplot(223)
#    counts, bins = pl.histogram(segment_x3[~np.isnan(segment_x3)], bins=20)
#    pl.bar(bins[:-1], counts,width=0.03*(bins[-1]-bins[0]))
#    pl.title('segment parallelness')
#    pl.subplot(224)
#    counts, bins = pl.histogram(segment_x4[~np.isnan(segment_x4)], bins=20)
#    pl.bar(bins[:-1], counts,width=0.03*(bins[-1]-bins[0]))
#    pl.title('harmonic number')

segment_x1_allrep = np.asarray(segment_x1_allrep)
segment_x2_allrep = np.asarray(segment_x2_allrep)
segment_x3_allrep = np.asarray(segment_x3_allrep)
segment_x4_allrep = np.asarray(segment_x4_allrep)
segment_x5_allrep = np.asarray(segment_x5_allrep)
segment_x1_allrep = segment_x1_allrep[~np.isnan(segment_x1_allrep)]
segment_x2_allrep = segment_x2_allrep[~np.isnan(segment_x2_allrep)]
segment_x3_allrep = segment_x3_allrep[~np.isnan(segment_x3_allrep)]
segment_x4_allrep = segment_x4_allrep[~np.isnan(segment_x4_allrep)]
segment_x5_allrep = segment_x5_allrep[~np.isnan(segment_x5_allrep)]

pl.figure(figsize=(12,10), dpi=100)
pl.subplot(221)
counts, bins = pl.histogram(segment_x2_allrep, bins=200)
pl.plot(bins[:-1], counts)
pl.title('harmonic dist.')
pl.subplot(222)
counts, bins = pl.histogram(segment_x3_allrep, bins=200)
pl.plot(bins[:-1], counts)
pl.title('segment slope')
pl.subplot(223)
counts, bins = pl.histogram(segment_x4_allrep, bins=200)
pl.plot(bins[:-1], counts)
pl.title('segment parallelness')
pl.subplot(224)
counts, bins = pl.histogram(segment_x5_allrep, bins=200)
pl.plot(bins[:-1], counts)
pl.title('harmonic number')

percentage = np.zeros((100,))
for prct_thr in range(0,100):
    thresholds = [0.1, np.percentile(segment_x2_allrep,prct_thr), np.percentile(segment_x3_allrep,prct_thr), np.percentile(segment_x4_allrep,prct_thr), np.percentile(segment_x5_allrep,prct_thr)]
    survived_segments = (segment_x1_allrep<thresholds[0])&(segment_x2_allrep<thresholds[1])&(segment_x3_allrep<thresholds[2])&(segment_x4_allrep<thresholds[3])&(segment_x5_allrep<thresholds[4])
    percentage[prct_thr] = survived_segments.sum().astype(float)/len(segment_x1_allrep)

pl.figure()
pl.plot(np.arange(0,100), percentage)    