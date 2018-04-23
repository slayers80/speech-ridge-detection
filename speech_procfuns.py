# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 14:00:16 2015

@author: lwang
"""

import wave
import scipy
import struct
import numpy as np
from scipy.signal import argrelextrema
from skimage import measure
#from scipy.signal import remez
#from scipy.signal import convolve
import pylab as pl
import mne

def read_wavfile(filename):
    wave_file = wave.open(filename, 'r')
    nframes = wave_file.getnframes()
    nchannels = wave_file.getnchannels()
    sampling_frequency = wave_file.getframerate()
    T = nframes / float(sampling_frequency)
    nbits = wave_file.getsampwidth()
    read_frames = wave_file.readframes(nframes)
    wave_file.close()
    data = struct.unpack("%dh" %  nchannels*nframes, read_frames)
    return T, nbits, data, nframes, nchannels, sampling_frequency
    
def write_wavfile(filename, dataarray_in_float, nframes, nchannels, sampling_frequency, sampwidth):
    wave_file = wave.open(filename, 'w')
    wave_file.setparams((nchannels, sampwidth, sampling_frequency, nframes, 'NONE', 'not compressed'))
    
    normfactor = 2**(sampwidth*8-1)
    
    values = []
    
    for i in range(0, nframes):
        for j in range(0, nchannels):
            value = dataarray_in_float[i,j]*normfactor
            packed_value = struct.pack('h', int(value))
            values.append(packed_value)

    value_str = ''.join(values)
    wave_file.writeframes(value_str)
    wave_file.close()
    
def stft(x, fs, framesz, hop, sigma=0):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)    
        
    freq = np.linspace(0,fs,framesamp+1)  
    
    if sigma == 0:
        w = scipy.hamming(framesamp)
    else:
        N = framesamp
        sigma = (sigma/1000.0)
        t = np.linspace(-N/2+0.5, N/2-0.5, N)/fs
        w = np.exp(-(t/sigma)**2)
        dw = w*t/(sigma**2)*(-2)
    
    if len(x.shape) == 1:
        X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
                         for i in range(0, len(x)-framesamp, hopsamp)])
    else:
        dim = x.shape
        ntrial = np.cumprod(dim[1:])[-1]
        
        y = np.reshape(x, (dim[0],ntrial))        
        for itrial in range(ntrial):
            print '# ' + str(itrial+1) + ' in ' + str(ntrial)
            y1 = y[:,itrial]
            Y1 = scipy.array([scipy.fft(w*y1[i:i+framesamp]) 
                         for i in range(0, len(y1)-framesamp, hopsamp)])
            if itrial == 0:
                Y = np.zeros((Y1.shape[0], Y1.shape[1], ntrial),dtype=np.complex_)
            Y[:,:,itrial] = Y1
        X = np.reshape(Y, (Y1.shape + dim[1:]))
        
    if sigma == 0:
        return X, freq[:-1]
    else:
        if len(x.shape) == 1:
            dwX = scipy.array([scipy.fft(dw*x[i:i+framesamp]) 
                             for i in range(0, len(x)-framesamp, hopsamp)])
                                 
            dtX = scipy.array([scipy.fft(t*w*x[i:i+framesamp]) 
                             for i in range(0, len(x)-framesamp, hopsamp)])                  
        else:
            dim = x.shape
            ntrial = np.cumprod(dim[1:])[-1]
            
            y = np.reshape(x, (dim[0],ntrial))        
            for itrial in range(ntrial):
                print '# ' + str(itrial+1) + ' in ' + str(ntrial)
                y1 = y[:,itrial]
                Y1 = scipy.array([scipy.fft(dw*y1[i:i+framesamp]) 
                             for i in range(0, len(y1)-framesamp, hopsamp)])
                if itrial == 0:
                    Yw = np.zeros((Y1.shape[0], Y1.shape[1], ntrial),dtype=np.complex_)
                Yw[:,:,itrial] = Y1
                
            dwX = np.reshape(Yw, (Y1.shape + dim[1:]))
            dtX = dwX/(-2*(1/sigma**2))  # only valid for gabor window function
        
        dw = dwX/X/(2*np.pi)
        dt = dtX/X
        
        return X, dw, dt, freq[:-1]
 
        

def istft(X, fs, T, hop):
    x = scipy.zeros(int(T*fs))
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)    
        
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    return x    

### Running mean/Moving average
def running_mean(l, N):
    sum = 0
    result = list( 0 for x in l)

    for i in range( 0, N ):
        sum = sum + l[i]
        result[i] = sum / (i+1)

    for i in range( N, len(l) ):
        sum = sum - l[i-N] + l[i]
        result[i] = sum / N

    return result
    
def rms(data_1d):
    x = np.sqrt(np.mean(np.square(data_1d)))
    return x
    
def db(x):
    return 20*np.log10(x)
    
def db2mag(x):
    return 10.0**(x/20.0)    

def clip(x, threshold):
    y = np.zeros(x.shape)
    y[x>threshold] = 1
    y[x<-threshold] = -1
    return y
    
def comb_filter(x, period, alpha=0.6):
    # simple feedback comb filter
    x = np.asarray(x)
    n = len(x)
    y = x.copy()
    for i in range(period, n):
        y[i] = x[i] + alpha*y[i-period]
    y = y/rms(y)*rms(x)
    return y

#    # Combify a FIR LPF (based on DE's pitchfilter)     building up time is too long (~len(f_b)) which is a problem for short frame processing
#    fir_ord = 6;
#    cutoff = 0.1;
#    b = remez(fir_ord+1, [0,cutoff-0.05, cutoff+0.05,0.5], [1.0, 0.0]);    
#    f_b = np.zeros(1 + len(b)*period);
#    f_b[0:-1:period] = b
#    y = convolve(x, f_b);  
#    y = y[fir_ord/2:fir_ord/2+len(x)] 
##    f_a = 1;    # for the output...
#    return y

def tf_reassignment(y, sampling_frequency, framesz, hop, sigma):
    
#    nsamples = len(y)    
#    T = 1.0*nsamples/sampling_frequency
    
    gabor_orig, gabor_orig_dw, gabor_orig_dt, freq = stft(y, sampling_frequency, framesz, hop, sigma=sigma)
    
    f_bins_shift = np.round(np.imag(gabor_orig_dw)/freq[1]).astype(int)
    t_bins_shift = np.round(np.real(gabor_orig_dt)/hop).astype(int)
    
    t_bins, f_bins = gabor_orig.shape
    
    gabor_reassigned = np.zeros_like(gabor_orig)
    
    for i_tbin in range(t_bins):
        for i_fbin in range(f_bins):
            new_i_fbin = i_fbin - f_bins_shift[i_tbin, i_fbin]
            new_i_tbin = i_tbin + t_bins_shift[i_tbin, i_fbin]
            if (
                new_i_fbin >= 0 and 
                new_i_fbin < f_bins and
                new_i_tbin >= 0 and 
                new_i_tbin < t_bins
               ):
                
                gabor_reassigned[new_i_tbin, new_i_fbin] = gabor_reassigned[new_i_tbin, new_i_fbin] + gabor_orig[i_tbin,i_fbin]
            else:
                gabor_reassigned[i_tbin, i_fbin] = gabor_reassigned[i_tbin, i_fbin] + gabor_orig[i_tbin, i_fbin]
                
    return gabor_orig, gabor_reassigned
    
def ridge_detection(y, sampling_frequency, framesz, hop,sigma,prct_thr=98, cf_cutoff=5000, prune=True, plot=True):

    BWallangles = [[0 for x in range(8)] for x1 in range(len(sigma))]
    nsamples = len(y)    
    T = 1.0*nsamples/sampling_frequency
    
    gabor_orig_allsigma = []    
    for sigma_i in range(len(sigma)):    
        gabor_orig, gabor_orig_dw, gabor_orig_dt, freq = stft(y, sampling_frequency, framesz, hop, sigma=sigma[sigma_i])
        gabor_orig_allsigma.append(gabor_orig)
        
        if plot:
            pl.figure()
            pl.imshow(db(np.square(scipy.absolute(gabor_orig))).T,origin='lower',aspect='auto', 
                      extent=[0, T, 0, sampling_frequency], interpolation='nearest')
            pl.ylim((0,sampling_frequency/2))
            pl.xlabel('Time')
            pl.ylabel('Frequency')
            vmin, vmax = pl.gci().get_clim()
            
            pl.figure()
            pl.imshow(db(np.square(scipy.absolute(gabor_orig_dw))).T,origin='lower',aspect='auto', 
                      extent=[0, T, 0, sampling_frequency], interpolation='nearest')
            pl.ylim((0,sampling_frequency/2))
            pl.xlabel('Time')
            pl.ylabel('Frequency')
            
        
        
        lowpass_gabor_orig_dw = gabor_orig_dw[:,freq<=cf_cutoff]                
        
        # pruning using mixed partial derivative dtdw
        if prune:
            lowfreq = freq[freq<=cf_cutoff]
            freq_orig = np.repeat(lowfreq[:,None], lowpass_gabor_orig_dw.shape[0], axis=1).T
            freq_shift = np.imag(lowpass_gabor_orig_dw)
            freq_new = freq_orig - freq_shift
            [gx,gy] = np.gradient(freq_new)
            
            prune_mask = np.abs(gy/freq[1]) < 0.6
        else:
            prune_mask = np.ones(lowpass_gabor_orig_dw.shape).astype(bool)
        
        n_angle = 1
        
        if plot:
            fig = pl.figure()    
        for angle_i in range(n_angle):
            theta = np.pi/8*angle_i
            s = (-1)*(np.imag(lowpass_gabor_orig_dw*np.exp(1j*theta))<0)+(np.imag(lowpass_gabor_orig_dw*np.exp(1j*theta))>0)
            [gx,gy]=np.gradient(s)
            BW=((-gx*np.cos(theta+np.pi/2)+gy*np.sin(theta+np.pi/2))>.1)
            
            BW = BW & prune_mask
            
            CC = measure.regionprops(measure.label(BW))
            weightv = np.zeros((len(CC),))
            powerv = np.zeros((len(CC),))
            for i in range(len(CC)):
                weightv[i] = CC[i].area
                inds = CC[i].coords
                powerv[i] = np.mean(np.abs(gabor_orig[inds[:,0], inds[:,1]]))
            
            a = np.logical_and(weightv>=np.percentile(weightv, prct_thr),powerv>=np.percentile(powerv, 10))
            tempv = np.zeros_like(BW)
            
            for index in np.nonzero(a)[0]:
                ind = CC[index].coords
                tempv[ind[:,0], ind[:,1]] = 1
            BWallangles[sigma_i][angle_i]=tempv.astype(int)
            
            if plot:
                ax = fig.add_subplot(2,4,angle_i+1)
                ax.imshow(tempv.T,origin='lower',aspect='auto', extent=[0, T, 0, cf_cutoff], cmap='binary',interpolation='nearest')
                ax.set_xlabel('Time')
                ax.set_ylabel('Frequency')
    
    consensus_allsigma = []
    for sigma_i in range(len(sigma)):
        consensus = np.zeros(BW.shape)
        if sigma_i == len(sigma)-1:
            neighboor_sigma = sigma_i-1
        else:
            neighboor_sigma = sigma_i+1                    
        for angle_i in range(n_angle):
#            if angle_i==0:
#                cv= BWallangles[sigma_i][0]+BWallangles[sigma_i][1]+BWallangles[sigma_i][7] \
#                    + BWallangles[neighboor_sigma][0]
#                consensus=consensus+(cv>1).astype(int)
#            elif angle_i==7:
#                cv= BWallangles[sigma_i][0]+BWallangles[sigma_i][6]+BWallangles[sigma_i][7] \
#                    + BWallangles[neighboor_sigma][7]
#                consensus=consensus+(cv>1).astype(int)
#            else:
#            cv= BWallangles[sigma_i][angle_i] + BWallangles[sigma_i][angle_i-1]+BWallangles[sigma_i][angle_i+1] \
#                + BWallangles[neighboor_sigma][angle_i]
            cv= BWallangles[sigma_i][angle_i] 
            consensus=consensus+(cv>0).astype(int)
        consensus_allsigma.append(consensus>0)
        
    if plot:    
        pl.figure()
        for i in range(len(sigma)):
            pl.subplot(2,2,i+1)
            pl.imshow(consensus_allsigma[i].T,origin='lower',aspect='auto', extent=[0, T, 0, cf_cutoff], cmap='binary',interpolation='nearest')
        pl.xlabel('Time')
        pl.ylabel('Frequency')
        pl.clim((0,1))
    
    return gabor_orig_allsigma, consensus_allsigma
    
def consonant_detection(Y_stft, freq, tspan_fft, sampling_frequency, cf_cutoff=5000, plot=True):
    
    highpass_gabor_orig = Y_stft[:,np.logical_and(freq>cf_cutoff,freq<sampling_frequency/2)]
    lowpass_gabor_orig = Y_stft[:,freq<=cf_cutoff]

    lowpass_gabor_orig_energy = np.abs(lowpass_gabor_orig).sum(axis=1)
    highpass_gabor_orig_energy = np.abs(highpass_gabor_orig).sum(axis=1)
    highpass_gabor_orig_energy = np.roll(np.asarray(running_mean(highpass_gabor_orig_energy,100)),-50)
    
    if plot:
        fig1=pl.figure()
        ax1 = fig1.add_subplot(111)
        line1 = ax1.imshow(db(np.abs(Y_stft))[:,np.logical_and(freq>=cf_cutoff, freq<sampling_frequency/2)].T, origin='lower', aspect='auto', extent=[0, T, cf_cutoff, sampling_frequency/2], interpolation='nearest')
        pl.xlabel('Time')
        pl.ylabel('Frequency')
        ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
        line2 = ax2.plot(tspan_fft, highpass_gabor_orig_energy,'k')
        #line3 = ax2.plot(tspan_fft, np.abs(lowpass_gabor_orig).sum(axis=1),'k-.')
        #line4 = ax2.plot(tspan_fft[1:], np.diff(highpass_gabor_orig_energy),'g')
        line4 = ax2.axhline(np.mean(highpass_gabor_orig_energy),color='k', linestyle='--')
        line5 = ax2.plot(tspan_fft,np.abs(highpass_gabor_orig_energy-highpass_gabor_orig_energy.mean()),color='g')
        _max, _min = peakdetect(np.abs(highpass_gabor_orig_energy-highpass_gabor_orig_energy.mean()),lookahead=2)
        #_max, _min = peakdetect(np.diff(100*np.diff(highpass_gabor_orig_energy)))
        #xm = [p[0] for p in _max]
        #ym = [p[1] for p in _max]
        #xm = np.asarray(xm)
        #ym = np.asarray(ym)
        ##xm = xm[ym>ym.mean()]
        ##ym = ym[ym>ym.mean()]
        #for x in xm:
        #    line5 = ax2.axvline(tspan_fft[x+1], color='k', linestyle='--')
        #line6 = ax2.plot(tspan_fft[xm+1], ym, 'go', markersize=4)
        xm = [p[0] for p in _min]
        ym = [p[1] for p in _min]
        xm = np.asarray(xm)
        ym = np.asarray(ym)
    #            xm = xm[ym<0.03]
    #            ym = ym[ym<0.03]
    
        if plot:
            line7 = ax2.plot(tspan_fft[xm+1], ym, 'ro', markersize=4)
        
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")
            pl.ylabel("high frequency energy")
    
    energy_ratio_candidate_consonant_windows = np.asarray([highpass_gabor_orig_energy[xm[i]:xm[i+1]].mean()/highpass_gabor_orig_energy.mean() for i in range(len(xm)-1)])
    consonant_windows = np.asarray([[xm[i],xm[i+1]] for i in range(len(xm)-1) if energy_ratio_candidate_consonant_windows[i]>1.3 ])
    half_consonant_mask = np.zeros((Y_stft.shape[0], int(Y_stft.shape[1]/2)))
    for x in consonant_windows:
        if lowpass_gabor_orig_energy[x[0]:x[1]].mean()/lowpass_gabor_orig_energy.mean() < 1.2:
            half_consonant_mask[x[0]:x[1],:] = 1 

            if plot:            
                line8 = ax2.axvline(tspan_fft[x[0]+1], color='k', linestyle='--')
                line9 = ax2.axvline(tspan_fft[x[1]+1], color='k', linestyle='--')
               
    return half_consonant_mask               
    
def pitch_ACF(y, sampling_frequency, pitchrange, winsize, shiftsize, clip_thr, energy_thr):
    
    nframes = len(y)
    lag_range = np.sort(np.round(sampling_frequency/pitchrange))
    
    wlen = int(winsize*sampling_frequency)
    wshift_len = int(shiftsize*sampling_frequency)
    n_win = 1 + int((nframes-wlen)/wshift_len)
    
    pitch_est = np.zeros((n_win,3))
    period2enh = np.zeros(n_win)

    for iw in range(0,n_win):
        y_win = y[iw*wshift_len:iw*wshift_len+wlen]  #*scipy.signal.slepian(wlen,width=0.001)
#        print 'Frame #', iw, 'out of ', n_win
        
        max1 = np.max(np.abs(y_win[0:int(wlen/3)]))
        max2 = np.max(np.abs(y_win[int(wlen*2/3):]))
        thr = np.min([max1,max2])*clip_thr
        y_win_cl = clip(y_win, thr)
        
        corr_y = scipy.signal.correlate(y_win_cl,y_win_cl,mode='full')
        lags = np.asarray(range(-wlen+1,wlen,1))
        bias_factor = 1.0/(wlen-np.abs(lags))
        corr_y_unbias = corr_y*bias_factor
        
        lag_pos = lags[lags>0]
        corr_pos = corr_y_unbias[lags>0]
        corr_pos[np.logical_or(lag_pos<lag_range[0], lag_pos>lag_range[1])] = 0
        
        localpeak_inds = argrelextrema(corr_pos, np.greater)
        
        lag_peak1 = 0
        if len(localpeak_inds[0]) > 0:            
            pitchpeak_inds = localpeak_inds[0][corr_pos[localpeak_inds[0]]>corr_y_unbias[lags==0]*energy_thr]
            if len(pitchpeak_inds) > 0:                
                sort_ind = sorted(range(len(pitchpeak_inds)), key=lambda k: corr_pos[pitchpeak_inds[k]], reverse=True)  # sort is in ascending order
                if len(pitchpeak_inds) == 1:
                    lag_peak1 = lag_pos[pitchpeak_inds[sort_ind[0]]]
                    pitch_est[iw,0] = 1.0*sampling_frequency/lag_peak1
                    period2enh[iw] = lag_peak1
                elif len(pitchpeak_inds) == 2:
                    lag_peak1 = lag_pos[pitchpeak_inds[sort_ind[0]]]
                    lag_peak2 = lag_pos[pitchpeak_inds[sort_ind[1]]]
                    pitch_est[iw,0] = 1.0*sampling_frequency/lag_peak1
                    pitch_est[iw,1] = 1.0*sampling_frequency/lag_peak2
                    lag_allpeak = np.sort([lag_peak1, lag_peak2])
                    ratio=lag_allpeak[1]/float(lag_allpeak[0])
                    residue = ratio-np.round(ratio)
                    if residue < 0.05:
                        period2enh[iw] = lag_allpeak[0]
                elif len(pitchpeak_inds) >= 3:
                    lag_peak1 = lag_pos[pitchpeak_inds[sort_ind[0]]]
                    lag_peak2 = lag_pos[pitchpeak_inds[sort_ind[1]]]
                    lag_peak3 = lag_pos[pitchpeak_inds[sort_ind[2]]]
                    pitch_est[iw,0] = 1.0*sampling_frequency/lag_peak1
                    pitch_est[iw,1] = 1.0*sampling_frequency/lag_peak2
                    pitch_est[iw,2] = 1.0*sampling_frequency/lag_peak3
                    lag_allpeak = np.sort([lag_peak1, lag_peak2])
                    ratio=lag_allpeak[1]/float(lag_allpeak[0])
                    residue = ratio-np.round(ratio)
                    if residue < 0.05:
                        period2enh[iw] = lag_allpeak[0]
    period2enh[period2enh==lag_range[0]]=0
    period2enh[period2enh==lag_range[1]]=0
    period2enh = scipy.signal.medfilt(period2enh, 3)
    
    return pitch_est, period2enh
    
def pitch_enhance(y, period2enh, alpha, sampling_frequency, winsize, shiftsize):

    nframes = len(y)    
    
    wlen = int(winsize*sampling_frequency)
    wshift_len = int(shiftsize*sampling_frequency)
    n_win = 1 + int((nframes-wlen)/wshift_len)
    
    
    if len(period2enh)!=n_win:
        print 'Signal length does not match pitch estimations!'
        import sys
        sys.exit()
        
    y_enh = np.copy(y)
    
    for iw in range(0,n_win):
        y_win = y[iw*wshift_len:iw*wshift_len+wlen]  #*scipy.signal.slepian(wlen,width=0.001)
#        print 'Frame #', iw, 'out of ', n_win    
        
        # pitch enhancement by comb filter        
        if period2enh[iw] == 0:
            y_enh_win = y_win
        else:
            y_enh_win = comb_filter(y_win, int(period2enh[iw]), alpha)
        
        if iw == 0:
            y_enh[:wlen] = y_enh_win
        else:
            y_enh[wlen+(iw-2)*wshift_len:wlen+(iw-1)*wshift_len] = y_enh[wlen+(iw-2)*wshift_len:wlen+(iw-1)*wshift_len]*np.linspace(1,0,wshift_len) + \
                                                                    y_enh_win[wlen-2*wshift_len:wlen-wshift_len]*np.linspace(0,1,wshift_len)
            y_enh[wlen+(iw-1)*wshift_len:wlen+iw*wshift_len] = y_enh_win[wlen-wshift_len:wlen]
            
#        #### test the concatenation by reconstruct the original signal
#        if iw == 0:
#            y_recon[:wlen] = y_win            
#        else:
#            y_recon[wlen+(iw-1)*wshift_len:wlen+iw*wshift_len] = y_win[wlen-wshift_len:wlen]            
        
#        lag_max = lag_pos[np.argmax(corr_pos)]
#        offset = 0
#        while lag_max == lag_range[0]+offset:
#            offset = offset+10
#            print '    offset = ', offset
#            corr_pos[np.logical_and(lag_pos>=lag_max, lag_pos<lag_range[0]+offset)] = 0    
#            lag_max = lag_pos[np.argmax(corr_pos)] 
        
    return y_enh
    
    
def multband_pitch_enhance(y, alpha, sampling_frequency, winsize, shiftsize, pitch_range, clip_thr, energy_thr, isplot=False):
    
    #### pitch extraction ####
    
    y_lp = mne.filter.low_pass_filter(y, sampling_frequency, 1200)
    y_bp_raw = mne.filter.band_pass_filter(y, sampling_frequency, 1200, 3000)
    y_bp_hilt = np.abs(scipy.signal.hilbert(y_bp_raw))
    y_bp = mne.filter.low_pass_filter(y_bp_hilt, sampling_frequency, 800)
    y_bp2_raw = mne.filter.band_pass_filter(y, sampling_frequency, 3000, 10000)
    y_bp2_hilt = np.abs(scipy.signal.hilbert(y_bp2_raw))
    y_bp2 = mne.filter.low_pass_filter(y_bp2_hilt, sampling_frequency, 800)
    
    pitch_y_lp, mainperiod_y_lp = pitch_ACF(y_lp, sampling_frequency, pitch_range, winsize, shiftsize, clip_thr, energy_thr)
    mainpitch_y_lp = sampling_frequency/mainperiod_y_lp
    mainpitch_y_lp[mainpitch_y_lp==np.inf] = 0    
    
    pitch_y_bp, mainperiod_y_bp = pitch_ACF(y_bp, sampling_frequency, pitch_range, winsize, shiftsize, clip_thr, energy_thr)
    mainpitch_y_bp = sampling_frequency/mainperiod_y_bp
    mainpitch_y_bp[mainpitch_y_bp==np.inf] = 0
    
    pitch_y_bp2, mainperiod_y_bp2 = pitch_ACF(y_bp2, sampling_frequency, pitch_range, winsize, shiftsize, clip_thr, energy_thr)
    mainpitch_y_bp2 = sampling_frequency/mainperiod_y_bp2
    mainpitch_y_bp2[mainpitch_y_bp2==np.inf] = 0
    
    ### pitch enhancement ###
    y_enh = pitch_enhance(y_lp, mainperiod_y_lp, alpha, sampling_frequency, winsize, shiftsize)
    y_lp_enh = mne.filter.low_pass_filter(y_enh, sampling_frequency, 1200)
    y_enh = pitch_enhance(y_bp_raw, mainperiod_y_bp, alpha, sampling_frequency, winsize, shiftsize)
    y_bp_enh = mne.filter.band_pass_filter(y_enh, sampling_frequency, 1200, 3000)
    y_enh = pitch_enhance(y_bp2_raw, mainperiod_y_bp2, alpha, sampling_frequency, winsize, shiftsize)
    y_bp2_enh = mne.filter.band_pass_filter(y_enh, sampling_frequency, 3000, 10000)
    
    y_enh = y_lp_enh + y_bp_enh + y_bp2_enh
    
#    if isplot == 1:
#        
#        import pylab as pl
#        
#        n_win = len(pitch_y_lp)
#        fig1=pl.figure()
#        ax1 = fig1.add_subplot(111)
#        line1 = ax1.imshow(pxx_t[:,0:27].T, origin='lower', aspect='auto', extent=[0, T, 0, 450],
#                     interpolation='nearest')
#        pl.xlabel('Time')
#        pl.ylabel('Frequency')
#        pl.title('Clean Target (<1200Hz)')
#        pl.show()    
#        ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
#        line2 = ax2.plot((np.arange(0,n_win)*shiftsize+winsize/2.0), pitch_y_lp,'o', markersize=6)
#        line3 = ax2.plot((np.arange(0,n_win)*shiftsize+winsize/2.0), mainpitch_y_lp,'ko', markersize=4)
#        ax2.yaxis.tick_right()
#        ax2.yaxis.set_label_position("right")
#        ax2.set_ylim([0,450])
#        pl.ylabel("Pitch")
#        
#        fig1=pl.figure()
#        ax1 = fig1.add_subplot(111)
#        line1 = ax1.imshow(pxx_t[:,0:27].T, origin='lower', aspect='auto', extent=[0, T, 0, 450],
#                     interpolation='nearest')
#        pl.xlabel('Time')
#        pl.ylabel('Frequency')
#        pl.title('Clean Target (1200~3000Hz)')
#        pl.show()    
#        ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
#        line2 = ax2.plot((np.arange(0,n_win)*shiftsize+winsize/2.0), pitch_y_bp,'o', markersize=6)
#        line3 = ax2.plot((np.arange(0,n_win)*shiftsize+winsize/2.0), mainpitch_y_bp,'ko', markersize=4)
#        ax2.yaxis.tick_right()
#        ax2.yaxis.set_label_position("right")
#        ax2.set_ylim([0,450])
#        pl.ylabel("Pitch")
#    
#        fig1=pl.figure()
#        ax1 = fig1.add_subplot(111)
#        line1 = ax1.imshow(pxx_t[:,0:27].T, origin='lower', aspect='auto', extent=[0, T, 0, 450],
#                     interpolation='nearest')
#        pl.xlabel('Time')
#        pl.ylabel('Frequency')
#        pl.title('Clean Target (3000~5000Hz)')
#        pl.show()    
#        ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
#        line2 = ax2.plot((np.arange(0,n_win)*shiftsize+winsize/2.0), pitch_y_bp2,'o', markersize=6)
#        line3 = ax2.plot((np.arange(0,n_win)*shiftsize+winsize/2.0), mainpitch_y_bp2,'ko', markersize=4)
#        ax2.yaxis.tick_right()
#        ax2.yaxis.set_label_position("right")
#        ax2.set_ylim([0,450])
#        pl.ylabel("Pitch")
        
    return y_enh

def crest_factor(maskimg):
    nx, ny = maskimg.shape    
    
    maskimg1d = maskimg.flatten()
    crest_factor = maskimg1d.max()/rms(maskimg1d)
    
    xmat = np.ones((ny,1))*range(nx)
    ymat = np.reshape(np.asarray(range(ny)), (ny,1))*np.ones((1,nx)) 
        
    centroid_px = int(np.multiply(xmat,maskimg).sum() / maskimg.sum())
    centroid_py = int(np.multiply(ymat,maskimg).sum() / maskimg.sum())
    
    return crest_factor, centroid_px, centroid_py
    
def crest_factor_image(image, maskersize):
    nx, ny = image.shape
    
    cf_image = np.ones((nx,ny))
    for i in range(nx-maskersize):
        for j in range(ny-maskersize):
            maskimg = image[i:i+maskersize, j:j+maskersize]
            cf_image_mask = cf_image[i:i+maskersize, j:j+maskersize]
            cf, px, py = crest_factor(maskimg)
            if np.logical_and(px<maskersize, py<maskersize):            
                cf_image_mask[px,py] = cf
            else:
                print i,j
                
                
            
    return cf_image
    