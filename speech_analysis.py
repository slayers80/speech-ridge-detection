# -*- coding: utf-8 -*-
"""
Created on Wed Jul 08 13:48:53 2015

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

#subprocess.call(['ffmpeg', '-i', origaudioroot+'SA1.mp3', origaudioroot+'SA1.wav'])

start_time = time.time()

# Main function ##
#origaudioroot = 'C:/Users/lwang/My Work/EEG_Experiments/Experiment_Scripts/Ex012_STHLAlgrth/quicksin_ANL/original_stimuli/separated_lists/'
resultfolder = 'I:/My Drive/Research/Algorithm/'
#T, nbits, data, nframes, nchannels, sampling_frequency = read_wavfile(origaudioroot+'list_01.wav')

#origaudioroot = 'C:/Users/lwang/Google Drive/Speech Corpus/TIMIT-resaved/TEST/'
origaudioroot = 'I:/My Drive/Research/Algorithm/'

count = 0
for path, subdirs, files in os.walk(resultfolder):
    for name in files:
#        if name[-3:]=="WAV":
        if name=="binary_105_clean.wav":
            filename = os.path.join(path, name)
            print filename
            count = count+1
        
            pl.close('all')

            T, nbits, data, nframes, nchannels, sampling_frequency = read_wavfile(filename)
            
            
            a = np.asarray(data)
            
            y = np.zeros((nframes, nchannels))
            
            normfactor = 2**(nbits*8-1)
            
            y[:,0] = 1.0*a[0::nchannels]/normfactor
            
            T = 1.0*nframes/sampling_frequency            
            
            s = y[:,0]
            
            
            
            ##### Speech in pink noise demo  ####
#            noise = np.random.normal(loc=0, scale=1, size=(nframes,))
#            Noise = np.fft.rfft(noise)
#            f = np.fft.fftfreq(nframes, d=1./sampling_frequency)[:len(Noise)]
#            f[-1]=np.abs(f[-1])
#            alpha = 0.09
#            gain = np.ones_like(f)
#            gain = np.abs((f**(-alpha)))
#            gain[gain>1] = 1
#            Noise_pink = Noise * gain
#            noise_pink = np.fft.irfft(Noise_pink)    
#            noise_pink = noise_pink/rms(noise_pink)*rms(y)
            
            
            
#            #### Speech-shaped noise
#            Y = np.fft.fft(y_all)
#            ssNoise = abs(Y)*np.exp(np.random.uniform(low=0,high=1,size=y_all.shape[0])*1j*np.pi*2)
#            ssnoise = np.real((np.fft.ifft(ssNoise)))
#            ssnoise = ssnoise/rms(ssnoise)*rms(y[:,0])
            
            T, nbits, data, nframes, nchannels, sampling_frequency = read_wavfile(resultfolder+"binary_105_noise.wav")
            a = np.asarray(data)
            
            y = np.zeros((nframes, nchannels))
            
            normfactor = 2**(nbits*8-1)
            
            n = 1.0*a[0::nchannels]/normfactor
            
            snr = 4.0
            alpha = np.sqrt(np.sum(s**2)/(np.sum(n**2)*(10**(snr/10))))
            
            n = n*alpha
            
            snr1=10*np.log10(np.sum(s**2)/np.sum((n)**2))
            
            y_orig = s + n
            
            #### STFT analysis  ####
#            framesz = 0.128  # with a frame size of 60 milliseconds
#            hop = 0.003 # 0.0003125      # and hop size of 15 milliseconds.
            framesz = 0.02  # with a frame size of 60 milliseconds
            hop = 0.001 # 0.0003125      # and hop size of 15 milliseconds.
            t = scipy.linspace(0, T, T*sampling_frequency, endpoint=False)
            cf_cutoff = 5000
            
            S, S_reassigned = tf_reassignment(s, sampling_frequency, framesz, hop, sigma=5)
            N, N_reassigned = tf_reassignment(n[:len(s)], sampling_frequency, framesz, hop, sigma=5)
            SN, SN_reassigned = tf_reassignment(s+n[:len(s)], sampling_frequency, framesz, hop, sigma=5)
            
            
            pl.figure()
            pl.subplot(121)
            pl.imshow(db(np.abs(S)).T,origin='lower',aspect='auto', extent=[0, T, 0, sampling_frequency], cmap='jet',interpolation='nearest')
            pl.xlabel('Time')
            pl.ylabel('Frequency')            
            pl.ylim((0,cf_cutoff))
            pl.subplot(122)            
            pl.imshow(db(np.abs(S_reassigned)).T,origin='lower',aspect='auto', extent=[0, T, 0, sampling_frequency], cmap='jet',interpolation='nearest')
            pl.xlabel('Time')
            pl.ylabel('Frequency')            
            pl.ylim((0,cf_cutoff))
            
            pl.figure()
            pl.subplot(121)
            pl.imshow(db(np.abs(N)).T,origin='lower',aspect='auto', extent=[0, T, 0, sampling_frequency], cmap='jet',interpolation='nearest')
            pl.xlabel('Time')
            pl.ylabel('Frequency')            
            pl.ylim((0,cf_cutoff))
            pl.subplot(122)            
            pl.imshow(db(np.abs(N_reassigned)).T,origin='lower',aspect='auto', extent=[0, T, 0, sampling_frequency], cmap='jet',interpolation='nearest')
            pl.xlabel('Time')
            pl.ylabel('Frequency')            
            pl.ylim((0,cf_cutoff))
            
            pl.figure()
            pl.subplot(121)
            pl.imshow(db(np.abs(SN)).T,origin='lower',aspect='auto', extent=[0, T, 0, sampling_frequency], cmap='jet',interpolation='nearest')
            pl.xlabel('Time')
            pl.ylabel('Frequency')            
            pl.ylim((0,cf_cutoff))
            pl.subplot(122)            
            pl.imshow(db(np.abs(SN_reassigned)).T,origin='lower',aspect='auto', extent=[0, T, 0, sampling_frequency], cmap='jet',interpolation='nearest')
            pl.xlabel('Time')
            pl.ylabel('Frequency')            
            pl.ylim((0,cf_cutoff))
            
            ibm_mask = (np.abs(S)>np.abs(N)).astype(int)
            
            pl.figure()
            pl.subplot(121)
            pl.imshow(db(np.abs(S+N)).T,origin='lower',aspect='auto', extent=[0, T, 0, sampling_frequency], cmap='jet',interpolation='nearest')
            pl.xlabel('Time')
            pl.ylabel('Frequency')            
            pl.ylim((0,cf_cutoff))
            pl.subplot(122)            
            pl.imshow((ibm_mask).T,origin='lower',aspect='auto', extent=[0, T, 0, sampling_frequency], cmap='binary',interpolation='nearest')
            pl.xlabel('Time')
            pl.ylabel('Frequency')            
            pl.ylim((0,cf_cutoff))
            
            # vowel extraction
            sigma = [4,5,6]
            
            gabor_orig_allsigma, consensus_allsigma = ridge_detection(s+n, sampling_frequency, framesz, hop,sigma,prct_thr=95, cf_cutoff=cf_cutoff,prune=False,plot=False)
            freq = np.linspace(0,sampling_frequency,num=int(sampling_frequency*framesz),endpoint=False)
            tspan_fft = np.linspace(0,T,gabor_orig_allsigma[0].shape[0])
            
            
            final_consensus = np.asarray(consensus_allsigma).sum(axis=0)
            pl.figure()
            pl.subplot(121)
            pl.imshow(db(np.abs(gabor_orig_allsigma[1])).T,origin='lower',aspect='auto', extent=[0, T, 0, sampling_frequency], cmap='jet',interpolation='nearest')
            pl.xlabel('Time')
            pl.ylabel('Frequency')            
            pl.ylim((0,cf_cutoff))
            pl.subplot(122)            
            pl.imshow((final_consensus>1).T,origin='lower',aspect='auto', extent=[0, T, 0, cf_cutoff], cmap='binary',interpolation='nearest')
            pl.xlabel('Time')
            pl.ylabel('Frequency')
            
            pl.figure()
            pl.subplot(121)
            pl.imshow(db(np.abs(gabor_orig_allsigma[1])).T,origin='lower',aspect='auto', extent=[0, T, 0, sampling_frequency], cmap='jet',interpolation='nearest')
            pl.xlabel('Time')
            pl.ylabel('Frequency')            
            pl.ylim((0,cf_cutoff))
            pl.subplot(122)            
            pl.imshow((ibm_mask[:,freq<=cf_cutoff] & (final_consensus>1)).T,origin='lower',aspect='auto', extent=[0, T, 0, cf_cutoff], cmap='binary',interpolation='nearest')
            pl.xlabel('Time')
            pl.ylabel('Frequency')
            
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
                            segments_slope[x+i_tbin,y] = slope_segments[i]/f_segments[i]*10000
                            segments_harm_dist[x+i_tbin,y] = resid[i, pitch_range==pitch_est[i_tbin/framesize]]
                            segments_parallelism[x+i_tbin,y] = (slope_segments[i]-slope_segments.mean())/(slope_segments.std()+np.finfo(np.float32).eps)
                            
                        harmf_est = pitch_est[i_tbin/framesize]*np.arange(1,np.floor(cf_cutoff/pitch_est[i_tbin/framesize]))                    
                        harmonics_to_keep[i_tbin:i_tbin+framesize, np.round(harmf_est/freq[1]).astype(int)] += 1
                        
                        segment_ind_to_keep = resid[:, pitch_range==pitch_est[i_tbin/framesize]]<harm_freq_margin
                        inds2, _ = np.nonzero(segment_ind_to_keep)
                        if len(inds2) > 0:
                            for ind2 in inds2:
                                segments_to_keep[inds[ind2][:,0]+i_tbin,inds[ind2][:,1]] += 1
                                
            segments_str_max = np.max(segments_str,axis=2)
            
            segments_combined_feature = np.zeros(ridge_mask.shape)*np.nan
            CC = measure.regionprops(measure.label(ridge_mask))
            for i, segment in enumerate(CC):
                x = segment.coords[:,0]
                y = segment.coords[:,1]
                segment_x1 = segments_harm_dist[x,y].mean()
                segment_x2 = segments_slope[x,y].std()
                segment_x3 = np.abs(segments_parallelism[x,y]).mean()
                segment_x4 = segments_harm_num[x,y].std()
                segments_combined_feature[x,y] = (segment_x4<1)&(segment_x1<0.1)&(segment_x2<1.5)&(segment_x3<0.9)
#                segments_combined_feature[x,y] = (segment_x4<2.7)&(segment_x1<0.1)&(segment_x2<0.5)&(segment_x3<0.9)
                
            segments_to_keep_pruned = np.zeros(ridge_mask.shape)   
            segments_to_keep_pruned[segments_combined_feature==1] = 1
            
#            CC = measure.regionprops(measure.label(segments_to_keep))
#            regions_area = np.array([x.area for x in CC])
#            
#            segments_to_keep_pruned = np.zeros(segments_to_keep.shape)
#            region_area_thresh = np.percentile(regions_area, 85)
#            for ii in range(len(CC)):
#                if CC[ii].area >= region_area_thresh:
#                    segments_to_keep_pruned[CC[ii].coords[:,0],CC[ii].coords[:,1]] = True
            
            pl.figure()
            pl.subplot(311)
            for key in pitch_cand.keys():
                pl.scatter(tspan_fft[key]*np.ones(pitch_cand[key].shape),pitch_cand[key])
                pl.xlim((-0.03,0.52))
            pl.subplot(312)
            pl.scatter(np.arange(0,n_frames*framesize,framesize)/1000., pitch_est)
            pl.subplot(313)
            pl.scatter(np.arange(0,n_frames*framesize,framesize)/1000., pitch_str,color='r')
            
            pl.figure()
            pl.subplot(121)
            pl.imshow((segments_str_max).T,origin='lower',aspect='auto', extent=[0, T, 0, cf_cutoff], cmap='jet',interpolation='nearest')
            pl.xlabel('Time')
            pl.ylabel('Frequency')
            pl.ylim((0,cf_cutoff))
            pl.subplot(122)            
            pl.imshow((segments_str_max>15).T,origin='lower',aspect='auto', extent=[0, T, 0, cf_cutoff], cmap='binary',interpolation='nearest')
            pl.xlabel('Time')
            pl.ylabel('Frequency')
            
            pl.figure()
            pl.subplot(121)
            pl.imshow((segments_to_keep).T,origin='lower',aspect='auto', extent=[0, T, 0, cf_cutoff], cmap='binary',interpolation='nearest')
            pl.xlabel('Time')
            pl.ylabel('Frequency')
            pl.ylim((0,cf_cutoff))            
#            pl.colorbar()
            pl.subplot(122)
            pl.imshow(segments_to_keep_pruned.T,origin='lower',aspect='auto', extent=[0, T, 0, cf_cutoff], cmap='binary',interpolation='nearest')
            pl.xlabel('Time')
            pl.ylabel('Frequency')
            
            
            pitch_ridge_mask = ridge_mask + (harmonics_to_keep>0)*2
            pitch_ridge_mask[segments_to_keep>0] = 3
            colors = [(1,1,1),(1, 0.5, 0.5), (0.5, 0.5, 1), (0, 0, 0)]  # White -> light Red -> light Blue -> Black
            n_bin = 4  # Discretizes the interpolation into bins
            cmap_name = 'my_list'
            fig = pl.figure()
            ax = fig.add_subplot(111)
            cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
            pl.imshow(pitch_ridge_mask.T, origin='lower', aspect='auto',interpolation='nearest', extent=[0, T, 0, cf_cutoff], cmap=cm)
#            ax.set_title("N bins: %s" % n_bin)
#            fig.colorbar(im, ax=ax)
            
            
            half_vowel_mask = np.concatenate( ((segments_to_keep_pruned), np.zeros( (len(tspan_fft), int(len(freq)/2)-final_consensus.shape[1])) ), axis=1)
#            gabor_orig_dB_vowel_mask = db(np.square(scipy.absolute(gabor_orig_allsigma[-2][:,:half_vowel_mask.shape[1]])))
#            vmin, vmax = gabor_orig_dB_vowel_mask.min(), gabor_orig_dB_vowel_mask.max()            
#            gabor_orig_dB_vowel_mask[half_vowel_mask.astype(bool)] = vmin-(vmax-vmin)/100
#            jet_colors = pl.cm.jet(np.linspace(0,1,num=100))
#            black_colors = pl.cm.binary(np.ones(1))
#            colors = np.vstack((black_colors, jet_colors))
#            mycmap = pl.mpl.colors.LinearSegmentedColormap.from_list('mycolormap', colors)
#            pl.figure()
#            pl.imshow(gabor_orig_dB_vowel_mask.T, origin='lower',aspect='auto', extent=[0, T, 0, sampling_frequency/2], 
#                      cmap=mycmap, interpolation='nearest')
#            pl.xlabel('Time')
#            pl.ylabel('Frequency')
#            pl.ylim((0,cf_cutoff))
##            pl.colorbar()
            
#            # consonant extraction
#            half_consonant_mask = consonant_detection(gabor_orig_allsigma[-1], freq, tspan_fft, sampling_frequency, cf_cutoff=5000, plot=True)
#                
#            half_vowel_mask[half_consonant_mask.astype(bool)] = False            
#            half_speech_mask = half_vowel_mask/(half_vowel_mask.sum()+1) + 8*half_consonant_mask/(half_consonant_mask.sum()+1)
            half_speech_mask = half_vowel_mask
            speech_mask = np.concatenate((half_speech_mask,np.fliplr(half_speech_mask)),axis=1)
#            
#            
#            pl.figure()
#            pl.imshow(speech_mask.T,origin='lower',aspect='auto', 
#                      extent=[0, T, 0, sampling_frequency], cmap='binary', interpolation='nearest')
#            pl.ylim((0, sampling_frequency/2))
#            pl.xlabel('Time')
#            pl.ylabel('Frequency')
#            
            gabor_enh = speech_mask*gabor_orig_allsigma[-2]
#            
            pl.figure()
            pl.imshow(db(np.square(scipy.absolute(gabor_enh))).T,origin='lower',aspect='auto', 
                      extent=[0, T, 0, sampling_frequency], interpolation='nearest')
            pl.ylim((0, sampling_frequency/2))
            pl.xlabel('Time')
            pl.ylabel('Frequency')
            pl.ylim((0,cf_cutoff))
#            pl.clim(vmin=vmin, vmax=vmax)            
#            
            y_enh = istft(gabor_enh, sampling_frequency, T, hop)
            y_enh = y_enh/rms(y_enh)*rms(y_orig)
#           
            pl.figure()
            pl.plot(y_orig)
            pl.plot(y_enh)
            
            gabor_enh_recon, gabor_enh_recon_dw,gabor_enh_recon_dt, freq = stft(y_enh, sampling_frequency, framesz, hop, sigma=sigma[-1])
            pl.figure()
            pl.imshow(db(np.square(scipy.absolute(gabor_enh_recon))).T,origin='lower',aspect='auto', 
                      extent=[0, T, 0, sampling_frequency], interpolation='nearest')
            pl.ylim((0,sampling_frequency/2))
            pl.xlabel('Time')
            pl.ylabel('Frequency')
            pl.clim(vmin,vmax)
#            
            yout = y_orig
            if np.abs(yout).max() > 1:
                yout = yout/np.abs(yout).max()*0.99
            write_wavfile(filename[:-4]+'-ssnoise_5dB.wav', yout[:,None], yout.shape[0], nchannels, sampling_frequency, nbits)
            
            yout = y_enh
            if np.abs(yout).max() > 1:
                yout = yout/np.abs(yout).max()*0.99
            write_wavfile(filename[:-4]+'_enh_allfeaturepruneridge-4dBsnr.wav', yout[:,None], yout.shape[0], nchannels, sampling_frequency, nbits)
            
            

#            end_time = time.time()

#            #### ACF pitch extraction
            pitch_range = np.asarray([50, 500])
            clip_thr = 0.7
            energy_thr = 0.4 
            winsize = 0.06
            shiftsize = 0.01
            
            pitch_est, period2enh = pitch_ACF(s+n, sampling_frequency, pitch_range, winsize, shiftsize, clip_thr, energy_thr)
            
            fig1=pl.figure()
            ax1 = fig1.add_subplot(111)
            line1 = ax1.imshow(db(np.square(scipy.absolute(gabor_orig_allsigma[-2]))).T, origin='lower', aspect='auto', extent=[0, T, 0, sampling_frequency],
                         interpolation='nearest')
            ax1.set_ylim([0,2000])
            pl.xlabel('Time')
            pl.ylabel('Frequency') 
#            fig1.colorbar(line1)
            pl.show()    
#            ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
            line2 = ax1.plot(np.linspace(0,T, num=pitch_est.shape[0]), pitch_est,'o', markersize=6)            
            ax1.set_xlim([0,T])
#            ax2.yaxis.tick_right()
#            ax2.yaxis.set_label_position("right")
#            ax2.set_ylim([0,2000])
#            pl.ylabel("Pitch")

#            
#            figure2pdf(np.array([1,2]),filename[:-4]+"-ssnoise_0dB.pdf")
#            
