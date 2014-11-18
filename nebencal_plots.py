import sys
import os
import math
import re
import subprocess
import random
import yaml
import cPickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
#sns.set(rc={'image.cmap': "RdBu_r"})
colormap="RdYlBu"
sns.set(rc={'image.cmap': colormap})
from operator import itemgetter, attrgetter
from rtree import index
from scipy.optimize import curve_fit
from pyspherematch import spherematch

from nebencal_utils import read_precam
from nebencal_utils import read_sdss
from nebencal_utils import read_betoule
from nebencal_utils import read_tertiaries
from nebencal_utils import global_object
from nebencal_utils import read_global_objs
from nebencal_utils import standard_quality
from nebencal_utils import apply_zps
from nebencal_utils import apply_color_term
from nebencal_utils import apply_stdcal
from nebencal_utils import read_ccdpos

colorterm_dir = '/Users/bauer/software/python/des/des_calib/ting/color-term'
sys.path.append(colorterm_dir)
import color_term_cal


def calc_line(x, a, b):
    return a*x + b

def filter_go(go, config):
    if go.dec < config['general']['dec_min'] or go.dec > config['general']['dec_max']:
        return False
    if config['general']['ra_min'] > config['general']['ra_max']: # 300 60
        if go.ra > config['general']['ra_max'] and go.ra < config['general']['ra_min']:
            return False
    else:
        if go.ra < config['general']['ra_min'] or go.ra > config['general']['ra_max']:
            return False
    # quality cut.....
    go.objects = filter(standard_quality, go.objects)
    if len(go.objects) < 2:
        return False
    return True

def make_plots(config):
    
    band = config['general']['filter']
    
    globals_dir = config['general']['globals_dir']
    current_dir = os.getcwd()

    nside = config['general']['nside_file']
    n_ccds = 63
    match_radius = 1.0/3600.
    magerr_sys2 = 0.0004

    plot_file = 'nebencal_plots_' + band + '.pdf'
    pp = PdfPages(plot_file)

    mag_classes={}
    
    fileband = band

    # read in any catalogs we would like to treat as standards (i.e. precam, sdss...)
    standard_stars = []
    standard_map = index.Index()
    if config['general']['use_precam']:
        read_precam( standard_stars, standard_map, config['general']['precam_filename'], band )
    if config['general']['use_sdss']:
        read_sdss( standard_stars, standard_map, config['general']['sdss_filename'], band )
    if config['general']['use_tertiaries']:
        read_tertiaries( standard_stars, standard_map, config['general']['tertiary_filename'], band )

    # use a validation catalog that we didn't calibrate to...
    validation_stars = []
    validation_map = index.Index()
    if config['general']['sdss_validation'] != "None":
        read_sdss( validation_stars, validation_map, config['general']['sdss_validation'], band )
    if config['general']['betoule_validation'] != "None":
        read_betoule( validation_stars, validation_map, config['general']['betoule_validation'], band )
    if config['general']['tertiary_validation'] != "None":
        read_tertiaries( validation_stars, validation_map, config['general']['tertiary_validation'], band )
    if len(validation_stars) == 0:
        print "No validation catalog?"
        exit(1)
    
    # read in the nominal ccd offsets (zp_phots), which will be our starting point
    zp_list = []
    zp_phots = dict()
    zp_phot_file = open( config['general']['zp_phot_filename'], 'r' )
    zp_phots['id_string'] = 'ccd'
    zp_phots['operand'] = 1
    for line in zp_phot_file:
        entries = line.split()
        if entries[0][0] == '#':
            continue
        zp_phots[int(entries[0])] = -1.0*float(entries[4])
    print "Read in ccd offsets (zp_phots)"
    
    # fill in any missing ones to avoid key errors
    for i in range(n_ccds):
        if i not in zp_phots:
            zp_phots[i] = -999.

    # NOTE: assumes that the id_string is an integer!
    # read in the rest of the zps
    if config['calibrations'] is not None:
        for calibration in config['calibrations']:
            for i, zp_filename in enumerate(calibration['outfilenames']):
                zp_dict = dict()
                zp_dict['operand'] = calibration['operands'][i]
                zp_dict['id_string'] = calibration['id_strings'][i]
                zp_file = open( zp_filename, 'r' )
                for line in zp_file:
                    entries = line.split()
                    if entries[0][0] == '#':
                        continue
                    zp_dict[int(entries[0])] = {'zp': float(entries[1]), 'label': int(entries[2])}
                zp_list.append(zp_dict)
                print "Read in {0} zeropoints from {1} for id_string {2}, operand {3}".format(len(zp_dict.keys())-2, zp_filename, calibration['id_strings'][i], calibration['operands'][i])

    # add the standard zp if necessary
    std_zp_dict = None
    if config['general']['use_standards']:
        std_zp_dict = dict()
        std_zp_dict['operand'] = None
        std_zp_dict['id_string'] = None
        zp_file = open(config['general']['stdzp_outfilename'], 'r')
        for line in zp_file:
            entries = line.split()
            std_zp_dict[int(entries[0])] = float(entries[1])
        print 'Read in the zeropoint to standards'
    
    # read in the CCD positions in the focal plane
    fp_xs, fp_ys = read_ccdpos()


    # which pixels exist?  check file names.
    pix_dict = dict()
    dir_files = os.listdir(globals_dir)
    n_globals = 0
    for filename in dir_files:
        if re.match('gobjs_' + str(fileband) + '_nside' + str(nside) + '_p', filename) is not None:
            pixel_match = re.search('p(\d+)', filename)
            if pixel_match is None:
                print "Problem parsing pixel number"
            pixel = int(pixel_match.group()[1:])
            pix_dict[pixel] = True
            response = None
            fname = globals_dir + '/' + filename
            response = subprocess.check_output(["wc","-l",fname])
            match = re.search('(\d+)', response)
            response = match.group(0)
            # print "response: %s" %response
            n_globals += int(response)

    pix_wobjs = sorted(pix_dict.keys())
    npix_wobjs = len(pix_wobjs)
    print "%d pixels have data; %d total" %(npix_wobjs, n_globals)
    
    
    n_max_std = 5000
    n_max_des = 5000
    std_data = { 'ras':np.zeros(n_max_std), 'decs':np.zeros(n_max_std), 'mags':np.zeros(n_max_std), 'mags_calib':np.zeros(n_max_std), 'mags_std':np.zeros(n_max_std), 'xs':np.zeros(n_max_std), 'ys':np.zeros(n_max_std), 'bad':np.zeros(n_max_std,dtype='bool') }
    des_data = { 'ras':np.zeros(n_max_des), 'decs':np.zeros(n_max_des), 'mags':np.zeros(n_max_des), 'mag_errs':np.zeros(n_max_des), 'mags_calib':np.zeros(n_max_des), 'mags_mean':np.zeros(n_max_des), 'mags_mean_calib':np.zeros(n_max_des), 'xs':np.zeros(n_max_des), 'ys':np.zeros(n_max_des), 'bad':np.zeros(n_max_des,dtype='bool') }
    std_index = 0
    des_index = 0
    
    validation_ras = [o['ra']-360. if o['ra']>270. else o['ra'] for o in validation_stars]
    validation_decs = [o['dec'] for o in validation_stars]
    
    bad_idvals = {}
    for zps in zp_list:
        bad_idvals[zps['id_string']] = {}
    
    for p in range(npix_wobjs):
    
        print "pixel {0}/{1}\r".format(p+1,npix_wobjs),
    
        pix = pix_wobjs[p]
        
        global_objs = read_global_objs(band, globals_dir, nside, nside, pix, verbose=False)
        if global_objs is None or len(global_objs) == 0:
            continue
    
        global_ras = [o.ra-360. if o.ra>270. else o.ra for o in global_objs]
        global_decs = [o.dec for o in global_objs]
    
        inds1, inds2, dists = spherematch( global_ras, global_decs, validation_ras, validation_decs, tol=1./3600. )
        
        for i in range(len(inds1)):
            go = global_objs[inds1[i]]
            
            if not filter_go(go, config):
                continue
            
            if go.ra > 270.:
                go.ra -= 360.
            ndet = len(go.objects)
            sdss_match = validation_stars[inds2[i]]
            
            # increase the array lengths if necessary
            if( std_index+ndet >= n_max_std ):
                for key in std_data.keys():
                    std_data[key] = np.hstack((std_data[key],np.zeros(n_max_std)))
                n_max_std += n_max_std
            
            ccds = np.array([o['ccd'] for o in go.objects],dtype='int')
            zp_ps = np.array([zp_phots[ccd] for ccd in ccds])
            mags_before = np.array([o['mag_psf'] for o in go.objects]) + zp_ps
            mags_after = np.array(mags_before)
            xs = np.array([o['x_image'] + fp_xs[o['ccd']] for o in go.objects])
            ys = np.array([o['y_image'] + fp_ys[o['ccd']] for o in go.objects])
            
            for j,obj in zip(range(ndet),go.objects):
                mags_after[j] = apply_zps(mags_after[j], obj, zp_list, bad_idvals)
                if config['general']['use_standards']:
                    mags_after[j] = apply_stdcal(mags_after[j], obj, zp_list, std_zp_dict)
            
            std_data['ras'][std_index:std_index+ndet] = go.ra*np.ones(ndet)
            std_data['decs'][std_index:std_index+ndet] = go.dec*np.ones(ndet)
            std_data['mags'][std_index:std_index+ndet] = mags_before
            std_data['mags_calib'][std_index:std_index+ndet] = mags_after
            std_data['mags_std'][std_index:std_index+ndet] = sdss_match['mag_psf']*np.ones(ndet)
            std_data['xs'][std_index:std_index+ndet] = xs
            std_data['ys'][std_index:std_index+ndet] = ys
            std_data['bad'][std_index:std_index+ndet] = (zp_ps<-900.)
            std_index += ndet
        
        # end loop over matches with standards
        
        # only look at a random sample if the list of objects is really long
        indices = np.array(range(len(global_objs)), dtype='int')
        if len(global_objs) > 100000:
            np.random.shuffle(indices)
            indices = indices[0:100000]
        
        # collect stats
        # good_objs = filter(filter_go, global_objs[indices])
        for g in indices:
            go = global_objs[g]
            
            if not filter_go(go, config):
                continue
            
            if go.ra > 270.:
                go.ra -= 360.
            
            ndet = len(go.objects)
            if ndet<2:
                continue
            
            # increase the array lengths if necessary
            if( des_index+ndet >= n_max_des ):
                for key in des_data.keys():
                    des_data[key] = np.hstack((des_data[key],np.zeros(n_max_des)))
                n_max_des += n_max_des
            
            ccds = np.array([o['ccd'] for o in go.objects],dtype='int')
            zp_ps = np.array([zp_phots[ccd] for ccd in ccds])
            
            good_indices = (zp_ps>-900)
            if np.sum(good_indices)<2:
                continue
            
            mags_before = np.array([o['mag_psf'] for o in go.objects]) + zp_ps
            mags_after = np.array(mags_before)
            mag_errors2 = np.array([o['magerr_psf']*o['magerr_psf'] for o in go.objects]) + magerr_sys2
            xs = np.array([o['x_image'] + fp_xs[o['ccd']] for o in go.objects])
            ys = np.array([o['y_image'] + fp_ys[o['ccd']] for o in go.objects])
            
            for i,obj in zip(range(ndet),go.objects):
                mags_after[i] = apply_zps(mags_after[i], obj, zp_list, bad_idvals)
                if config['general']['use_standards']:
                    mags_after[i] = apply_stdcal(mags_after[i], obj, zp_list, std_zp_dict)
            
            invsigma_array = 1.0/mag_errors2[good_indices]
            sum_invsigma2 = invsigma_array.sum()
            mag_before = (mags_before[good_indices]*invsigma_array).sum() / sum_invsigma2
            mag_after = (mags_after[good_indices]*invsigma_array).sum() / sum_invsigma2
            if mag_before<0.:
                print 'BAD MAG_BEFORE {0}'.format(mag_before)
                print mags_before
                print invsigma_array
                print sum_invsigma2
            
            des_data['ras'][des_index:des_index+ndet] = go.ra*np.ones(ndet)
            des_data['decs'][des_index:des_index+ndet] = go.dec*np.ones(ndet)
            des_data['mags'][des_index:des_index+ndet] = mags_before
            des_data['mag_errs'][des_index:des_index+ndet] = mag_errors2
            des_data['mags_calib'][des_index:des_index+ndet] = mags_after
            des_data['mags_mean'][des_index:des_index+ndet] = mag_before*np.ones(ndet)
            des_data['mags_mean_calib'][des_index:des_index+ndet] = mag_after*np.ones(ndet)
            des_data['xs'][des_index:des_index+ndet] = xs
            des_data['ys'][des_index:des_index+ndet] = ys
            des_data['bad'][des_index:des_index+ndet] = (zp_ps<-900.)
            
            des_index += ndet
    
    # crop the arrays
    good_indices = (std_data['bad'] == False)
    n_bad = np.sum(std_data['bad'])
    for key in std_data.keys():
        std_data[key] = std_data[key][good_indices]
        std_data[key] = std_data[key][0:std_index-n_bad]
    print '{0} bad zp_phots among {1} standard measurements'.format(int(n_bad), len(std_data['mags']))
    good_indices = (des_data['bad'] == False)
    n_bad = np.sum(des_data['bad'])
    for key in des_data.keys():
        des_data[key] = des_data[key][good_indices]
        des_data[key] = des_data[key][0:des_index-n_bad]
    print '{0} bad zp_phots among {1} DES measurements'.format(int(n_bad), len(des_data['mags']))
    
    print ""
    stddev_des_before = np.std(des_data['mags']-des_data['mags_mean'])
    stddev_des_after = np.std(des_data['mags_calib']-des_data['mags_mean_calib'])
    stddev_std_before = np.std(std_data['mags']-std_data['mags_std'])
    stddev_std_after = np.std(std_data['mags_calib']-std_data['mags_std'])
    print 'Std Dev before: {0}'.format(stddev_des_before)
    print 'Std Dev after: {0}'.format(stddev_des_after)
    
    # plots!
    
    fig = plt.figure()
    offset_before = std_data['mags']-std_data['mags_std']
    title = "Mag offset before cal, med %e, std dev %e" %(np.median(offset_before), np.std(offset_before))
    print title
    ax0 = fig.add_subplot(1,1,1, title=title)
    ax0.set_yscale('log')
    ax0.hist(offset_before, 100)
    pp.savefig()

    fig = plt.figure()
    offset_after = std_data['mags_calib']-std_data['mags_std']
    title = "Mag offset after cal, med %e, std dev %e" %(np.median(offset_after), np.std(offset_after))
    print title
    ax0 = fig.add_subplot(1,1,1, title=title)
    ax0.set_yscale('log')
    ax0.hist(offset_after, 100)
    pp.savefig()

    # histogram of diff of individual objects before calibration
    fig = plt.figure()
    diff_before = des_data['mags']-des_data['mags_mean']
    title = "Mag diff from mean before cal (std dev %e)" %np.std(diff_before)
    print title
    xlab = "Magnitudes (%s band)" %band
    ylab = ""
    ax0 = fig.add_subplot(1,1,1, xlabel=xlab, ylabel=ylab, title=title)
    ax0.set_yscale('log')
    ax0.hist(diff_before, 100)
    pp.savefig()

    # histogram of diff of individual objects after calibration
    fig = plt.figure()
    diff_after = des_data['mags_calib']-des_data['mags_mean_calib']
    title = "Mag diff from mean after cal (std dev %e)" %np.std(diff_after)
    print title
    xlab = "Magnitudes (%s band)" %band
    ax0 = fig.add_subplot(1,1,1, xlabel=xlab, ylabel=ylab, title=title)
    ax0.set_yscale('log')
    ax0.hist(diff_after, 100)
    pp.savefig()
    
    # diff of individual objects after calibration, vs mag
    fig = plt.figure()
    ax0 = fig.add_subplot(1,1,1, xlabel=xlab, ylabel=ylab, title=title)
    inbins = (des_data['mags_calib']>15.0) & (des_data['mags_calib']<21.0)
    mags_inbins = des_data['mags_calib'][inbins]
    resids_inbins = des_data['mags_calib'][inbins]-des_data['mags_mean_calib'][inbins]
    magbins = np.arange(15,21,0.2)
    indices = np.digitize(mags_inbins, magbins)
    means = np.zeros(len(magbins-1))
    errors = np.zeros(len(magbins-1))
    magbin_meds = np.zeros(len(magbins-1))
    for b in range(len(magbins)-1):
        magbin_meds[b] = 0.5*(magbins[b] + magbins[b+1])
        bin_resids = resids_inbins[indices==b]
        if len(bin_resids) == 0:
            continue
        means[b] = np.mean(abs(bin_resids))
        errors[b] = np.std(abs(bin_resids))/np.sqrt(len(bin_resids))
    ax0.errorbar(magbin_meds, means, yerr=errors, fmt='o')
    ax0.set_xlim(15.0,21.0)
    ax0.set_ylim(0.0,0.02)
    pp.savefig()
    
    
    title = "Resid of mean from stds before cal"
    fig = plt.figure()
    ylab = "DES - std mag"
    ax0 = fig.add_subplot(1,1,1, xlabel=xlab, ylabel=ylab, title=title)
    im0 = ax0.hexbin(std_data['mags'], std_data['mags']-std_data['mags_std'], bins='log', extent=(np.min(std_data['mags']), np.max(std_data['mags']), np.median(std_data['mags']-std_data['mags_calib'])-stddev_std_before, np.median(std_data['mags']-std_data['mags_calib'])+stddev_std_before)) 
    pp.savefig()
    
    fig = plt.figure()
    title = "Resid of mean from stds after cal"
    ax0 = fig.add_subplot(1,1,1, xlabel=xlab, ylabel=ylab, title=title)
    im0 = ax0.hexbin(std_data['mags_calib'], std_data['mags_calib']-std_data['mags_std'], bins='log', extent=(np.min(std_data['mags_calib']), np.max(std_data['mags_calib']), np.median(std_data['mags_calib']-std_data['mags_std'])-stddev_std_after, np.median(std_data['mags_calib']-std_data['mags_std'])+stddev_std_after)) 
    pp.savefig()
    
    fig = plt.figure()
    title = "Resid of mean from stds by ra and dec before cal"
    xlab = "RA"
    ylab = "Dec"
    ax0 = fig.add_subplot(211, xlabel=xlab, ylabel=ylab, title=title)
    im0 = ax0.hexbin(std_data['ras'], std_data['decs'], std_data['mags']-std_data['mags_std'], vmin=np.median(std_data['mags']-std_data['mags_std'])-stddev_std_before, vmax=np.median(std_data['mags']-std_data['mags_std'])+stddev_std_before)
    fig.colorbar(im0)
    ax0.set_aspect('equal')
    
    title = "Resid of mean from stds by ra and dec after cal"
    xlab = "RA"
    ylab = "Dec"
    ax1 = fig.add_subplot(212, xlabel=xlab, ylabel=ylab, title=title)
    im1 = ax1.hexbin(std_data['ras'], std_data['decs'], std_data['mags_calib']-std_data['mags_std'], vmin=np.median(std_data['mags_calib']-std_data['mags_std'])-stddev_std_after, vmax=np.median(std_data['mags_calib']-std_data['mags_std'])+stddev_std_after)
    fig.colorbar(im1)
    ax1.set_aspect('equal')
    pp.savefig()
    
    sns.set(style="white", rc={'image.cmap': colormap})
    
    fig = plt.figure()
    title = "Diff of objects' mags from mean, before cal"
    ax0 = fig.add_subplot(211, xlabel=xlab, ylabel=ylab, title=title)
    im0 = ax0.hexbin(des_data['ras'], des_data['decs'], des_data['mags']-des_data['mags_mean'], reduce_C_function = np.mean, vmin=np.median(des_data['mags']-des_data['mags_mean'])-stddev_des_before, vmax=np.median(des_data['mags']-des_data['mags_mean'])+stddev_des_before) 
    fig.colorbar(im0)
    ax0.set_aspect('equal')
    
    title = "Diff of objects' mags from mean, after cal"
    ax1 = fig.add_subplot(212, xlabel=xlab, ylabel=ylab, title=title)
    im1 = ax1.hexbin(des_data['ras'], des_data['decs'], des_data['mags_calib']-des_data['mags_mean_calib'], reduce_C_function = np.mean, vmin=np.median(des_data['mags_calib']-des_data['mags_mean_calib'])-stddev_des_after, vmax=np.median(des_data['mags_calib']-des_data['mags_mean_calib'])+stddev_des_after) 
    ax1.set_aspect('equal')
    fig.colorbar(im1)
    pp.savefig()
    
    fig = plt.figure()
    title = "Mag zeropoints (mean)"
    ax0 = fig.add_subplot(211, title=title)
    im0 = ax0.hexbin(des_data['ras'], des_data['decs'], des_data['mags_calib']-des_data['mags'], reduce_C_function = np.mean) 
    ax0.set_aspect('equal')
    fig.colorbar(im0)
    title = "Mag zeropoints (median)"
    ax1 = fig.add_subplot(212, title=title)
    im1 = ax1.hexbin(des_data['ras'], des_data['decs'], des_data['mags_calib']-des_data['mags'], reduce_C_function = np.median) 
    ax1.set_aspect('equal')
    fig.colorbar(im1)
    pp.savefig()
    
    
    fig = plt.figure()
    title = "Residual of indiv meas from stds before cal"
    ax0 = fig.add_subplot(1,1,1, title=title)
    im0 = ax0.hexbin(std_data['xs'], std_data['ys'], std_data['mags']-std_data['mags_std'], gridsize=200, reduce_C_function = np.mean, vmin=np.median(std_data['mags']-std_data['mags_std'])-stddev_std_before, vmax=np.median(std_data['mags']-std_data['mags_std'])+stddev_std_before) 
    fig.colorbar(im0)
    pp.savefig()

    fig = plt.figure()
    title = "Residual of indiv meas from stds after cal"
    ax0 = fig.add_subplot(1,1,1, title=title)
    im0 = ax0.hexbin(std_data['xs'], std_data['ys'], std_data['mags_calib']-std_data['mags_std'], gridsize=200, reduce_C_function = np.mean, vmin=np.median(std_data['mags_calib']-std_data['mags_std'])-stddev_std_after, vmax=np.median(std_data['mags_calib']-std_data['mags_std'])+stddev_std_after) 
    fig.colorbar(im0)
    pp.savefig()

    fig = plt.figure()
    title = "Residual of indiv meas from DES mean before cal"
    ax0 = fig.add_subplot(1,1,1, title=title)
    im0 = ax0.hexbin(des_data['xs'], des_data['ys'], des_data['mags']-des_data['mags_mean'], gridsize=200, reduce_C_function = np.mean, vmin=-stddev_des_before, vmax=stddev_des_before) 
    fig.colorbar(im0)
    pp.savefig()

    fig = plt.figure()
    title = "Residual of indiv meas from DES mean after cal"
    ax0 = fig.add_subplot(1,1,1, title=title)
    im0 = ax0.hexbin(des_data['xs'], des_data['ys'], des_data['mags_calib']-des_data['mags_mean_calib'], gridsize=200, reduce_C_function = np.mean, vmin=-stddev_des_after, vmax=stddev_des_after) 
    fig.colorbar(im0)
    pp.savefig()


    pp.close()


def main():
    
    if len(sys.argv) != 2:
        print "Usage: nebencal_plots.py config_filename"
        print "       To print out a default config file, run: nebencal_plots.py default"
        exit(1)
    elif sys.argv[1] == 'default':
        config = open( os.path.dirname(os.path.realpath(__file__)) + '/configs/config_sample', 'r' )
        print "\n### DEFAULT EXAMPLE CONFIG FILE ###\n"
        for line in config:
            sys.stdout.write(line)
        exit(0)
    
    config_filename = sys.argv[1]
    print "Making nebencalibration plots using config file %s!" %config_filename
    
    config_file = open(config_filename)
    config = yaml.load(config_file)
    
    make_plots(config)



if __name__ == '__main__':
    main()

