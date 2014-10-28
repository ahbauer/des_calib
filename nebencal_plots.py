import sys
import os
import math
import re
import subprocess
# from operator import itemgetter
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
# sb_dark = seaborn.dark_palette("skyblue", 8, reverse=True)
# seaborn.set(palette = seaborn.color_palette("coolwarm", 7))
# seaborn.set_palette("hls")
sns.set(rc={'image.cmap': "RdBu_r"})
from operator import itemgetter, attrgetter
from rtree import index
# from scipy.stats import scoreatpercentile
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
    
    # palette = sns.color_palette("coolwarm")
    # sns.set_palette(palette) 
    
    globals_dir = config['general']['globals_dir']
    current_dir = os.getcwd()
    use_spxzps = False
    use_imgzps = True
    use_expzps = True
    use_exp_fpr_zps = False

    nside = config['general']['nside_file']
    n_ccds = 63
    match_radius = 1.0/3600.
    magerr_sys2 = 0.0004

    plot_file = 'nebencal_plots_' + band + '.pdf'
    pp = PdfPages(plot_file)

    mag_classes={}
    
    fileband = band
    # if band == 'y':
    #     fileband = 'Y'

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
    posfile = open("/Users/bauer/surveys/DES/ccdPos-v2.par", 'r')
    posarray = posfile.readlines()
    fp_xs = []
    fp_ys = []
    fp_xindices = []
    fp_yindices = []
    # put in a dummy ccd=0
    fp_xs.append(0.)
    fp_ys.append(0.)
    fp_xindices.append(0.)
    fp_yindices.append(0.)
    for i in range(21,83,1):
        entries = posarray[i].split(" ")
        fp_yindices.append(int(entries[2])-1)
        fp_xindices.append(int(entries[3])-1)
        fp_xs.append(66.6667*(float(entries[4])-211.0605) - 1024) # in pixels
        fp_ys.append(66.6667*float(entries[5]) - 2048)
    # fp_ys.append(fp_xs[-1])
    # fp_xs.append(fp_ys[-1])
    fp_yindices.append(fp_yindices[-1])
    fp_xindices.append(fp_xindices[-1])
    fp_xoffsets = [4,3,2,1,1,0,0,1,1,2,3,4]
    fp_xindices = 2*np.array(fp_xindices)
    fp_yindices = np.array(fp_yindices)
    print "parsed focal plane positions for %d ccds" %len(fp_xs)


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

    n_hist = n_globals # 5000000
    image_ras = dict()
    image_decs = dict()
    image_diffas = dict()
    image_diffbs = dict()
    image_ns = dict()
    image_ccds = dict()
    image_resids_b = dict()
    image_resids_a = dict()
    image_befores = dict()
    image_afters = dict()
    sdss_mags = dict()
    validation_ras = [o['ra']-360. if o['ra']>270. else o['ra'] for o in validation_stars]
    validation_decs = [o['dec'] for o in validation_stars]
    standard_ras = [o['ra']-360. if o['ra']>270. else o['ra'] for o in standard_stars]
    standard_decs = [o['dec'] for o in standard_stars]
    sum_rms_before = 0.
    sum_rms_after = 0.
    n_rms = 0
    rms_befores = np.zeros(n_hist)
    rms_afters = np.zeros(n_hist)
    meandiff_befores = np.zeros(n_hist)
    meandiff_afters = np.zeros(n_hist)
    hist_count = 0
    total_magdiffsb = []
    total_magdiffsa = []

    plot_resids_b2 = []
    plot_resids_a2 = []
    plot_resids_c2 = []
    plot_resids_d2 = []
    plot_resids_e2 = []
    plot_resids_f2 = []
    plot_resids_b2_xs = []
    plot_resids_b2_ys = []
    plot_resids_ras = []
    plot_resids_decs = []
    rel_resids_magafter = []
    rel_resids_b2_xs = []
    rel_resids_b2_ys = []
    rel_resids_ras = []
    rel_resids_decs = []
    
    bad_idvals = {}
    for zps in zp_list:
        bad_idvals[zps['id_string']] = {}
    
    for p in range(npix_wobjs):
    
        print "pixel {0}/{1}\r".format(p+1,npix_wobjs),
    
        pix = pix_wobjs[p]
        
        global_objs = read_global_objs(band, globals_dir, nside, nside, pix, verbose=False)
        if global_objs is None or len(global_objs) == 0:
            continue
        # filename = 'gobjs_' + str(band) + '_nside' + str(nside) + '_p' + str(pix)
        # file = open(os.path.join(globals_dir,filename), 'rb')
        # global_objs = cPickle.load(file)
        # file.close()
    
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
            
            mags_before = np.zeros(ndet)
            mags_after = np.zeros(ndet)
            mag_errors2 = np.zeros(ndet)
            iids = np.zeros(ndet, dtype='int')
            ccds = np.zeros(ndet, dtype='int')
            xs = np.zeros(ndet)
            ys = np.zeros(ndet)
            d=0
            for d0 in range(ndet):
                image_id = go.objects[d0]['image_id']
                if image_id == 1:
                    continue
                mag_psf = go.objects[d0]['mag_psf']
                magerr_psf = go.objects[d0]['magerr_psf']
                xs[d]= go.objects[d0]['x_image']
                ys[d] = go.objects[d0]['y_image']
                ccd = go.objects[d0]['ccd']
                spx = int(4*np.floor(ys[d]/512.) + np.floor(xs[d]/512.) + 32*(ccd))
                exposureid = go.objects[d0]['exposureid']
                image_ccds[image_id] = ccd
                ccds[d] = ccd
                mags_before[d] = mag_psf
                mags_after[d] = mag_psf
                badmag = False
                if ccd in zp_phots:
                    mags_before[d] += zp_phots[ccd]
                    mags_after[d] += zp_phots[ccd]
                else:
                    badmag = True
                # if config['general']['use_color_terms']:
                #     mags_before[d] = apply_color_term(mags_before[d], go, fp_xs, fp_ys, mag_classes)
                #     mags_after[d] = apply_color_term(mags_after[d], go, fp_xs, fp_ys, mag_classes)
                mags_after[d] = apply_zps( mags_after[d], go.objects[d0], zp_list, bad_idvals )
                if not mags_after[d]:
                    badmag = True
                    continue
                if config['general']['use_standards']:
                    mags_after[d] = apply_stdcal( mags_after[d], go.objects[d0], zp_list, std_zp_dict )
                    if not mags_after[d]:
                        badmag = True
                        continue
                mag_errors2[d] = magerr_psf*magerr_psf + magerr_sys2
                iids[d] = image_id
                total_magdiffsb.append(mags_before[d] - sdss_match['mag_psf'])
                total_magdiffsa.append(mags_after[d] - sdss_match['mag_psf'])
                try:
                    image_ras[image_id] += go.ra
                    image_decs[image_id] += go.dec
                    image_diffbs[image_id] += mags_before[d] - sdss_match['mag_psf']
                    image_diffas[image_id] += mags_after[d] - sdss_match['mag_psf']
                    image_befores[image_id] += mags_before[d]
                    image_afters[image_id] += mags_after[d]
                    sdss_mags[image_id] += sdss_match['mag_psf']
                    image_ns[image_id] += 1
                except KeyError:
                    # the current image id is not in the image_ids dictionary yet
                    image_ras[image_id] = go.ra
                    image_decs[image_id] = go.dec
                    image_diffbs[image_id] = mags_before[d] - sdss_match['mag_psf']
                    image_diffas[image_id] = mags_after[d] - sdss_match['mag_psf']
                    image_befores[image_id] = mags_before[d]
                    image_afters[image_id] = mags_after[d]
                    sdss_mags[image_id] = sdss_match['mag_psf']
                    image_ns[image_id] = 1
                d+=1
            ndet=d
            if ndet < 2:
                continue
            mags_before = mags_before[0:ndet]
            mags_after = mags_after[0:ndet]
            mag_errors2 = mag_errors2[0:ndet]
            iids = iids[0:ndet]
            ccds = ccds[0:ndet]
            xs = xs[0:ndet]
            ys = ys[0:ndet]
            invsigma_array = 1.0/mag_errors2
            sum_invsigma2 = invsigma_array.sum()
            sum_m_before = (mags_before*invsigma_array).sum() / sum_invsigma2
            sum_m_after = (mags_after*invsigma_array).sum() / sum_invsigma2
        
            for d in range(ndet):
                plot_resids_b2.append(mags_before[d] - sdss_match['mag_psf'])
                plot_resids_b2_xs.append(xs[d] + fp_xs[ccds[d]])
                plot_resids_b2_ys.append(ys[d] + fp_ys[ccds[d]])
                plot_resids_a2.append(mags_after[d] - sdss_match['mag_psf'])
                plot_resids_ras.append(go.ra)
                plot_resids_decs.append(go.dec)
                try:
                    image_resids_b[iids[d]] += mags_before[d] - sdss_match['mag_psf']
                    image_resids_a[iids[d]] += mags_after[d] - sdss_match['mag_psf']
                except KeyError:
                    image_resids_b[iids[d]] = mags_before[d] - sdss_match['mag_psf']
                    image_resids_a[iids[d]] = mags_after[d] - sdss_match['mag_psf']

            # calculate rms before and after
            rms_before = np.std(mags_before-sdss_match['mag_psf'])
            rms_after = np.std(mags_after-sdss_match['mag_psf'])
            # meandiff_before = sum_m_before - sdss_match['mag_psf']
            # meandiff_after = sum_m_after - sdss_match['mag_psf']
            # if meandiff_after < -3.:
            #     print "%f %f %f %f %d" %(sum_m_before, sum_m_after, zps[image_id], sdss_match['mag_psf'], image_id)
            sum_rms_before += rms_before
            sum_rms_after += rms_after
        
            rms_befores[hist_count] = rms_before
            rms_afters[hist_count] = rms_after
            # meandiff_befores[hist_count] = meandiff_before
            # meandiff_afters[hist_count] = meandiff_after
            hist_count += 1
        
            n_rms += 1
        
        # end loop over matches with standards
        
        # only look at a random sample if the list of objects is really long
        if len(global_objs) > 100000:
            indices = range(len(global_objs))
            np.random.shuffle(indices)
            indices = indices[0:100000]
        else:
            indices = range(len(global_objs))
        
        for i in indices:
            go = global_objs[i]
            
            if not filter_go(go, config):
                continue
            
            if go.ra > 270.:
                go.ra -= 360.
            ndet = len(go.objects)
            
            # want: plot_resids_c2, plot_resids_d2, copies of: plot_resids_b2_xs, plot_resids_b2_ys, plot_resids_ras, plot_resids_decs
            mags_before = np.zeros(ndet)
            mags_after = np.zeros(ndet)
            mag_errors2 = np.zeros(ndet)
            ccds = np.zeros(ndet, dtype='int')
            xs = np.zeros(ndet)
            ys = np.zeros(ndet)
            d=0
            for d0 in range(ndet):
                image_id = go.objects[d0]['image_id']
                if image_id == 1:
                    continue
                mag_psf = go.objects[d0]['mag_psf']
                magerr_psf = go.objects[d0]['magerr_psf']
                xs[d]= go.objects[d0]['x_image']
                ys[d] = go.objects[d0]['y_image']
                ccd = go.objects[d0]['ccd']
                spx = int(4*np.floor(ys[d]/512.) + np.floor(xs[d]/512.) + 32*(ccd))
                exposureid = go.objects[d0]['exposureid']
                ccds[d] = ccd
                mags_before[d] = mag_psf
                mags_after[d] = mag_psf
                mag_errors2[d] = magerr_psf*magerr_psf + magerr_sys2
                badmag = False
                if ccd in zp_phots:
                    mags_before[d] += zp_phots[ccd]
                    mags_after[d] += zp_phots[ccd]
                else:
                    badmag = True
                # if config['general']['use_color_terms']:
                #     mags_before[d] = apply_color_term(mags_before[d], go, fp_xs, fp_ys, mag_classes)
                #     mags_after[d] = apply_color_term(mags_after[d], go, fp_xs, fp_ys, mag_classes)
                mags_after[d] = apply_zps( mags_after[d], go.objects[d0], zp_list, bad_idvals )
                if not mags_after[d]:
                    badmag = True
                    continue
                if config['general']['use_standards']:
                    mags_after[d] = apply_stdcal( mags_after[d], go.objects[d0], zp_list, std_zp_dict )
                    if not mags_after[d]:
                        badmag = True
                        continue
                d+=1
            ndet=d
            if ndet < 2:
                continue
            mags_before = mags_before[0:ndet]
            mags_after = mags_after[0:ndet]
            mag_errors2 = mag_errors2[0:ndet]
            xs = xs[0:ndet]
            ys = ys[0:ndet]
            invsigma_array = 1.0/mag_errors2
            sum_invsigma2 = invsigma_array.sum()
            sum_m_before = (mags_before*invsigma_array).sum() / sum_invsigma2
            sum_m_after = (mags_after*invsigma_array).sum() / sum_invsigma2
            
            for d in range(ndet):
                plot_resids_c2.append(mags_before[d]-sum_m_before)
                plot_resids_d2.append(mags_after[d]-sum_m_after)
                plot_resids_e2.append(mags_after[d]-sum_m_before)
                plot_resids_f2.append(mags_after[d]-mags_before[d])
                rel_resids_b2_xs.append(xs[d] + fp_xs[ccds[d]])
                rel_resids_b2_ys.append(ys[d] + fp_ys[ccds[d]])
                rel_resids_ras.append(go.ra)
                rel_resids_decs.append(go.dec)
                rel_resids_magafter.append(mags_after[d])
    
    print ""
    
    for zps in zp_list:
        print 'zp id_string {0}: {1} good zps and {2} bad.'.format(zps['id_string'], len(zps.keys())-2, len(bad_idvals[zps['id_string']].keys()))

    plot_resids_b2_xs = np.array(plot_resids_b2_xs)
    plot_resids_b2_ys = np.array(plot_resids_b2_ys)
    plot_resids_b2_rs = np.sqrt(plot_resids_b2_xs*plot_resids_b2_xs + plot_resids_b2_ys*plot_resids_b2_ys)
    plot_resids_a2 = np.array(plot_resids_a2)
    plot_resids_b2 = np.array(plot_resids_b2)
    plot_resids_c2 = np.array(plot_resids_c2)
    plot_resids_d2 = np.array(plot_resids_d2)
    plot_resids_e2 = np.array(plot_resids_e2)
    plot_resids_f2 = np.array(plot_resids_f2)
    plot_resids_ras = np.array(plot_resids_ras)
    plot_resids_decs = np.array(plot_resids_decs)
    rel_resids_ras = np.array(rel_resids_ras)
    rel_resids_decs = np.array(rel_resids_decs)
    rel_resids_b2_xs = np.array(rel_resids_b2_xs)
    rel_resids_b2_ys = np.array(rel_resids_b2_ys)
    rel_resids_magafter = np.array(rel_resids_magafter)
    
    for iid in image_resids_a.keys():
        image_ras[iid] /= image_ns[iid]
        image_decs[iid] /= image_ns[iid]
        image_diffas[iid] /= image_ns[iid]
        image_diffbs[iid] /= image_ns[iid]
        image_resids_b[iid] /= image_ns[iid]
        image_resids_a[iid] /= image_ns[iid]
        image_befores[iid] /= image_ns[iid]
        image_afters[iid] /= image_ns[iid]
        sdss_mags[iid] /= image_ns[iid]

    sum_rms_before /= n_rms
    sum_rms_after /= n_rms
    print
    print "RMS before: %f" %sum_rms_before
    print "RMS after: %f" %sum_rms_after

    # plots!

    if len(total_magdiffsa) > 100000:
            total_magdiffsb = random.sample(total_magdiffsb, 100000)
            total_magdiffsa = random.sample(total_magdiffsa, 100000)
    
    rel_indices = range(len(rel_resids_ras))
    if len(rel_resids_ras) > 50000:
        np.random.shuffle(rel_indices)
        rel_indices = rel_indices[0:50000]
        
    fig = plt.figure()
    onesigma = (np.percentile(total_magdiffsb, 84) - np.percentile(total_magdiffsb, 16))/2.
    # title = "Mag offset before calibration, med %e, rms %e" %(np.median(total_magdiffsb), np.std(total_magdiffsb))
    title = "Mag offset before calibration, med %e, 68%% error %e" %(np.median(total_magdiffsb), onesigma)
    print title
    ax0 = fig.add_subplot(1,1,1, title=title)
    ax0.set_yscale('log')
    ax0.hist(total_magdiffsb, 100)
    pp.savefig()

    fig = plt.figure()
    onesigma = (np.percentile(total_magdiffsa, 84) - np.percentile(total_magdiffsa, 16))/2.
    # title = "Mag offset after calibration, med %e, rms %e" %(np.median(total_magdiffsa), np.std(total_magdiffsa))
    title = "Mag offset after calibration, med %e, 68%% error %e" %(np.median(total_magdiffsa), onesigma)
    print title
    ax0 = fig.add_subplot(1,1,1, title=title)
    ax0.set_yscale('log')
    ax0.hist(total_magdiffsa, 100)
    pp.savefig()

    # histogram of diff of individual objects before calibration
    fig = plt.figure()
    title = "Mag diff from mean before cal (std dev %e)" %np.std(plot_resids_c2)
    print title
    xlab = "Magnitudes (%s band)" %band
    ylab = ""
    ax0 = fig.add_subplot(1,1,1, xlabel=xlab, ylabel=ylab, title=title)
    ax0.set_yscale('log')
    ax0.hist(plot_resids_c2, 50)
    pp.savefig()

    # histogram of diff of individual objects after calibration
    fig = plt.figure()
    title = "Mag diff from mean after cal (std dev %e)" %np.std(plot_resids_d2[rel_indices])
    print title
    xlab = "Magnitudes (%s band)" %band
    ax0 = fig.add_subplot(1,1,1, xlabel=xlab, ylabel=ylab, title=title)
    ax0.set_yscale('log')
    ax0.hist(plot_resids_d2[rel_indices], 50)
    pp.savefig()
    
    # diff of individual objects after calibration, vs mag
    fig = plt.figure()
    title = "Mag diff from mean after cal (std dev %e)" %np.std(plot_resids_d2[rel_indices])
    xlab = "Magnitudes (%s band)" %band
    ax0 = fig.add_subplot(1,1,1, xlabel=xlab, ylabel=ylab, title=title)
    # ax0.hexbin(rel_resids_magafter[rel_indices], plot_resids_d2[rel_indices], bins='log', extent=(np.min(rel_resids_magafter[rel_indices]),np.max(rel_resids_magafter[rel_indices]), -0.05, 0.05 ))
    # ax0.hexbin(rel_resids_magafter[rel_indices], abs(plot_resids_d2[rel_indices]), bins='log', extent=(np.min(rel_resids_magafter[rel_indices]),np.max(rel_resids_magafter[rel_indices]), 0.0, 0.02 ))
    mags_inbins = rel_resids_magafter[(rel_resids_magafter[rel_indices]>15.0) & (rel_resids_magafter[rel_indices]<21.0)]
    resids_inbins = plot_resids_d2[(rel_resids_magafter[rel_indices]>15.0) & (rel_resids_magafter[rel_indices]<21.0)]
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
    
    # zp by ra and dec
    ra_array = []
    dec_array = []
    diffb_array = []
    diffa_array = []
    sdss_array = []
    image_before_array = []
    image_after_array = []
    n_array = []
    ccd_array = np.zeros(n_ccds+1)
    ras_35 = []
    decs_35 = []
    for key in image_resids_a:
        ra_array.append(image_ras[key])
        dec_array.append(image_decs[key])
        diffb_array.append(image_diffbs[key])
        diffa_array.append(image_diffas[key])
        n_array.append(image_ns[key])
        image_before_array.append(image_befores[key])
        sdss_array.append(sdss_mags[key])
        image_after_array.append(image_afters[key])
        ccd_array[image_ccds[key]] += 1
        if image_ccds[key] == 35:
            ras_35.append(image_ras[key])
            decs_35.append(image_decs[key])
    ra_array = np.array(ra_array)
    dec_array = np.array(dec_array)
    diffb_array = np.array(diffb_array)
    diffa_array = np.array(diffa_array)

    mdiffstdb = np.std(total_magdiffsb)
    mdiffstda = np.std(total_magdiffsa)

    title = "Resid of mean from stds before cal"
    fig = plt.figure()
    ax0 = fig.add_subplot(1,1,1, xlabel=xlab, ylabel=ylab, title=title)
    ylab = "DES - std mag"
    im0 = ax0.hexbin(image_before_array, diffb_array, bins='log', extent=(np.min(image_before_array), np.max(image_before_array), np.median(total_magdiffsb)-3*mdiffstdb, np.median(total_magdiffsb)+3*mdiffstdb)) 
    fig.colorbar(im0)
    pp.savefig()

    fig = plt.figure()
    title = "Resid of mean from stds after cal"
    ax0 = fig.add_subplot(1,1,1, xlabel=xlab, ylabel=ylab, title=title)
    im0 = ax0.hexbin(image_after_array, diffa_array, bins='log', extent=(np.min(image_after_array), np.max(image_after_array), np.median(total_magdiffsa)-3*mdiffstda, np.median(total_magdiffsa)+3*mdiffstda)) 
    fig.colorbar(im0)
    pp.savefig()



    fig = plt.figure()
    title = "Standard star locations"
    xlab = "RA"
    ylab = "Dec"
    ax0 = fig.add_subplot(1,1,1, xlabel=xlab, ylabel=ylab, title=title)
    im0 = ax0.hexbin(standard_ras, standard_decs, extent=(np.min(ra_array),np.max(ra_array),np.min(dec_array),np.max(dec_array))) 
    # fig.colorbar(im0)
    pp.savefig()
    
    fig = plt.figure()
    title = "Pointings (Central locations of CCD 35)"
    xlab = "RA"
    ylab = "Dec"
    ax0 = fig.add_subplot(1,1,1, xlabel=xlab, ylabel=ylab, title=title)
    print "%d instances of ccd 35" %len(ras_35)
    im0 = ax0.hexbin(ras_35, decs_35, bins='log',extent=(np.min(ra_array),np.max(ra_array),np.min(dec_array),np.max(dec_array)))
    pp.savefig()

    # the plots get really big at this point, so let's try plotting a subsample.
    if len(ra_array) > 100000:
        indices = range(len(ra_array))
        np.random.shuffle(indices)
        indices = indices[0:100000]
    else:
        indices = range(len(ra_array))
    
    fig = plt.figure()
    title = "Resid of mean from stds by ra and dec before cal"
    xlab = "RA"
    ylab = "Dec"
    ax0 = fig.add_subplot(211, xlabel=xlab, ylabel=ylab, title=title)
    im0 = ax0.hexbin(ra_array[indices], dec_array[indices], diffb_array[indices], vmin=np.median(total_magdiffsb)-2*mdiffstdb, vmax=np.median(total_magdiffsb)+2*mdiffstdb,gridsize=(int(np.max(ra_array)-np.min(ra_array)),int(np.max(dec_array)-np.min(dec_array))))
    fig.colorbar(im0)
    ax0.set_aspect('equal')
    
    title = "Resid of mean from stds by ra and dec after cal"
    xlab = "RA"
    ylab = "Dec"
    ax1 = fig.add_subplot(212, xlabel=xlab, ylabel=ylab, title=title)
    im1 = ax1.hexbin(ra_array[indices], dec_array[indices], diffa_array[indices], vmin=np.median(total_magdiffsa)-2*mdiffstda, vmax=np.median(total_magdiffsa)+2*mdiffstda,gridsize=(int(np.max(ra_array)-np.min(ra_array)),int(np.max(dec_array)-np.min(dec_array))))
    fig.colorbar(im1)
    ax1.set_aspect('equal')
    pp.savefig()
    
    fig = plt.figure()
    title = "Resid of mean from stds by ra after cal"
    ax0 = fig.add_subplot(111, xlabel=xlab, ylabel=ylab, title=title)
    ax0.hexbin(ra_array[indices],diffa_array[indices])
    pp.savefig()

    # the plots get really big at this point, so let's try plotting a subsample.
    if len(plot_resids_b2) > 50000:
        indices = range(len(plot_resids_b2))
        np.random.shuffle(indices)
        indices = indices[0:50000]
    else:
        indices = range(len(plot_resids_b2))
    

    fig = plt.figure()
    title = "Diff of objects' mags from mean, before cal"
    ax0 = fig.add_subplot(211, title=title)
    im0 = ax0.hexbin(rel_resids_ras[rel_indices], rel_resids_decs[rel_indices], plot_resids_c2[rel_indices], reduce_C_function = np.mean, vmin=np.median(plot_resids_c2[rel_indices])-sum_rms_before, vmax=np.median(plot_resids_c2[rel_indices])+sum_rms_before) 
    fig.colorbar(im0)
    ax0.set_aspect('equal')
    
    title = "Diff of objects' mags from mean, after cal"
    ax1 = fig.add_subplot(212, title=title)
    im1 = ax1.hexbin(rel_resids_ras[rel_indices], rel_resids_decs[rel_indices], plot_resids_d2[rel_indices], reduce_C_function = np.mean, vmin=np.median(plot_resids_d2[rel_indices])-sum_rms_after, vmax=np.median(plot_resids_d2[rel_indices])+sum_rms_after) 
    ax1.set_aspect('equal')
    fig.colorbar(im1)
    pp.savefig()
    
    fig = plt.figure()
    title = "Mag zeropoints (mean)"
    ax0 = fig.add_subplot(211, title=title)
    im0 = ax0.hexbin(rel_resids_ras[rel_indices], rel_resids_decs[rel_indices], plot_resids_f2[rel_indices], reduce_C_function = np.mean) 
    ax0.set_aspect('equal')
    fig.colorbar(im0)
    title = "Mag zeropoints (median)"
    ax1 = fig.add_subplot(212, title=title)
    im1 = ax1.hexbin(rel_resids_ras[rel_indices], rel_resids_decs[rel_indices], plot_resids_f2[rel_indices], reduce_C_function = np.median) 
    ax1.set_aspect('equal')
    fig.colorbar(im1)
    pp.savefig()
    
    
    fig = plt.figure()
    title = "Residual of indiv meas from stds before cal"
    ax0 = fig.add_subplot(1,1,1, title=title)
    #mdiffstdb
    im0 = ax0.hexbin(plot_resids_b2_xs[indices], plot_resids_b2_ys[indices], plot_resids_b2[indices], gridsize=400, reduce_C_function = np.mean, vmin=np.median(total_magdiffsb)-3*sum_rms_before, vmax=np.median(total_magdiffsb)+3*sum_rms_before) 
    fig.colorbar(im0)
    pp.savefig()

    fig = plt.figure()
    title = "Residual of indiv meas from stds after cal"
    ax0 = fig.add_subplot(1,1,1, title=title)
    #mdiffstda
    im0 = ax0.hexbin(plot_resids_b2_xs[indices], plot_resids_b2_ys[indices], plot_resids_a2[indices], gridsize=400, reduce_C_function = np.mean, vmin=np.median(total_magdiffsa)-3*sum_rms_after, vmax=np.median(total_magdiffsa)+3*sum_rms_after) 
    fig.colorbar(im0)
    pp.savefig()

    fig = plt.figure()
    title = "Residual of indiv meas from DES mean before cal"
    ax0 = fig.add_subplot(1,1,1, title=title)
    im0 = ax0.hexbin(rel_resids_b2_xs[rel_indices], rel_resids_b2_ys[rel_indices], plot_resids_c2[rel_indices], gridsize=400, reduce_C_function = np.mean, vmin=-3*sum_rms_before, vmax=3*sum_rms_before) 
    fig.colorbar(im0)
    pp.savefig()

    fig = plt.figure()
    title = "Residual of indiv meas from DES mean after cal"
    ax0 = fig.add_subplot(1,1,1, title=title)
    im0 = ax0.hexbin(rel_resids_b2_xs[rel_indices], rel_resids_b2_ys[rel_indices], plot_resids_d2[rel_indices], gridsize=400, reduce_C_function = np.mean, vmin=-3*sum_rms_after, vmax=3*sum_rms_after) 
    fig.colorbar(im0)
    pp.savefig()

    fig = plt.figure()
    title = "Residual of indiv meas after cal from DES mean before cal"
    ax0 = fig.add_subplot(1,1,1, title=title)
    im0 = ax0.hexbin(rel_resids_b2_xs[rel_indices], rel_resids_b2_ys[rel_indices], plot_resids_e2[rel_indices], gridsize=400, reduce_C_function = np.median, vmin=np.median(plot_resids_e2[rel_indices])-3*sum_rms_after, vmax=np.median(plot_resids_e2[rel_indices])+3*sum_rms_after) 
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

