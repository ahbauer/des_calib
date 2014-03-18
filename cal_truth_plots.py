import sys
import os
import math
import re
import subprocess
# from operator import itemgetter
import random
import cPickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from operator import itemgetter, attrgetter
# from scipy.stats import scoreatpercentile
from pyspherematch import spherematch

good_quality_magerr = 0.01
def good_quality(star):
    if star['magerr_psf'] < 0.05:
        return True
    return False
    
class global_object(object):
    def __init__(self):
        self.ra = None
        self.dec = None
        self.objects = []

class object(object):
    def __init__(self):
        self.ra = None
        self.dec = None
        self.mag = None
        self.mag_err = None


band = 'g'

if len(sys.argv) == 2:
    band = sys.argv[1]

globals_dir = '/Users/bauer/surveys/DES/y1p1/equatorial'
current_dir = os.getcwd()
use_spxzps = True
use_imgzps = True
use_expzps = True

nside = 32
n_ccds = 63
match_radius = 1.0/3600.

plot_file = 'nebencal_truth_plots_' + band + '.pdf'
pp = PdfPages(plot_file)


# read in the nominal ccd offsets (zp_phots), which will be our starting point
zp_phot_filename = "/Users/bauer/surveys/DES/zp_phots/" + band + ".dat"
zp_phot_file = open( zp_phot_filename, 'r' )
zp_phots = np.zeros(n_ccds+1)
for line in zp_phot_file:
    entries = line.split()
    if entries[0][0] == '#':
        continue
    zp_phots[int(entries[0])] = -1.0*float(entries[4])
print "read in ccd offsets (zp_phots)"

# read in the superpixel zero point solutions
spx_zps = None
if use_spxzps:
    spx_zps = dict()
    spx_zp_array = []
    spx_zp_file = current_dir + '/nebencal_spx_zps_' + str(band)
    file = open(spx_zp_file, 'r')
    filelines = file.readlines()
    for line in filelines:
        entries = line.split()
        spx_zps[int(entries[0])] = float(entries[1])
        spx_zp_array.append(float(entries[1]))
    print "read in %d superpixel zps" %len(spx_zps.keys())


# read in the exposure zero point solutions
exp_zps = None
if use_expzps:
    exp_zps = dict()
    exp_zp_array = []
    exp_zp_file = current_dir + '/nebencal_exp_zps_' + str(band)
    file = open(exp_zp_file, 'r')
    filelines = file.readlines()
    for line in filelines:
        entries = line.split()
        exp_zps[int(entries[0])] = float(entries[1])
        exp_zp_array.append(float(entries[1]))
    print "read in %d exposure zps" %len(exp_zps.keys())


# read in the zero point solutions
zps = None
if use_imgzps:
    zps = dict()
    zp_array = []
    zp_file = current_dir + '/nebencal_img_zps_' + str(band)
    file = open(zp_file, 'r')
    filelines = file.readlines()
    for line in filelines:
        entries = line.split()
        zps[int(entries[0])] = float(entries[1])
        zp_array.append(float(entries[1]))
    print "read in %d image zps" %len(zps.keys())


# read in the CCD positions in the focal plane
posfile = open("/Users/bauer/surveys/DES/ccdPos-v2.par", 'r')
posarray = posfile.readlines()
fp_xs = []
fp_ys = []
fp_xindices = []
fp_yindices = []
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


precam_ras = []
precam_decs = []
# read in the precam standards
precam_path = "/Users/bauer/surveys/DES/precam/PreCamStandarStars"
precam_name = band.upper() + ".Stand1percent.s"
precam_file = open( os.path.join(precam_path, precam_name), 'r' )
count = 0
for line in precam_file:
    entries = line.split(" ")
    if( entries[0][0] == '#' ):
        continue
    ra = float(entries[1])
    if ra > 300.:
        ra -= 360.;
    precam_ras.append(ra)
    precam_decs.append(float(entries[2]))
print "Read in %d PreCam standards" %len(precam_ras)


# read in the truth 
# sdssfile = open("/Users/bauer/surveys/DES/y1p1/equatorial/sdss/SDSSDR10_SouthGalCap/SouthGalCapStdCat_DES.csv", 'r')
sdssfile = open("/Users/bauer/surveys/DES/y1p1/equatorial/sdss/SDSSDR10_SouthGalCap/stripe82_sample2.csv", 'r')
sdss_objs = []
count = 0
for line in sdssfile:
    entries = line.split(",")
    # header!                                                                   
    if entries[0] == 'id':
        continue
    sdss_obj = object()
    sdss_obj.ra = float(entries[2])
    if sdss_obj.ra > 300.:
        sdss_obj.ra -= 360.
    sdss_obj.dec = float(entries[3])
    index = 0
    if band == 'u':
        index = 4
    elif band == 'g':
        index = 5
    elif band == 'r':
        index = 6
    elif band == 'i':
        index = 7
    elif band == 'z':
        index = 8
    elif band == 'y':
        index = 9
    else:
        print "Um, desired band = %s" %band
        exit(1)
    sdss_obj.mag = float(entries[index])

    if sdss_obj.mag > 0.:
        sdss_objs.append(sdss_obj)
        count += 1

sdss_ras = [o.ra for o in sdss_objs]
sdss_decs = [o.dec for o in sdss_objs]
print "read in %d SDSS truth objects" %count



# which pixels exist?  check file names.
pix_dict = dict()
dir_files = os.listdir(globals_dir)
n_globals = 0
for filename in dir_files:
    if re.match('gobjs_' + str(band) + '_nside' + str(nside) + '_p', filename) is not None:
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
plot_resids_b2_xs = []
plot_resids_b2_ys = []

    
for p in range(npix_wobjs):
    
    print "pixel {0}/{1}\r".format(p+1,npix_wobjs),
    
    pix = pix_wobjs[p]
    
    filename = 'gobjs_' + str(band) + '_nside' + str(nside) + '_p' + str(pix)
    file = open(os.path.join(globals_dir,filename), 'rb')
    global_objs = cPickle.load(file)
    file.close()
    
    global_ras = [o.ra-360. if o.ra>300. else o.ra for o in global_objs]
    global_decs = [o.dec for o in global_objs]
    
    inds1, inds2, dists = spherematch( global_ras, global_decs, sdss_ras, sdss_decs, tol=1./3600. )
    
    for i in range(len(inds1)):
        go = global_objs[inds1[i]]
        
        if go.ra > 300.:
            go.ra -= 360.
        ndet = len(go.objects)
        sdss_match = sdss_objs[inds2[i]]

        mags_before = np.zeros(ndet)
        mags_after = np.zeros(ndet)
        mag_errors2 = np.zeros(ndet)
        iids = np.zeros(ndet, dtype='int')
        ccds = np.zeros(ndet, dtype='int')
        xs = np.zeros(ndet)
        ys = np.zeros(ndet)
        d=0
        for d0 in range(ndet):
            mag_psf = go.objects[d]['mag_psf']
            magerr_psf = go.objects[d]['magerr_psf']
            xs[d]= go.objects[d]['x_image']
            ys[d] = go.objects[d]['y_image']
            image_id = go.objects[d]['image_id']
            if image_id == 1:
                continue
            ccd = go.objects[d]['ccd']-1
            spx = int(4*np.floor(ys[d]/512.) + np.floor(xs[d]/512.) + 32*(ccd+1))
            exposureid = go.objects[d]['exposureid']
            image_ccds[image_id] = ccd
            ccds[d] = ccd
            mags_before[d] = mag_psf + zp_phots[ccd]
            mags_after[d] = mag_psf + zp_phots[ccd]
            if use_spxzps:
                if spx not in spx_zps:
                    continue
                mags_after[d] += spx_zps[spx]
            if use_imgzps:
                if image_id not in zps:
                    continue
                mags_after[d] += zps[image_id] 
            if use_expzps:
                if exposureid not in exp_zps:
                    continue
                mags_after[d] += exp_zps[exposureid]

            mag_errors2[d] = magerr_psf*magerr_psf + 0.0001
            iids[d] = image_id
            total_magdiffsb.append(mags_before[d] - sdss_match.mag)
            total_magdiffsa.append(mags_after[d] - sdss_match.mag)
            try:
                image_ras[image_id] += go.ra
                image_decs[image_id] += go.dec
                image_diffbs[image_id] += mags_before[d] - sdss_match.mag
                image_diffas[image_id] += mags_after[d] - sdss_match.mag
                image_befores[image_id] += mags_before[d]
                image_afters[image_id] += mags_after[d]
                sdss_mags[image_id] += sdss_match.mag
                image_ns[image_id] += 1
            except KeyError:
                # the current image id is not in the image_ids dictionary yet
                image_ras[image_id] = go.ra
                image_decs[image_id] = go.dec
                image_diffbs[image_id] = mags_before[d] - sdss_match.mag
                image_diffas[image_id] = mags_after[d] - sdss_match.mag
                image_befores[image_id] = mags_before[d]
                image_afters[image_id] = mags_after[d]
                sdss_mags[image_id] = sdss_match.mag
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
            plot_resids_c2.append(mags_before[d]-sum_m_before)
            plot_resids_d2.append(mags_after[d]-sum_m_after)
            plot_resids_b2.append(mags_before[d] - sdss_match.mag)
            plot_resids_b2_xs.append(xs[d] + fp_xs[ccds[d]])
            plot_resids_b2_ys.append(ys[d] + fp_ys[ccds[d]])
            plot_resids_a2.append(mags_after[d] - sdss_match.mag)
            try:
                image_resids_b[iids[d]] += mags_before[d] - sdss_match.mag
                image_resids_a[iids[d]] += mags_after[d] - sdss_match.mag
            except KeyError:
                image_resids_b[iids[d]] = mags_before[d] - sdss_match.mag
                image_resids_a[iids[d]] = mags_after[d] - sdss_match.mag

        # calculate rms before and after
        rms_before = np.std(mags_before-sdss_match.mag)
        rms_after = np.std(mags_after-sdss_match.mag)
        # meandiff_before = sum_m_before - sdss_match.mag
        # meandiff_after = sum_m_after - sdss_match.mag
        # if meandiff_after < -3.:
        #     print "%f %f %f %f %d" %(sum_m_before, sum_m_after, zps[image_id], sdss_match.mag, image_id)
        sum_rms_before += rms_before
        sum_rms_after += rms_after
        
        rms_befores[hist_count] = rms_before
        rms_afters[hist_count] = rms_after
        # meandiff_befores[hist_count] = meandiff_before
        # meandiff_afters[hist_count] = meandiff_after
        hist_count += 1
        
        n_rms += 1

if n_rms > 0:
    sum_rms_before = np.sum(rms_befores)/n_rms
    sum_rms_after = np.sum(rms_afters)/n_rms
else:
    print "Um, n_rms=%d!!!!!!" %n_rms

for iid in image_ras.keys():
    image_ras[iid] /= image_ns[iid]
    image_decs[iid] /= image_ns[iid]
    image_diffas[iid] /= image_ns[iid]
    image_diffbs[iid] /= image_ns[iid]
    image_resids_b[iid] /= image_ns[iid]
    image_resids_a[iid] /= image_ns[iid]
    image_befores[iid] /= image_ns[iid]
    image_afters[iid] /= image_ns[iid]
    sdss_mags[iid] /= image_ns[iid]


print
print "RMS before: %f" %sum_rms_before
print "RMS after: %f" %sum_rms_after

# plots!
if hist_count > 100000:
    rms_befores = random.sample(rms_befores, 100000)
    rms_afters = random.sample(rms_afters, 100000)
    # meandiff_befores = random.sample(meandiff_befores, 100000)
    # meandiff_afters = random.sample(meandiff_afters, 100000)

if len(total_magdiffsa) > 100000:
        total_magdiffsb = random.sample(total_magdiffsb, 100000)
        total_magdiffsa = random.sample(total_magdiffsa, 100000)

fig = plt.figure()
title = "Total mag diff before calibration, med %e, rms %e" %(np.median(total_magdiffsb), np.std(total_magdiffsb))
print title
ax0 = fig.add_subplot(1,1,1, title=title)
ax0.set_yscale('log')
ax0.hist(total_magdiffsb, 100)
pp.savefig()

plt.clf()
fig = plt.figure()
title = "Total mag diff after calibration, med %e, rms %e" %(np.median(total_magdiffsa), np.std(total_magdiffsa))
print title
ax0 = fig.add_subplot(1,1,1, title=title)
ax0.set_yscale('log')
ax0.hist(total_magdiffsa, 100)
pp.savefig()

# histogram of RMS of individual objects before calibration
plt.clf()
title = "RMS of objects' mags, before cal (mean %e)" %sum_rms_before
print title
xlab = "Magnitudes (%s band)" %band
ylab = ""
ax0 = fig.add_subplot(1,1,1, xlabel=xlab, ylabel=ylab, title=title)
ax0.set_yscale('log')
ax0.hist(rms_befores, 50)
pp.savefig()

# histogram of RMS of individual objects after calibration
plt.clf()
title = "RMS of objects' mags, after cal (mean %e)" %sum_rms_after
print title
xlab = "Magnitudes (%s band)" %band
ax0 = fig.add_subplot(1,1,1, xlabel=xlab, ylabel=ylab, title=title)
ax0.set_yscale('log')
ax0.hist(rms_afters, 50)
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
ccd_array = np.zeros(n_ccds)
residb_array = np.zeros(n_ccds)
resida_array = np.zeros(n_ccds)
for key in image_ras:
    ra_array.append(image_ras[key])
    dec_array.append(image_decs[key])
    diffb_array.append(image_diffbs[key])
    diffa_array.append(image_diffas[key])
    n_array.append(image_ns[key])
    image_before_array.append(image_befores[key])
    sdss_array.append(sdss_mags[key])
    image_after_array.append(image_afters[key])
    ccd_array[image_ccds[key]] += 1
    residb_array[image_ccds[key]] += image_resids_b[key]
    resida_array[image_ccds[key]] += image_resids_a[key]
for ccd in range(63):
    if ccd_array[ccd]>0:
        residb_array[ccd] /= ccd_array[ccd]
        resida_array[ccd] /= ccd_array[ccd]
ra_array = np.array(ra_array)
dec_array = np.array(dec_array)

mdiffstdb = np.std(total_magdiffsb)
mdiffstda = np.std(total_magdiffsa)

plt.clf()
title = "Resid of mean from SDSS by ra and dec before cal"
ax0 = fig.add_subplot(1,1,1, xlabel=xlab, ylabel=ylab, title=title)
ylab = "DES - SDSS mag"
im0 = ax0.hexbin(image_before_array, diffb_array, bins='log', extent=(np.min(image_before_array), np.max(image_before_array), np.median(total_magdiffsb)-3*mdiffstdb, np.median(total_magdiffsb)+3*mdiffstdb)) 
fig.colorbar(im0)
pp.savefig()

plt.clf()
title = "Resid of mean from SDSS by ra and dec after cal"
ax0 = fig.add_subplot(1,1,1, xlabel=xlab, ylabel=ylab, title=title)
im0 = ax0.hexbin(image_after_array, diffa_array, bins='log', extent=(np.min(image_after_array), np.max(image_after_array), np.median(total_magdiffsa)-3*mdiffstda, np.median(total_magdiffsa)+3*mdiffstda)) 
fig.colorbar(im0)
pp.savefig()



plt.clf()
title = "Precam star locations"
xlab = "RA"
ylab = "Dec"
ax0 = fig.add_subplot(1,1,1, xlabel=xlab, ylabel=ylab, title=title)
im0 = ax0.hexbin(precam_ras, precam_decs, bins='log',extent=(np.min(ra_array),np.max(ra_array),np.min(dec_array),np.max(dec_array))) 
fig.colorbar(im0)
pp.savefig()


plt.clf()
title = "Resid of mean from SDSS by ra and dec before cal"
xlab = "RA"
ylab = "Dec"
ax0 = fig.add_subplot(111, xlabel=xlab, ylabel=ylab, title=title)
im0 = ax0.hexbin(ra_array, dec_array, diffb_array, vmin=np.median(total_magdiffsb)-3*mdiffstdb, vmax=np.median(total_magdiffsb)+3*mdiffstdb)
fig.colorbar(im0)
pp.savefig()

plt.clf()
title = "Resid of mean from SDSS by ra and dec after cal"
xlab = "RA"
ylab = "Dec"
ax0 = fig.add_subplot(111, xlabel=xlab, ylabel=ylab, title=title)
im0 = ax0.hexbin(ra_array, dec_array, diffa_array, vmin=np.median(total_magdiffsa)-3*mdiffstda, vmax=np.median(total_magdiffsa)+3*mdiffstda)
fig.colorbar(im0)
pp.savefig()


plt.clf()
fig = plt.figure()
title = "Residual of indiv meas from SDSS before cal"
ax0 = fig.add_subplot(1,1,1, title=title)
#mdiffstdb
im0 = ax0.hexbin(plot_resids_b2_xs, plot_resids_b2_ys, plot_resids_b2, gridsize=400, reduce_C_function = np.median, vmin=np.median(total_magdiffsb)-3*sum_rms_before, vmax=np.median(total_magdiffsb)+3*sum_rms_before) 
fig.colorbar(im0)
pp.savefig()

plt.clf()
fig = plt.figure()
title = "Residual of indiv meas from SDSS after cal"
ax0 = fig.add_subplot(1,1,1, title=title)
#mdiffstda
im0 = ax0.hexbin(plot_resids_b2_xs, plot_resids_b2_ys, plot_resids_a2, gridsize=400, reduce_C_function = np.median, vmin=np.median(total_magdiffsa)-3*sum_rms_after, vmax=np.median(total_magdiffsa)+3*sum_rms_after) 
fig.colorbar(im0)
pp.savefig()


plt.clf()
fig = plt.figure()
title = "Residual of indiv meas from DES mean before cal"
ax0 = fig.add_subplot(1,1,1, title=title)
im0 = ax0.hexbin(plot_resids_b2_xs, plot_resids_b2_ys, plot_resids_c2, gridsize=400, reduce_C_function = np.median, vmin=-3*sum_rms_before, vmax=3*sum_rms_before) 
fig.colorbar(im0)
pp.savefig()

plt.clf()
fig = plt.figure()
title = "Residual of indiv meas from DES mean after cal"
ax0 = fig.add_subplot(1,1,1, title=title)
im0 = ax0.hexbin(plot_resids_b2_xs, plot_resids_b2_ys, plot_resids_d2, gridsize=400, reduce_C_function = np.median, vmin=-3*sum_rms_after, vmax=3*sum_rms_after) 
fig.colorbar(im0)
pp.savefig()

pp.close()

