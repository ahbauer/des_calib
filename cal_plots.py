import sys
import os
import math
import re
import subprocess
# from operator import itemgetter
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# from scipy.stats import scoreatpercentile

band = 'z'
globals_dir = '/Users/bauer/surveys/DES/y1p1/equatorial'
nside = 32
n_ccds = 63
spxsize = 256
n_xspx = 8
n_yspx = 16

plot_file = 'nebencal_plots_' + band + '.pdf'
pp = PdfPages(plot_file)

# read in the zero point solutions
zps = dict()
zp_array = []
zp_file = globals_dir + '/' + str(band) + '_results/nebencal_zps_' + str(band)
file = open(zp_file, 'r')
filelines = file.readlines()
for line in filelines:
    entries = line.split()
    zps[int(entries[0])] = float(entries[1])
    zp_array.append(float(entries[1]))
print "read in %d zps" %len(zps.keys())

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
fp_ys.append(fp_xs[-1])
fp_xs.append(fp_ys[-1])
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
image_ns = dict()
image_ccds = dict()
image_resids_b = dict()
image_resids_a = dict()
sum_rms_before = 0.
sum_rms_after = 0.
n_rms = 0
rms_befores = np.zeros(n_hist)
rms_afters = np.zeros(n_hist)
rand_sample = random.sample(range(n_globals), n_hist)
hist_count = 0

plot_resids_b = []
plot_resids_a = []
plot_resid_ns = []
for ccd in range(n_ccds):
    plot_resids_b.append(np.zeros((n_yspx,n_xspx)))
    plot_resids_a.append(np.zeros((n_yspx,n_xspx)))
    plot_resid_ns.append(np.zeros((n_yspx,n_xspx)))
    
for p in range(npix_wobjs):
    
    print "pixel {0}/{1}\r".format(p+1,npix_wobjs),
    
    pix = pix_wobjs[p]
    
    filename = 'gobjs_' + str(band) + '_nside' + str(nside) + '_p' + str(pix)
    file = open(os.path.join(globals_dir,filename), 'r')
    filelines = file.readlines()
    for line in filelines:
        # print "reading a line"
        entries = line.split()
        ra = float(entries[0])
        dec = float(entries[1])
        ndet = int((len(entries)-2)/6)
        #max_nimgs = max_matrix_size
        mags_before = np.zeros(ndet)
        mags_after = np.zeros(ndet)
        mag_errors = np.zeros(ndet)
        iids = np.zeros(ndet, dtype='int')
        ccds = np.zeros(ndet, dtype='int')
        xs = np.zeros(ndet)
        ys = np.zeros(ndet)
        index = 2
        for d in range(ndet):
            # star = dict()
            # star['ra'] = ra
            # star['dec'] = dec
            mag_psf = float(entries[index])
            magerr_psf = float(entries[index+1])
            xs[d]= float(entries[index+2])
            ys[d] = float(entries[index+3])
            image_id = int(entries[index+4])
            ccd = int(entries[index+5])
            image_ccds[image_id] = ccd
            ccds[d] = ccd
            mags_before[d] = mag_psf
            mags_after[d] = mag_psf + zps[image_id]
            mag_errors[d] = np.sqrt(magerr_psf*magerr_psf + 0.0001)
            iids[d] = image_id
            try:
                image_ras[image_id] += ra
                image_decs[image_id] += dec
                image_ns[image_id] += 1
            except KeyError:
                # the current image id is not in the image_ids dictionary yet
                image_ras[image_id] = ra
                image_decs[image_id] = dec
                image_ns[image_id] = 1
            index += 6
        invsigma_array = 1.0/np.square(mag_errors)
        sum_invsigma2 = invsigma_array.sum()
        sum_m_before = (mags_before*invsigma_array).sum() / sum_invsigma2
        sum_m_after = (mags_after*invsigma_array).sum() / sum_invsigma2
        
        for d in range(ndet):
            superpix_x = int(xs[d]/spxsize)
            superpix_y = int(ys[d]/spxsize)
            try:
                image_resids_b[iids[d]] += mags_before[d] - sum_m_before
                image_resids_a[iids[d]] += mags_after[d] - sum_m_after
                plot_resids_b[ccds[d]][superpix_y,superpix_x] += mags_before[d] - sum_m_before
                plot_resids_a[ccds[d]][superpix_y,superpix_x] += mags_after[d] - sum_m_after
                plot_resid_ns[ccds[d]][superpix_y,superpix_x] += 1
            except KeyError:
                image_resids_b[iids[d]] = mags_before[d] - sum_m_before
                image_resids_a[iids[d]] = mags_after[d] - sum_m_after
                plot_resids_b[ccds[d]][superpix_y,superpix_x] = mags_before[d] - sum_m_before
                plot_resids_a[ccds[d]][superpix_y,superpix_x] = mags_after[d] - sum_m_after
                plot_resid_ns[ccds[d]][superpix_y,superpix_x] = 1

        # calculate rms before and after
        rms_before = np.std(mags_before-sum_m_before)
        rms_after = np.std(mags_after-sum_m_after)
        # sum_rms_before += rms_before
        # sum_rms_after += rms_after
        
        # if n_rms in rand_sample:
        rms_befores[hist_count] = rms_before
        rms_afters[hist_count] = rms_after
        hist_count += 1
        
        n_rms += 1
    
sum_rms_before = np.sum(rms_befores)/n_rms
sum_rms_after = np.sum(rms_afters)/n_rms

for iid in image_ras.keys():
    image_ras[iid] /= image_ns[iid]
    image_decs[iid] /= image_ns[iid]
    image_resids_b[iid] /= image_ns[iid]
    image_resids_a[iid] /= image_ns[iid]

for ccd in range(n_ccds):
    for x in range(n_xspx):
        for y in range(n_yspx):
            if plot_resid_ns[ccd][y,x] > 0:
                plot_resids_b[ccd][y,x] /= plot_resid_ns[ccd][y,x]
                plot_resids_a[ccd][y,x] /= plot_resid_ns[ccd][y,x]

print
print "RMS before: %f" %sum_rms_before
print "RMS after: %f" %sum_rms_after

# plots!
if hist_count > 100000:
    rms_befores = random.sample(rms_befores, 100000)
    rms_afters = random.sample(rms_afters, 100000)

fig = plt.figure()
title = "Nebencal ZPs, filter %s" %band
xlab = "Magnitudes"
ylab = ""
ax0 = fig.add_subplot(1,1,1, xlabel=xlab, title=title)
ax0.set_yscale('log')
# histogram of all zps
# plt.clf()
ax0.hist(zp_array, 50)
pp.savefig()

# histogram of RMS of individual objects before calibration
plt.clf()
title = "RMS of objects' mags, before cal (mean %e)" %sum_rms_before
ax0 = fig.add_subplot(1,1,1, xlabel=xlab, title=title)
ax0.set_yscale('log')
xlab = "Magnitudes"
ylab = ""
ax0.hist(rms_befores, 50)
pp.savefig()

# histogram of RMS of individual objects after calibration
plt.clf()
title = "RMS of objects' mags, after cal (mean %e)" %sum_rms_after
xlab = "Magnitudes"
ylab = ""
ax0 = fig.add_subplot(1,1,1, xlabel=xlab, title=title)
ax0.set_yscale('log')
ax0.hist(rms_afters, 50)
pp.savefig()

# zp by ra and dec
ra_array = []
dec_array = []
zp_array = []
zp_array2 = np.zeros(n_ccds)
ccd_array = np.zeros(n_ccds)
residb_array = np.zeros(n_ccds)
resida_array = np.zeros(n_ccds)
for key in image_ras:
    if zps[key] > -1.0 and zps[key] < 1.0:
        ra_array.append(image_ras[key])
        dec_array.append(image_decs[key])
        zp_array.append(zps[key])
        ccd_array[image_ccds[key]] += 1
        zp_array2[image_ccds[key]] += zps[key]
        residb_array[image_ccds[key]] += image_resids_b[key]
        resida_array[image_ccds[key]] += image_resids_a[key]
for ccd in range(63):
    if ccd_array[ccd]>0:
        residb_array[ccd] /= ccd_array[ccd]
        resida_array[ccd] /= ccd_array[ccd]
        zp_array2[ccd] /= ccd_array[ccd]
plt.clf()
title = "ZP by ra and dec"
xlab = "RA"
ylab = "Dec"
ax3D = fig.add_subplot(111, projection='3d', xlabel=xlab, ylabel=ylab, title=title)
ax3D.view_init(90, 0)
ax3D.scatter(ra_array, dec_array, zp_array, c=zp_array)
pp.savefig()


# zp by ccd number (arranged nicely??)
plt.clf()
title = "ZP by CCD"
ylab = "Magnitudes"
xlab = "CCD ID"
ax0 = fig.add_subplot(1,1,1, xlabel=xlab, title=title)
ax0.scatter(range(n_ccds), zp_array2)
pp.savefig()

plt.clf()
title = "ZP by CCD"
xlab = ""
ylab = ""
ax3D = fig.add_subplot(111, projection='3d', xlabel=xlab, ylabel=ylab, title=title)
ax3D.view_init(90, 0)
ax3D.scatter(fp_xs, fp_ys, zp_array2, c=zp_array2, marker='s', cmap='spectral')
pp.savefig()

# residual from mean vs chip
plt.clf()
title = "Residual by CCD before cal"
ylab = "Magnitudes"
xlab = "CCD ID"
ax0 = fig.add_subplot(1,1,1, xlabel=xlab, ylabel=ylab, title=title)
ax0.scatter(range(n_ccds), residb_array)
pp.savefig()

vertices = [(-1024,-2048),(-1024,2048),(1024,2048),(1024,-2048)]

plt.clf()
title = "Residual by CCD before cal"
xlab = ""
ylab = ""
ax3D = fig.add_subplot(111, projection='3d', xlabel=xlab, ylabel=ylab, title=title)
ax3D.view_init(90, 0)
ax3D.scatter(fp_xs, fp_ys, residb_array, c=residb_array, marker='s', cmap='spectral')
pp.savefig()

# residual from mean vs x,y on chip
# 3D plots of ccd residuals

# spx = []
# spy = []
# resids = []
# for x in range(n_xspx):
#     for y in range(n_yspx):
#         spx.append(x)
#         spy.append(y)
#         resids.append(plot_resids_b[ccd][y,x])
fig = plt.figure()
for ccd in range(n_ccds):
    # if ccd%8 == 0:
    #     if ccd>0:
    #         pp.savefig()
    #     plt.clf()
    #     fig = plt.figure()
    #     loc = 1
    # ax1 = fig.add_subplot(2,4,loc)
    # print "trying %d %d" %(fp_yindices[ccd],fp_xindices[ccd]+fp_xoffsets[fp_yindices[ccd]])
    ax1 = plt.subplot2grid((12,14), (fp_yindices[ccd],fp_xindices[ccd]+fp_xoffsets[fp_yindices[ccd]]), colspan=2)
    # ax3D = fig.add_subplot(111, projection='3d')
    # ax3D.view_init(90, 0)
    # ax3D.scatter(spx, spy, resids, c=resids)
    ax1.axis('off')
    ax1.imshow(np.transpose(plot_resids_b[ccd]),interpolation='nearest')
    # xlabel = "x superpix"
    # ylabel = "y superpix"
    # title = "ccd %d" %ccd
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.title(title)
    # loc += 1
pp.savefig()

plt.clf()
title = "Residual by CCD after cal"
ylab = "Magnitudes"
xlab = "CCD ID"
ax0 = fig.add_subplot(1,1,1, xlabel=xlab, ylabel=ylab, title=title)
ax0.scatter(range(n_ccds), resida_array)
pp.savefig()

plt.clf()
title = "Residual by CCD after cal"
xlab = ""
ylab = ""
ax3D = fig.add_subplot(111, projection='3d', xlabel=xlab, ylabel=ylab, title=title)
ax3D.view_init(90, 0)
ax3D.scatter(fp_xs, fp_ys, resida_array, c=resida_array, marker='s', cmap='spectral')
pp.savefig()

# residual from mean vs x,y on chip
# 3D plots of ccd residuals
spx = []
spy = []
resids = []
for x in range(n_xspx):
    for y in range(n_yspx):
        spx.append(x)
        spy.append(y)
        resids.append(plot_resids_a[ccd][y,x])
for ccd in range(n_ccds):
    if ccd%8 == 0:
        if ccd>0:
            pp.savefig()
        plt.clf()
        fig = plt.figure()
        loc = 1
    ax1 = fig.add_subplot(2,4,loc)
    # ax3D = fig.add_subplot(111, projection='3d')
    # ax3D.view_init(90, 0)
    # ax3D.scatter(spx, spy, resids, c=resids)
    ax1.imshow(plot_resids_a[ccd],interpolation='nearest')
    # xlabel = "x superpix"
    # ylabel = "y superpix"
    title = "ccd %d" %ccd
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    plt.title(title)
    loc += 1
pp.savefig()

fig = plt.figure()
for ccd in range(n_ccds):
    # print "trying %d %d" %(fp_yindices[ccd],fp_xindices[ccd]+fp_xoffsets[fp_yindices[ccd]])
    ax1 = plt.subplot2grid((12,14), (fp_yindices[ccd],fp_xindices[ccd]+fp_xoffsets[fp_yindices[ccd]]), colspan=2)
    ax1.axis('off')
    ax1.imshow(np.transpose(plot_resids_a[ccd]),interpolation='nearest')
pp.savefig()
pp.close()
