import sys
import os
import yaml
import re
import cPickle
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pyspherematch import spherematch
from nebencal_utils import global_object

class multicolor_info(object):
    def __init__(self):
        self.ras = np.zeros(100000)
        self.decs = np.zeros(100000)
        self.pixels = np.zeros(100000)
        self.mags = np.zeros(100000)
        self.colors = np.zeros(100000)
        self.spxs = np.zeros(100000)
        self.count = 0
    def add(self, r, d, p, m, c, s):
        if self.count >= len(self.ras):
            self.ras = np.hstack((self.ras,np.zeros(self.count)))
            self.decs = np.hstack((self.decs,np.zeros(self.count)))
            self.pixels = np.hstack((self.pixels,np.zeros(self.count)))
            self.mags = np.hstack((self.mags,np.zeros(self.count)))
            self.colors = np.hstack((self.colors,np.zeros(self.count)))
            self.spxs = np.hstack((self.spxs,np.zeros(self.count)))
        self.ras[self.count] = r
        self.decs[self.count] = d
        self.pixels[self.count] = p
        self.mags[self.count] = m
        self.colors[self.count] = c
        self.spxs[self.count] = s
        self.count += 1
            
    def clean(self):
        self.ras = self.ras[0:self.count]
        self.decs = self.decs[0:self.count]
        self.pixels = self.pixels[0:self.count]
        self.mags = self.mags[0:self.count]
        self.colors = self.colors[0:self.count]
        self.spxs = self.spxs[0:self.count]

def return_spx(star, fp_xs, fp_ys):
    # spx = int(4*np.floor(star['y_image']/512.) + np.floor(star['x_image']/512.) + 32*(star['ccd']))
    fp_r = np.sqrt((star['x_image']+fp_xs[star['ccd']])*(star['x_image']+fp_xs[star['ccd']]) + (star['y_image']+fp_ys[star['ccd']])*(star['y_image']+fp_ys[star['ccd']]))
    # spx = int(np.floor(fp_r/1000.))
    spx = np.digitize([fp_r/148.], [0., 10., 30., 60., 100.])[0]-1  # william wester's binning
    return spx

def main():
    
    if len(sys.argv) != 2:
        print "Usage: spx_colorterms.py config_filename"
        print "       To print out a default config file, run: spx_colorterms.py default"
        exit(1)
    elif sys.argv[1] == 'default':
        # config = open( os.path.dirname(os.path.realpath(__file__)) + '/configs/config_sample', 'r' )
        # print "\n### DEFAULT EXAMPLE CONFIG FILE ###\n"
        # for line in config:
        #     sys.stdout.write(line)
        print "Sorry, default config not implemented yet."
        exit(0)
        
    # apply a color term, or just calculate the color and resave?
    apply_colorterm = False
    print 'Applying a color term correction? {0}'.format(apply_colorterm)
    
    magerr_sys2 = 0.0004
    spx_std = 0 # 1136 # in ccd35, but arbitrary.
    color_std = -0.9 # i-r # 0.2 #i-z # 0.0 # i-g
    min_color = -3
    max_color = 0.5
        
    # read in config
    config_filename = sys.argv[1]
    print "Calculating color terms using config file %s!" %config_filename    
    config_file = open(config_filename)
    config = yaml.load(config_file)
    
    # what bands do we want to read in?
    primary_band = config['general']['primary_filter']
    secondary_band = config['general']['secondary_filter']
    
    # read in previous calibration info that we want to apply
    
    # read in the nominal ccd offsets (zp_phots), which will be our starting point
    zp_list = { primary_band: [], secondary_band: [] }
    for band in (primary_band, secondary_band):
        zp_phots = dict()
        zp_phot_file = open( config['general']['zp_phot_filenames'][band], 'r' )
        zp_phots['id_string'] = 'ccd'
        zp_phots['operand'] = 1
        for line in zp_phot_file:
            entries = line.split()
            if entries[0][0] == '#':
                continue
            zp_phots[int(entries[0])] = -1.0*float(entries[4])
        zp_list[band].append(zp_phots)
        print "Read in {0} ccd offsets (zp_phots) for band {1}".format(len(zp_phots.keys())-2, band)
    
        # NOTE: assumes that the id_string is an integer!
        # read in the rest of the zps
        for calibration in config['calibrations'][band]:
            for i, zp_filename in enumerate(calibration['outfilenames']):
                zp_dict = dict()
                zp_dict['operand'] = calibration['operands'][i]
                zp_dict['id_string'] = calibration['id_strings'][i]
                zp_file = open( zp_filename, 'r' )
                for line in zp_file:
                    entries = line.split()
                    if entries[0][0] == '#':
                        continue
                    zp_dict[int(entries[0])] = float(entries[1])
                zp_list[band].append(zp_dict)
                print "Read in {0} zeropoints from {1} for id_string {2}, operand {3}".format(len(zp_dict.keys())-2, zp_filename, calibration['id_strings'][i], calibration['operands'][i])
    
    # read in the CCD positions in the focal plane
    posfile = open("/Users/bauer/surveys/DES/ccdPos-v2.par", 'r')
    posarray = posfile.readlines()
    fp_xs = []
    fp_ys = []
    # put in a dummy ccd=0
    fp_xs.append(0.)
    fp_ys.append(0.)
    for i in range(21,83,1):
        entries = posarray[i].split(" ")
        fp_xs.append(66.6667*(float(entries[4])-211.0605) - 1024) # in pixels
        fp_ys.append(66.6667*float(entries[5]) - 2048)
    print "parsed focal plane positions for %d ccds" %len(fp_xs)
    
    # read in global objects for each pixel seen in both bands
    globals_dir = config['general']['globals_dir']
    nside = config['general']['nside_file']
    mc_info = multicolor_info()
    
    # go through pixels in the primary band
    pix_dict = dict()
    dir_files = os.listdir(globals_dir)
    # print "Found {0} files in the global directory".format(len(dir_files))
    n_globals = dict()
    fileband_primary = primary_band
    if primary_band == 'y':
        fileband_primary = 'Y'
    for primary_fname in dir_files:
        # print "File {0}".format(primary_fname)
        if re.match('gobjs_' + str(fileband_primary) + '_nside' + str(nside) + '_p', primary_fname) is not None:
            pixel_match = re.search('p(\d+)', primary_fname)
            if pixel_match is None:
                print "Problem parsing pixel number"
            pixel = int(pixel_match.group()[1:])
            fileband = secondary_band
            if fileband == 'y':
                fileband = 'Y'
            
            # look for this pixel in the second band
            secondary_fname = 'gobjs_' + str(fileband) + '_nside' + str(nside) + '_p' + str(pixel)
            if not os.path.exists( os.path.join(globals_dir,secondary_fname) ):
                continue
            
            global_objects = {}
            secondary_file = open(os.path.join(globals_dir,secondary_fname), 'rb')
            global_objects[secondary_band] = cPickle.load(secondary_file)
            secondary_file.close()
            primary_file = open(os.path.join(globals_dir,primary_fname), 'rb')
            global_objects[primary_band] = cPickle.load(primary_file)
            primary_file.close()
            
            # match the objects
            ras_1 = [o.ra-360. if o.ra>300. else o.ra for o in global_objects[primary_band]]
            decs_1 = [o.dec for o in global_objects[primary_band]]
            ras_2 = [o.ra-360. if o.ra>300. else o.ra for o in global_objects[secondary_band]]
            decs_2 = [o.dec for o in global_objects[secondary_band]]
            # print "{0} and {1} objects".format(len(ras_1), len(ras_2))
            if len(ras_1) == 0 or len(ras_2) == 0:
                continue
            inds1, inds2, dists = spherematch( ras_1, decs_1, ras_2, decs_2, tol=1./3600. )
            # print "Found {0} matches in pixel {1}".format(len(inds1), pixel)
            
            # for each detection, apply the zps and get mean calibrated mags
            for n in range(len(inds1)):
                mean_mags = {}
                gos = []
                gos.append(global_objects[primary_band][inds1[n]])
                gos.append(global_objects[secondary_band][inds2[n]])
                for b, band in enumerate([primary_band, secondary_band]):
                    go = gos[b]
                    ndet = len(go.objects)
                    mags = {}
                    mag_errors2 = {}
                    d=0
                    for d in range(ndet):
                        spx = return_spx(go.objects[d], fp_xs, fp_ys)
                        if spx not in mags:
                            mags[spx] = []
                            mag_errors2[spx] = []
                        mag_after = go.objects[d]['mag_psf']
                        badmag = False
                        for zps in zp_list[band]:
                            operand = 1
                            if zps['operand'] not in [1,None,'None']:
                                operand = go.objects[d][zps['operand']]
                            id_string = go.objects[d][zps['id_string']]
                            if id_string in zps:
                                mag_after += operand*zps[id_string]
                            else:
                                badmag = True
                        if badmag:
                            continue
                        mags[spx].append(mag_after)
                        mag_errors2[spx].append(go.objects[d]['magerr_psf']*go.objects[d]['magerr_psf'] + magerr_sys2)
                    for spx in mags.keys():
                        if len(mags[spx]) == 0:
                            continue
                        invsigma_array = 1.0/np.array(mag_errors2[spx])
                        sum_invsigma2 = invsigma_array.sum()
                        if spx not in mean_mags:
                            mean_mags[spx] = {}
                        mean_mags[spx][band] = (np.array(mags[spx])*invsigma_array).sum() / sum_invsigma2
                
                if spx_std in mean_mags and primary_band in mean_mags[spx_std]:
                    for spx in mean_mags:
                        if spx == spx_std:
                            continue
                        if primary_band in mean_mags[spx] and secondary_band in mean_mags[spx]:
                            # print "adding {0} = {1} - {2} and {3} = {4} - {5}".format(mean_mags[spx][primary_band]-mean_mags[spx_std][primary_band], mean_mags[spx][primary_band], mean_mags[spx_std][primary_band],  mean_mags[spx][primary_band]-mean_mags[spx][secondary_band],  mean_mags[spx][primary_band], mean_mags[spx][secondary_band])
                            mc_info.add(go.ra, go.dec, pixel, mean_mags[spx_std][primary_band]-mean_mags[spx][primary_band], mean_mags[spx][primary_band]-mean_mags[spx][secondary_band], spx)
        print "  {0} entries in mc_info\r".format(mc_info.count),
    print "  {0} entries in mc_info".format(mc_info.count)
    
    mc_info.clean()
    
    # fit the color term.  plot the result!
    # g-g_spx0 = A(color - color_0)
    plot_file = 'spx_colorterms.pdf'
    pp = PdfPages(plot_file)
    spxs = np.unique(mc_info.spxs)
    nfigs = 0
    spx_coeffs = {}
    for s, spx in enumerate(spxs):
        dmags = mc_info.mags[(mc_info.spxs == spx) & (mc_info.colors>min_color) & (mc_info.colors<max_color)]
        colors = mc_info.colors[(mc_info.spxs == spx) & (mc_info.colors>min_color) & (mc_info.colors<max_color)]
        coeffs = np.polyfit(colors, dmags, 1)
        n_clip = 1
        n_iter = 0
        while n_clip:
            n_clip = len(dmags)
            resids = coeffs[1]+colors*coeffs[0] - dmags
            std_dev = np.std(resids)
            dmags = dmags[np.abs(resids) < 3*std_dev]
            colors = colors[np.abs(resids) < 3*std_dev]
            n_clip -= len(dmags)
            n_iter += 1
            coeffs = np.polyfit(colors, dmags, 1)
        spx_coeffs[spx] = coeffs
        
        # JUST KIDDING, USE DECAL COLOR TERMS
        spx_coeffs = {}
        # spx_coeffs = { 1.:[-0.010514, -0.00083], 2.:[-0.018073, -0.00118], 3.:[-0.027567, -0.00173] } # bigmacs
        spx_coeffs = { 1.:[-0.00935957, -0.00075208], 2.: [-0.01648467, -0.00108544], 3.:[-0.02511064, -0.00148729] } # pickles 
        coeffs = spx_coeffs[spx]
        
        print "  SPX {0} / {1}\r".format(spx, len(spxs)),
        fig = plt.figure()
        title = 'SPX {0}'.format(spx)
        xlab = 'SPX color'
        ylab = 'SPX - standard mag'
        ax0 = fig.add_subplot(1,1,1, title=title, xlabel=xlab, ylabel=ylab)
        im0 = ax0.hexbin(colors, dmags, bins='log') 
        fig.colorbar(im0)
        plt.plot( [min_color,max_color],[coeffs[1]+min_color*coeffs[0], coeffs[1]+max_color*coeffs[0]] )
        pp.savefig()
        nfigs += 1
    fig = plt.figure()
    title = 'Best-fit lines'
    ax0 = fig.add_subplot(1,1,1, title=title, xlabel=xlab, ylabel=ylab)
    for spx in spx_coeffs.keys():
        print spx_coeffs[spx]
        ax0.plot([min_color,max_color], [spx_coeffs[spx][1]+min_color*spx_coeffs[spx][0], spx_coeffs[spx][1]+max_color*spx_coeffs[spx][0]], label=str(spx))
    ax0.legend()
    pp.savefig()
    
    pp.close()
    
    print "Starting correction"
    
    # go through pixels again, calculate the corrected mags, and write out the matches to a new set of primary global objects
    # need to calculate the color of the objs.  (this would be better if we iterate.)
    for primary_fname in dir_files:
        
        if re.match('gobjs_' + str(fileband_primary) + '_nside' + str(nside) + '_p', primary_fname) is not None:
            pixel_match = re.search('p(\d+)', primary_fname)
            if pixel_match is None:
                print "Problem parsing pixel number"
            pixel = int(pixel_match.group()[1:])
            # look for this pixel in the second band
            secondary_fname = 'gobjs_' + str(fileband) + '_nside' + str(nside) + '_p' + str(pixel)
            if not os.path.exists( os.path.join(globals_dir,secondary_fname) ):
                continue
            
            global_objects = {}
            secondary_file = open(os.path.join(globals_dir,secondary_fname), 'rb')
            global_objects[secondary_band] = cPickle.load(secondary_file)
            secondary_file.close()
            primary_file = open(os.path.join(globals_dir,primary_fname), 'rb')
            global_objects[primary_band] = cPickle.load(primary_file)
            primary_file.close()
            
            # match the objects
            ras_1 = [o.ra-360. if o.ra>300. else o.ra for o in global_objects[primary_band]]
            decs_1 = [o.dec for o in global_objects[primary_band]]
            ras_2 = [o.ra-360. if o.ra>300. else o.ra for o in global_objects[secondary_band]]
            decs_2 = [o.dec for o in global_objects[secondary_band]]
            # print "{0} and {1} objects".format(len(ras_1), len(ras_2))
            if len(ras_1) == 0 or len(ras_2) == 0:
                continue
            inds1, inds2, dists = spherematch( ras_1, decs_1, ras_2, decs_2, tol=1./3600. )
            # print "Found {0} matches in pixel {1}".format(len(inds1), pixel)
            
            # for each detection, apply the zps and get mean calibrated mags
            new_gos = []
            for n in range(len(inds1)):
                mean_mags = {}
                gos = []
                gos.append(global_objects[primary_band][inds1[n]])
                gos.append(global_objects[secondary_band][inds2[n]])
                for b, band in enumerate([primary_band, secondary_band]):
                    go = gos[b]
                    ndet = len(go.objects)
                    mags = []
                    mag_errors2 = []
                    d=0
                    for d in range(ndet):
                        mag_after = go.objects[d]['mag_psf']
                        badmag = False
                        for zps in zp_list[band]:
                            operand = 1
                            if zps['operand'] not in [1,None,'None']:
                                operand = go.objects[d][zps['operand']]
                            id_string = go.objects[d][zps['id_string']]
                            if id_string in zps:
                                mag_after += operand*zps[id_string]
                            else:
                                badmag = True
                        if badmag:
                            continue
                        mags.append(mag_after)
                        mag_errors2.append(go.objects[d]['magerr_psf']*go.objects[d]['magerr_psf'] + magerr_sys2)
                    if len(mags) == 0:
                        continue
                    invsigma_array = 1.0/np.array(mag_errors2)
                    sum_invsigma2 = invsigma_array.sum()
                    mean_mags[band] = (np.array(mags)*invsigma_array).sum() / sum_invsigma2
                
                go = global_objects[primary_band][inds1[n]]
                ndet = len(go.objects)
                new_go = global_object()
                new_go.ra = go.ra
                new_go.dec = go.dec
                for d in range(ndet):
                    obj = go.objects[d]
                    spx = return_spx(obj, fp_xs, fp_ys)
                    if primary_band in mean_mags and secondary_band in mean_mags:
                        mag_after = obj['mag_psf']
                        badmag = False
                        for zps in zp_list[primary_band]:
                            operand = 1
                            if zps['operand'] not in [1,None,'None']:
                                operand = obj[zps['operand']]
                            id_string = obj[zps['id_string']]
                            if id_string in zps:
                                mag_after += operand*zps[id_string]
                            else:
                                badmag = True
                        if badmag:
                            continue
                        
                        mag_std = mag_after + (mean_mags[primary_band]-mean_mags[secondary_band])*coeffs[0] + coeffs[1]
                        mag_new = mag_std - color_std*coeffs[0] + coeffs[1]
                        if apply_colorterm:
                            obj['mag_psf'] += (mag_new-mag_after) # just change it by the ZP, or else i have to take zp_phots out of default calibration.
                        obj['color'] = mean_mags[primary_band]-mean_mags[secondary_band]
                        new_go.objects.append(obj)
                if primary_band in mean_mags and secondary_band in mean_mags and len(new_go.objects)>0:
                    go.color = mean_mags[primary_band]-mean_mags[secondary_band]
                    new_gos.append(new_go)
                            
            # write a new globals file in the new directory.
            outfile = open( os.path.join( config['general']['globals_out_dir'], 'gobjs_' + primary_band + '_nside' + str(nside) + '_p' + str(pixel)), 'wb' )
            cPickle.dump(new_gos, outfile, cPickle.HIGHEST_PROTOCOL)
            outfile.close()
            print "Pixel %d wrote %d global objects" %(pixel, len(new_gos))
                    

    

if __name__ == '__main__':
    main()
