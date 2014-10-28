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
        print "Usage: match_global_objs.py config_filename"
        print "       To print out a default config file, run: match_global_objs.py default"
        exit(1)
    elif sys.argv[1] == 'default':
        # config = open( os.path.dirname(os.path.realpath(__file__)) + '/configs/config_sample', 'r' )
        # print "\n### DEFAULT EXAMPLE CONFIG FILE ###\n"
        # for line in config:
        #     sys.stdout.write(line)
        print "Sorry, default config not implemented yet."
        exit(0)
    
    magerr_sys2 = 0.0004
        
    # read in config
    config_filename = sys.argv[1]
    print "Calculating colors using config file %s!" %config_filename    
    config_file = open(config_filename)
    config = yaml.load(config_file)
    
    # what bands do we want to read in?
    bands = config['general']['filters']
    
    # these two are for the color (primary - secondary)
    primary_band = config['general']['primary_filter'] 
    secondary_band = config['general']['secondary_filter']
    
    # read in previous calibration info that we want to apply
    
    # read in the nominal ccd offsets (zp_phots), which will be our starting point
    all_bands = []
    all_bands.extend(bands)
    all_bands.append(primary_band)
    all_bands.append(secondary_band)
    zp_list = {}
    zp_phots = {}
    for band in all_bands:
        if band in zp_list.keys():
            continue
        zp_list[band] = []
        zp_phots[band] = dict()
        zp_phot_file = open( config['general']['zp_phot_filenames'][band], 'r' )
        zp_phots[band]['id_string'] = 'ccd'
        zp_phots[band]['operand'] = 1
        for line in zp_phot_file:
            entries = line.split()
            if entries[0][0] == '#':
                continue
            zp_phots[band][int(entries[0])] = -1.0*float(entries[4])
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
                    zp_dict[int(entries[0])] = {'zp': float(entries[1]), 'label': int(entries[2])}
                zp_list[band].append(zp_dict)
                print "Read in {0} zeropoints from {1} for id_string {2}, operand {3}".format(len(zp_dict.keys())-2, zp_filename, calibration['id_strings'][i], calibration['operands'][i])
        
        # add the standard zp if necessary
        std_zp_dict = None
        if config['general']['use_standards']:
            std_zp_dict = dict()
            std_zp_dict['operand'] = None
            std_zp_dict['id_string'] = None
            zp_file = open(config['general']['stdzp_outfilename'][band], 'r')
            for line in zp_file:
                entries = line.split()
                std_zp_dict[int(entries[0])] = float(entries[1])
            print 'Read in the zeropoint to standards'
    
    
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
            new_global_objects = {}
            secondary_file = open(os.path.join(globals_dir,secondary_fname), 'rb')
            global_objects[secondary_band] = cPickle.load(secondary_file)
            new_global_objects[secondary_band] = []
            secondary_file.close()
            primary_file = open(os.path.join(globals_dir,primary_fname), 'rb')
            global_objects[primary_band] = cPickle.load(primary_file)
            new_global_objects[primary_band] = []
            primary_file.close()
            
            print 'Pixel {0}, {1} and {2} objects in {3} and {4}'.format(pixel,len(global_objects[primary_band]),len(global_objects[secondary_band]),primary_band,secondary_band)
            
            # match the objects
            ras_1 = [o.ra-360. if o.ra>300. else o.ra for o in global_objects[primary_band]]
            decs_1 = [o.dec for o in global_objects[primary_band]]
            ras_2 = [o.ra-360. if o.ra>300. else o.ra for o in global_objects[secondary_band]]
            decs_2 = [o.dec for o in global_objects[secondary_band]]
            # print "{0} and {1} objects".format(len(ras_1), len(ras_2))
            if len(ras_1) == 0 or len(ras_2) == 0:
                continue
            inds1, inds2, dists = spherematch( ras_1, decs_1, ras_2, decs_2, tol=1./3600. )
            print "Found {0} matches in pixel {1}".format(len(inds1), pixel)
            
            # for each match, apply the zps and get mean calibrated mags
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
                        ccd = go.objects[d]['ccd']
                        if ccd in zp_phots[band]:
                            mag_after += zp_phots[band][ccd]
                        else:
                            badmag = True
                        for zps in zp_list[band]:
                            # print 'trying zps {0} {1} {2} {3}'.format(zps['id_string'],zps['operand'], go.objects[d][zps['id_string']], go.objects[d][zps['id_string']] in zps)
                            operand = 1
                            if zps['operand'] not in [1,None,'None']:
                                operand = go.objects[d][zps['operand']]
                            if zps['id_string'] == 'None':
                                mag_after += operand*zps[0]
                            else:
                                id_string = go.objects[d][zps['id_string']]
                                if id_string in zps:
                                    mag_after += operand*zps[id_string]['zp']
                                else:
                                    badmag = True
                        if badmag:
                            continue
                        if config['general']['use_standards']:
                            # make sure the standard zp exists for this label (as defined for the first calibration)
                            zps = zp_list[band][0]
                            id_string = go.objects[d][zps['id_string']]
                            label = zps[id_string]['label']
                            if label not in std_zp_dict:
                                badmag = True
                            else:
                                mag_after += std_zp_dict[label]
                        if badmag:
                            continue
                        mags.append(mag_after)
                        mag_errors2.append(go.objects[d]['magerr_psf']*go.objects[d]['magerr_psf'] + magerr_sys2)
                    if len(mags) == 0:
                        continue
                    invsigma_array = 1.0/np.array(mag_errors2)
                    sum_invsigma2 = invsigma_array.sum()
                    mean_mags[band] = (np.array(mags)*invsigma_array).sum() / sum_invsigma2
                
                # save calibrated color info
                if primary_band in mean_mags and secondary_band in mean_mags:
                    global_objects[primary_band][inds1[n]].color = mean_mags[primary_band]-mean_mags[secondary_band]
                    global_objects[secondary_band][inds2[n]].color = mean_mags[primary_band]-mean_mags[secondary_band]
                    new_global_objects[primary_band].append(global_objects[primary_band][inds1[n]])
                    new_global_objects[secondary_band].append(global_objects[secondary_band][inds2[n]])
                    
            # print "Finished calculating colors."
            
            # if there are no objects with g-r colors, don't save anything for this pixel.
            if len(new_global_objects[primary_band]) == 0:
                print "No objects with both bands."
                continue
            
            # ok, now we have all the objects with the desired color info.
            # are there other bands we want to make global objects for with this info?
            
            for band in bands:
                if band == primary_band or band == secondary_band:
                    # write a new globals file in the new directory.
                    outfile = open( os.path.join( config['general']['globals_out_dir'], 'gobjs_' + band + '_nside' + str(nside) + '_p' + str(pixel)), 'wb' )
                    cPickle.dump(new_global_objects[band], outfile, cPickle.HIGHEST_PROTOCOL)
                    outfile.close()
                    print "Pixel %d wrote %d global objects for band %s" %(pixel, len(new_global_objects[band]), band)
                else:
                    # need to match up this third band with our objects with known color
                    fileband = band
                    if fileband == 'y':
                        fileband = 'Y'
                    third_fname = 'gobjs_' + str(fileband) + '_nside' + str(nside) + '_p' + str(pixel)
                    if not os.path.exists( os.path.join(globals_dir,third_fname) ):
                        continue
                    third_file = open(os.path.join(globals_dir,third_fname), 'rb')
                    global_objects[band] = cPickle.load(third_file)
                    third_file.close()
                    
                    # match the objects
                    ras_1 = [o.ra-360. if o.ra>300. else o.ra for o in new_global_objects[primary_band]]
                    decs_1 = [o.dec for o in new_global_objects[primary_band]]
                    ras_2 = [o.ra-360. if o.ra>300. else o.ra for o in global_objects[band]]
                    decs_2 = [o.dec for o in global_objects[band]]
                    # print "{0} and {1} objects".format(len(ras_1), len(ras_2))
                    if len(ras_1) == 0 or len(ras_2) == 0:
                        continue
                    inds1, inds2, dists = spherematch( ras_1, decs_1, ras_2, decs_2, tol=1./3600. )
                    
                    # copy the color info between the matches
                    new_global_objects[band] = []
                    for n in range(len(inds2)):
                        global_objects[band][inds2[n]].color = new_global_objects[primary_band][inds1[n]].color
                        new_global_objects[band].append(global_objects[band][inds2[n]])
                    
                    # write a new globals file in the new directory.
                    outfile = open( os.path.join( config['general']['globals_out_dir'], 'gobjs_' + band + '_nside' + str(nside) + '_p' + str(pixel)), 'wb' )
                    cPickle.dump(new_global_objects[band], outfile, cPickle.HIGHEST_PROTOCOL)
                    outfile.close()
                    print "Pixel %d wrote %d global objects for band %s" %(pixel, len(new_global_objects[band]), band)
            


if __name__ == '__main__':
    main()
