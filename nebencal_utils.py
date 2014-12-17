""" 
nebencal_utils:

functions used in both nebencal and nebencal_plots

AHB 4/2014
"""

import os
import sys
import numpy as np
import cPickle
import healpy
colorterm_dir = '/Users/bauer/software/python/des/des_calib/ting/color-term'
sys.path.append(colorterm_dir)
import color_term_cal

class global_object(object):
    def __init__(self):
        self.ra = None
        self.dec = None
        self.color = None
        self.objects = []

# for now.....
good_quality_magerr = 0.02
def good_quality(star):
    if star['magerr_psf'] < good_quality_magerr:
        return True
    return False

def standard_quality(star):
    exptimes = {'g':90., 'r':90., 'i':90., 'z':90., 'y':45.}
    if star['exptime'] == exptimes[star['band']] and star['image_id'] != 1 and 5.0 < star['mag_psf'] < 30.0 and 0.0 < star['magerr_psf'] < good_quality_magerr and star['x_image']>100. and star['x_image']<1900. and star['y_image']>100. and star['y_image']<3950. and star['cloud_nomad']<0.2 and star['gskyphot'] == 1:
        return True
    else:
        return False

def ok_quality(star):
    if star['magerr_psf'] < good_quality_magerr*2.0 and star['x_image']>100. and star['x_image']<1900. and star['y_image']>100. and star['y_image']<3950.:
        return True
    return False

def read_ccdpos():
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
    fp_yindices.append(fp_yindices[-1])
    fp_xindices.append(fp_xindices[-1])
    fp_xoffsets = [4,3,2,1,1,0,0,1,1,2,3,4]
    fp_xindices = 2*np.array(fp_xindices)
    fp_yindices = np.array(fp_yindices)
    # print "parsed focal plane positions for %d ccds" %len(fp_xs)
    return fp_xs, fp_ys

def apply_zps(mag, star, zp_list, bad_idvals):
    for zps in zp_list:
        operand = 1
        id_string = None
        if zps['operand'] not in [1,None,'None']:
            operand = star[zps['operand']]
        if zps['id_string'] is None:
            mag += operand*zps[0]
        else:
            id_string = star[zps['id_string']]
            if id_string in zps:
                mag += operand*zps[id_string]['zp']
            else:
                bad_idvals[zps['id_string']][id_string] = True
                return 0.0
    return mag

def apply_color_term(mag, go, index, fp_xs, fp_ys, mag_classes):
    star = go.objects[index]
    if star['exposureid'] not in mag_classes:
        pwv = color_term_cal.get_pwv(star['mjd'])
        mag_classes[star['exposureid']] = color_term_cal.color_term_correction(band,pwv,star['airmass'])
    fp_x = star['x_image'] + fp_xs[star['ccd']]
    fp_y = star['x_image'] + fp_ys[star['ccd']]
    fp_r = np.sqrt(fp_x*fp_x + fp_y*fp_y)/14880.1 # the largest fp_r i saw in an example run
    if go.color is None:
        print 'Warning, assigning standard color'
        go.color = 1.3 
    color_term = color_term_cal.fit_color_position(mag_classes[star['exposureid']],go.color,fp_r)
    mag += color_term
    return mag

def apply_stdcal(mag, star, zp_list, std_zp_dict):
    # make sure the standard zp exists for this label (as defined for the first calibration)
    zps = zp_list[0]
    id_string = star[zps['id_string']]
    label = zps[id_string]['label']
    if label not in std_zp_dict:
        return 0.0
    else:
        mag += std_zp_dict[label]
    return mag

def read_precam( precam_stars, precam_map, filename, band ):
    precam_file = open( filename, 'r' )
    # read in the precam standards and make an index
    count = 0
    for line in precam_file:
        entries = line.split(" ")
        if( entries[0][0] == '#' ):
            continue
        star = dict()
        star['ra'] = float(entries[1])
        star['dec'] = float(entries[2])
        star['mag_psf'] = float(entries[3])
        star['magerr_psf'] = 0.02 #float(entries[6])
        star['x_image'] = 1.
        star['y_image'] = 1.
        star['fp_r'] = 1.
        star['secz'] = 1.
        star['ccd'] = 0
        star['image_id'] = 1
        star['exposureid'] = 1
        star['superpix'] = -1
        star['airmass'] = 0
        star['color'] = 1.e-9
        star['cloud_nomad'] = 1.0
        star['band'] = band
        star['matched'] = False
        precam_stars.append(star)
        precam_map.insert( count, (star['ra'],star['dec'],star['ra'],star['dec']) )
        count += 1
    print "Read in %d PreCam standards" %count


def read_sdss( sdss_stars, sdss_map, filename, band ):
    sdssfile = open(filename, 'r')
    count=0
    for line in sdssfile:
        entries = line.split(",")
        # header!
        if entries[0] == 'id':
            continue
        sdss_obj = dict()
        sdss_obj['ra'] = float(entries[2])
        sdss_obj['dec'] = float(entries[3])
        if band == 'u':
            sdss_obj['mag_psf'] = float(entries[4])
        elif band == 'g':
            sdss_obj['mag_psf'] = float(entries[5])
        elif band == 'r':
            sdss_obj['mag_psf'] = float(entries[6])
        elif band == 'i':
            sdss_obj['mag_psf'] = float(entries[7])
        elif band == 'z':
            sdss_obj['mag_psf'] = float(entries[8])
        elif band == 'y':
            sdss_obj['mag_psf'] = float(entries[9])
        else:
            print "Um, while parsing SDSS objects, band = %s" %band
            exit(1)
        sdss_obj['band'] = band
        sdss_obj['magerr_psf'] = 0.02
        sdss_obj['x_image'] = 1.
        sdss_obj['y_image'] = 1.
        sdss_obj['fp_r'] = 1.
        sdss_obj['secz'] = 1.
        sdss_obj['ccd'] = 0
        sdss_obj['image_id'] = 1
        sdss_obj['exposureid'] = 1
        sdss_obj['superpix'] = -1
        sdss_obj['airmass'] = 0
        sdss_obj['color'] = 1.e-9
        sdss_obj['cloud_nomad'] = 1.0
        sdss_obj['matched'] = False
        if sdss_obj['mag_psf'] > 0.:
            sdss_stars.append(sdss_obj)
            sdss_map.insert( count, (sdss_obj['ra'],sdss_obj['dec'],sdss_obj['ra'],sdss_obj['dec']) )
            count += 1
    print "Read in %d SDSS standards" %count

def read_betoule( sdss_stars, sdss_map, filename, band ):
    sdssfile = open(filename, 'r')
    count=0
    for line in sdssfile:
        entries = line.split(" ")
        sdss_obj = dict()
        sdss_obj['ra'] = float(entries[0])
        sdss_obj['dec'] = float(entries[1])
        sdss_obj['mag_psf'] = float(entries[2])
        sdss_obj['magerr_psf'] = float(entries[3])
        
        sdss_obj['band'] = band
        sdss_obj['x_image'] = 1.
        sdss_obj['y_image'] = 1.
        sdss_obj['fp_r'] = 1.
        sdss_obj['secz'] = 1.
        sdss_obj['ccd'] = 0
        sdss_obj['image_id'] = 1
        sdss_obj['exposureid'] = 1
        sdss_obj['superpix'] = -1
        sdss_obj['airmass'] = 0
        sdss_obj['color'] = 1.e-9
        sdss_obj['cloud_nomad'] = 1.0
        sdss_obj['matched'] = False
        if sdss_obj['mag_psf'] > 0.:
            sdss_stars.append(sdss_obj)
            sdss_map.insert( count, (sdss_obj['ra'],sdss_obj['dec'],sdss_obj['ra'],sdss_obj['dec']) )
            count += 1
    print "Read in %d Betoule standards" %count

def read_tertiaries( sdss_stars, sdss_map, filename, band ):
    sdssfile = open(filename, 'r')
    count=0
    next(sdssfile)
    for line in sdssfile:
        entries = line.split(",")
        # header!
        if entries[0] == 'id':
            continue
        sdss_obj = dict()
        sdss_obj['ra'] = float(entries[1])
        sdss_obj['dec'] = float(entries[2])
        if band == 'u':
            sdss_obj['mag_psf'] = 999.
        elif band == 'g':
            sdss_obj['mag_psf'] = float(entries[3])
        elif band == 'r':
            sdss_obj['mag_psf'] = float(entries[5])
        elif band == 'i':
            sdss_obj['mag_psf'] = float(entries[7])
        elif band == 'z':
            sdss_obj['mag_psf'] = float(entries[9])
        elif band == 'y':
            sdss_obj['mag_psf'] = float(entries[11])
        else:
            print "Um, while parsing Tertiary objects, band = %s" %band
            exit(1)
        sdss_obj['band'] = band
        sdss_obj['magerr_psf'] = 0.02
        sdss_obj['x_image'] = 1.
        sdss_obj['y_image'] = 1.
        sdss_obj['fp_r'] = 1.
        sdss_obj['secz'] = 1.
        sdss_obj['ccd'] = 0
        sdss_obj['image_id'] = 1
        sdss_obj['exposureid'] = 1
        sdss_obj['superpix'] = -1
        sdss_obj['airmass'] = 0
        sdss_obj['color'] = 1.e-9
        sdss_obj['cloud_nomad'] = 1.0
        sdss_obj['matched'] = False
        if sdss_obj['mag_psf'] > 0.:
            sdss_stars.append(sdss_obj)
            sdss_map.insert( count, (sdss_obj['ra'],sdss_obj['dec'],sdss_obj['ra'],sdss_obj['dec']) )
            count += 1
    print "Read in %d SDSS standards" %count

def read_global_objs(band, globals_dir, nside, nside_file, pix, verbose=True):

    global_objs_list = []

    # if the pixelization resolution is the same as the file convention, then this is easy.
    if nside == nside_file:
        filename = 'gobjs_' + str(band) + '_nside' + str(nside) + '_p' + str(pix)
        # if there's no data for this pixel, forget it: nothing to calibrate.
        if not os.path.isfile(os.path.join(globals_dir,filename)):
            return None
        # print >> sys.stderr, "Reading in file-resolution pixel %d" %pix
        file = open(os.path.join(globals_dir,filename), 'rb')
        global_objs_list = cPickle.load(file)
        file.close()

        # if there weren't any objects for this low-res pixel, then we're done.
        if len(global_objs_list) == 0:
            return None

        # now include the 4 nearest neighboring pixels.
        all_neighbors = healpy.pixelfunc.get_all_neighbours(nside, pix)
        for pixn in range(0,8,2):
            filename = 'gobjs_' + str(band) + '_nside' + str(nside) + '_p' + str(all_neighbors[pixn])
            if os.path.isfile(filename):
                # print >> sys.stderr, "Reading in file-resolution pixel %d" %all_neighbors[pixn]
                file = open(os.path.join(globals_dir,filename), 'rb')
                global_objs_list.extend(cPickle.load(file))
                file.close()

    elif nside > nside_file:
        print "ERROR: Need to have global objects saved in a pixelization that is not lower-resolution than what you're using"
        throw

    elif nside == 0:
        # just read in everything!!
        npix_file = 12*nside_file*nside_file
        for pix in range(npix_file):
            filename = 'gobjs_' + str(band) + '_nside' + str(nside_file) + '_p' + str(pix)
            if os.path.isfile(os.path.join(globals_dir,filename)):
                # print >> sys.stderr, "Reading in file-resolution pixel %d" %pix
                file = open(os.path.join(globals_dir,filename), 'rb')
                global_objs_list.extend(cPickle.load(file))
                file.close()
        if len(global_objs_list) == 0:
            print "ERROR: no global objects found for filter %s, file nside %d" %(band, nside_file)
            throw;
        pix = 0

    else:
        # what nside_file pixels correspond to our central and neighbor pixels?
        # this snazzy trick is from the healpy source code.  works for the nest scheme only.
        npix_lowres = 12*nside*nside
        npix_file = 12*nside_file*nside_file
        rat2 = npix_file/npix_lowres
        map_file = np.arange(npix_file)
        map_lowres = map_file.reshape(npix_lowres,rat2)

        # so what is our pixel in nest scheme, low resolution?
        pix_lowres_nest = healpy.pixelfunc.ring2nest(nside,pix)
        # this is made up of the following pixels.  this is the snazzy trick.
        pixels_nest = map_lowres[pix_lowres_nest,:]
        # convert back to ring
        pixels_ring = healpy.pixelfunc.nest2ring(nside_file,pixels_nest)
        # and read them from the file.
        for pixel_ring in pixels_ring:
            filename = 'gobjs_' + str(band) + '_nside' + str(nside_file) + '_p' + str(pixel_ring)
            # if there's no data for this pixel, forget it: nothing to calibrate.
            if os.path.isfile(os.path.join(globals_dir,filename)):
                # print >> sys.stderr, "Reading in file-resolution pixel %d" %pixel_ring
                file = open(os.path.join(globals_dir,filename), 'rb')
                global_objs_list.extend(cPickle.load(file))
                file.close()
        # if there weren't any objects for this low-res pixel, then we're done.
        if len(global_objs_list) == 0:
            return None

        # now include the four nearest neighbors
        all_neighbors = healpy.pixelfunc.get_all_neighbours(nside, pix_lowres_nest, nest=True)
        for pixn in range(0,8,2):
            # what are these in the high res pixels?
            pixels_nest = map_lowres[all_neighbors[pixn],:]
            pixels_ring = healpy.pixelfunc.nest2ring(nside_file,pixels_nest)
            for pixel_ring in pixels_ring:
                filename = 'gobjs_' + str(band) + '_nside' + str(nside_file) + '_p' + str(pixel_ring)
                # if there's no data for this pixel, forget it: nothing to calibrate.
                if os.path.isfile(os.path.join(globals_dir,filename)):
                    # print >> sys.stderr, "Reading in file-resolution pixel %d" %pixel_ring
                    file = open(os.path.join(globals_dir,filename), 'rb')
                    global_objs_list.extend(cPickle.load(file))
                    file.close()
    
    if verbose:
        print 'read_global_objs: Returning {0} objects for pixel {1}'.format(len(global_objs_list),pix)
    
    return global_objs_list
