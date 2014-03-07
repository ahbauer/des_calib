import sys
import os
import math
import re
import copy
from operator import itemgetter
import cPickle
import numpy as np
from scipy.sparse import vstack as sparsevstack
from scipy.sparse import linalg
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from rtree import index
from scipy.stats import scoreatpercentile
import tables
import healpy
import yaml
from pyspherematch import spherematch

"""
nebencal.py

Performs an ubercal-like calibration.

Reads global objects made by make_global_objs.py, using DESDB output.

For now, makes one ZP per exposure and one per image.  Can be generalized to solve for spatial dependence, non-linearity.

There are some inputs that might want changing in main() and calibrate_by_filter()
"""

class headerline:
    def __init__(self):
        self.ra = None
        self.dec = None
        self.band = None
        self.mag_psf = None
        self.magerr_psf = None
        self.x_image = None
        self.y_image = None
        self.imageid = None
        self.ccd = None


class global_object:
    def __init__(self):
        self.ra = None
        self.dec = None
        self.objects = []


# for now.....
good_quality_magerr = 0.02
def good_quality(star):
    if star['magerr_psf'] < good_quality_magerr:
        return True
    return False

def standard_quality(star):
    if star['image_id'] != 1: # and star['magerr_psf'] < good_quality_magerr: # and star['gskyphot'] == 1:
        return True
    else:
        return False

def ok_quality(star):
    if star['magerr_psf'] < 0.04:
        return True
    return False

def unmatched(star):
    if star['matched']:
        return False
    return True

def read_precam( precam_stars, precam_map, config, band ):
    precam_file = open( config['general']['precam_filename'], 'r' )
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
        star['x_image'] = 0.
        star['y_image'] = 0.
        star['ccd'] = 0
        star['image_id'] = 1
        star['exposureid'] = 1
        star['band'] = band
        star['matched'] = False
        precam_stars.append(star)
        precam_map.insert( count, (star['ra'],star['dec'],star['ra'],star['dec']) )
        count += 1
    print "Read in %d PreCam standards" %count
    

def read_sdss( sdss_stars, sdss_map, config, band ):
    sdssfile = open(config['general']['sdss_filename'], 'r')
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
        sdss_obj['x_image'] = 0.
        sdss_obj['y_image'] = 0.
        sdss_obj['ccd'] = 0
        sdss_obj['image_id'] = 1
        sdss_obj['exposureid'] = 1
        sdss_obj['matched'] = False
        if sdss_obj['mag_psf'] > 0.:
            sdss_stars.append(sdss_obj)
            sdss_map.insert( count, (sdss_obj['ra'],sdss_obj['dec'],sdss_obj['ra'],sdss_obj['dec']) )
            count += 1
    print "Read in %d SDSS standards" %count

def nebencalibrate_pixel( inputs ):
    
    [pix, nside, nside_file, band, precal, precam_stars, precam_map, globals_dir, id_string, max_dets] = inputs
    
    use_precam = 0
    if len(precam_stars) > 0:
        use_precam = 1

    max_nepochs = 5 # ick!  just a starting point, though.
    magerr_sys2 = 0.0004
    
    stars = []
    star_map = index.Index()
    imgids = dict()
    image_id_dict = dict()
    global_objects = []
    has_precam = 0
    p_vector = None
    
    
    # while we're at it, add stuff to the matrices so we don't have to loop over global objects.
    
    match_radius = 1.0/3600.
    image_id_count = -1
    image_ras = dict()
    image_decs = dict()
    image_ns = dict()
    ndets = 0
    b_vector = []
    a_matrix = None
    c_vector = []
    gcount = 0
    sum_rms = 0.
    n_rms = 0
    matrix_size = 0
    n_good_e = 0
    n_bad_e = 0
    
    precam_count = 0
    
    # read the global objects in from the files.
    global_objs_list = []
    
    # if the pixelization resolution is the same as the file convention, then this is easy.
    if nside == nside_file:
        filename = 'gobjs_' + str(band) + '_nside' + str(nside) + '_p' + str(pix)
        # if there's no data for this pixel, forget it: nothing to calibrate.
        if not os.path.isfile(os.path.join(globals_dir,filename)):
            return None, None, None
        # print >> sys.stderr, "Reading in file-resolution pixel %d" %pix
        file = open(os.path.join(globals_dir,filename), 'rb')
        global_objs_list = cPickle.load(file)
        file.close()
        
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
            return None, None, None
            
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
        
    
    print "Starting pixel %d: %d global objects" %(pix, len(global_objs_list))
    
    max_matrix_size = len(global_objs_list)*max_nepochs
    a_matrix_xs = np.zeros(max_matrix_size)
    a_matrix_ys = np.zeros(max_matrix_size)
    a_matrix_vals = np.zeros(max_matrix_size)

    # is it worth finding out max_nimgs correctly?  YES.
    # also, check to see if there are too many objects per image/exposure
    imgids = dict()
    img_ndets = dict()
    img_errs = dict()
    max_errors = dict()
    ndet_for_cal = 0
    ndet_for_summi = 0
    worst_ok_err = dict()
    for go in global_objs_list:

        # cut out objects that are bad quality!
        go.objects = filter(good_quality, go.objects)
        if len(go.objects) < 2:
            continue
        
        for obj in go.objects:
            # if bad quality (or precam), don't use for calibration
            if standard_quality(obj):
                obj['for_mean'] = True
                ndet_for_summi += 1
            else:
                obj['for_mean'] = False
            ndet_for_cal += 1
            image_id = obj[id_string]
            imgids[image_id] = True
            
            if obj[id_string] in img_ndets:
                img_ndets[obj[id_string]] += 1
                if obj['magerr_psf'] < worst_ok_err[obj[id_string]][1]:
                    # replace the current worst error in the list
                    img_errs[obj[id_string]][worst_ok_err[obj[id_string]][0]] = obj['magerr_psf']
                    # now what's the worst?
                    max_index = np.argmax(img_errs[obj[id_string]])
                    worst_ok_err[obj[id_string]] = [max_index, img_errs[obj[id_string]][max_index]]
            else:
                img_ndets[obj[id_string]] = 0
                img_errs[obj[id_string]] = np.ones(max_dets) #[obj['magerr_psf']]
                worst_ok_err[obj[id_string]] = [0,1.0] # index, value
            
        if use_precam:
            ra_match_radius = match_radius/math.cos(go.dec*math.pi/180.)
            dec_match_radius = match_radius
            match_area = (go.ra-ra_match_radius, go.dec-dec_match_radius,
                            go.ra+ra_match_radius, go.dec+dec_match_radius)
            det_indices = list(precam_map.intersection(match_area))
            if len(det_indices) == 1:
                ndet_for_cal += 1
                precam_star = precam_stars[det_indices[0]]
                precam_star['for_mean'] = False
                go.objects.append(precam_star)
                imgids[1] = True
                worst_ok_err[1] = [0,1.0] # don't cut any precam objects
                precam_count += 1

    max_nimgs = len(imgids.keys())
    
    print "Finished first pass through global objects"
    n_to_clip = 0
    for i in worst_ok_err.keys():
        if worst_ok_err[i][1] < 1.:
            n_to_clip += 1
    print "%d regions to calibrate, %d with a clipped number of detections" %(len(worst_ok_err.keys()), n_to_clip)
    
    for go in global_objs_list:
        
        if len(go.objects) < 2:
            continue
        
        go.objects = [obj for obj in go.objects if obj['magerr_psf'] < worst_ok_err[obj[id_string]][1]]
        
        if len(go.objects) < 2:
            continue
        
        ndet_for_summi = len([obj for obj in go.objects if obj['for_mean'] == True] )
        ndet_for_cal = len(go.objects)
        ra = go.ra
        dec = go.dec
        
        if ndet_for_summi < 2:
            continue
        
        # magnitudes = [obj['mag_psf'] for obj in go.objects if obj['for_mean'] == True]
        # print magnitudes
        
        mags_for_summi = np.zeros(ndet_for_summi)
        mag_errors_for_summi = np.zeros(ndet_for_summi)
        mags_for_cal = np.zeros(ndet_for_cal)
        mag_errors_for_cal = np.zeros(ndet_for_cal)
        image_ids = np.zeros(ndet_for_cal, dtype=np.int)
        image_id_matrix = np.zeros([max_nimgs,ndet_for_cal], dtype=np.int)
        d_for_summi = 0
        d_for_cal = 0
        for star in go.objects:
            if star[id_string] != 1: # if not precam  CHEAP HACK
                star['magerr_psf'] = np.sqrt(star['magerr_psf']*star['magerr_psf'] + magerr_sys2)
            try:
                image_ids[d_for_cal] = image_id_dict[star[id_string]]
                image_id_matrix[image_ids[d_for_cal],d_for_cal] = 1
                image_ras[star[id_string]] += star['ra']
                image_decs[star[id_string]] += star['dec']
                image_ns[star[id_string]] += 1
            except KeyError:
                # the current image id is not in the image_ids dictionary yet
                image_id_count += 1

                if image_id_count >= max_nimgs:
                    print "WARNING, image_id_count = %d > max_nimgs = %d" %(image_id_count, max_nimgs)

                image_id_dict[star[id_string]] = image_id_count
                image_ids[d_for_cal] = image_id_dict[star[id_string]]
                image_id_matrix[image_ids[d_for_cal],d_for_cal] = 1
                image_ras[star[id_string]] = star['ra']
                image_decs[star[id_string]] = star['dec']
                image_ns[star[id_string]] = 1
            
            mags_for_cal[d_for_cal] = star['mag_psf']
            
            for p in precal:
                [pre_vector, precal_id_string, pre_median] = p
                if star[precal_id_string] in pre_vector:
                    mags_for_cal[d_for_cal] += pre_vector[star[precal_id_string]]
                    n_good_e += 1
                else:
                    mags_for_cal[d_for_cal] += pre_median
                    n_bad_e += 1
            
            mag_errors_for_cal[d_for_cal] = star['magerr_psf']
            
            if star['for_mean']:
                mags_for_summi[d_for_summi] = mags_for_cal[d_for_cal]
                mag_errors_for_summi[d_for_summi] = mag_errors_for_cal[d_for_cal]
                d_for_summi += 1
                
            d_for_cal += 1
            
        
        if d_for_summi != ndet_for_summi:
            print "WARNING, d_for_summi = %d, ndet_for_summi = %d" %(d_for_summi, ndet_for_summi)
        if d_for_cal != ndet_for_cal:
            print "WARNING, d_for_cal = %d, ndet_for_cal = %d" %(d_for_cal, ndet_for_cal)

        invsigma_array_for_cal = 1.0/np.square(mag_errors_for_cal)
        invsigma_array_for_summi = 1.0/np.square(mag_errors_for_summi)
        sum_invsigma2_for_summi = invsigma_array_for_summi.sum()
        sum_invsigma2_for_cal = invsigma_array_for_cal.sum()
        sum_m_i = (mags_for_summi*invsigma_array_for_summi).sum() / sum_invsigma2_for_summi

        invsigma_matrix = np.tile(invsigma_array_for_cal, (max_nimgs,1))
        sum_for_zps = (image_id_matrix*invsigma_matrix).sum(axis=1) / sum_invsigma2_for_cal
        
        b_vector = np.append( b_vector, mags_for_cal - sum_m_i )
        
        a_submatrix = np.tile(sum_for_zps, (ndet_for_cal,1) )
        a_submatrix[range(ndet_for_cal),image_ids[range(ndet_for_cal)]] -= 1.0

        a_submatrix = coo_matrix(a_submatrix)
        
        indices = np.where(a_submatrix.data != 0.)[0]
        if( matrix_size+len(indices) > max_matrix_size ):
            a_matrix_xs = np.hstack((a_matrix_xs,np.zeros(max_matrix_size)))
            a_matrix_ys = np.hstack((a_matrix_ys,np.zeros(max_matrix_size)))
            a_matrix_vals = np.hstack((a_matrix_vals,np.zeros(max_matrix_size)))
            max_matrix_size += max_matrix_size

        a_matrix_xs[matrix_size:(matrix_size+len(indices))] = a_submatrix.col[indices]
        a_matrix_ys[matrix_size:(matrix_size+len(indices))] = a_submatrix.row[indices]+ndets
        a_matrix_vals[matrix_size:(matrix_size+len(indices))] = a_submatrix.data[indices]
        matrix_size += len(indices)

        c_vector = np.append(c_vector, invsigma_array_for_cal)

        # add up some stats
        sum_rms += np.std(mags_for_summi-sum_m_i)
        n_rms += 1
        ndets += ndet_for_cal
            
    print "%d images" %len(imgids.keys())
    print "%d good previous zps, %d bad" %(n_good_e, n_bad_e)
    
    star_map = None
    
    if( ndets == 0 ):
        # continue
        return image_id_dict, p_vector, has_precam
    
    for iid in image_ras.keys():
        image_ras[iid] /= image_ns[iid]
        image_decs[iid] /= image_ns[iid]
    
    gcount = len(global_objs_list)
    print "Looped through %d global objects, %d measurements total, %d from precam" %( gcount, ndets, precam_count )
    print "Mean RMS of the stars: %e" %(sum_rms/n_rms)

    a_matrix = coo_matrix((a_matrix_vals[0:matrix_size], (a_matrix_ys[0:matrix_size], a_matrix_xs[0:matrix_size])), shape=(ndets,max_nimgs))

    c_matrix = lil_matrix((ndets,ndets))
    c_matrix.setdiag(1.0/c_vector)
    
    # print a_matrix
    # print b_vector

    print "Calculating intermediate matrices..."
    subterm = (a_matrix.transpose()).dot(c_matrix)
    termA = subterm.dot(a_matrix)
    termB = subterm.dot(b_vector)
    print "Solving!"
    # p_vector = linalg.bicgstab(termA,termB)[0]
    # p_vector = linalg.spsolve(termA,termB)
    p_vector = linalg.minres(termA,termB)[0]

    # normalize to the precam standards
    if use_precam:
        if 1 in image_id_dict:
            print "Normalizing to the PreCam standards!"
            p_vector -= p_vector[image_id_dict[1]]
            has_precam = 1
            
    print "Solution:"
    print p_vector
    
    
    # now calculate how we did
    sum_rms = 0.
    n_rms = 0
    sum_rms_formean = 0.
    n_rms_formean = 0.
    n_good_e = 0
    n_bad_e = 0
    for go in global_objs_list:
        
        if len(go.objects) < 2:
            continue
        
        mags = np.zeros(len(go.objects))
        mag_errors = np.zeros(len(go.objects))
        mags_formean = np.zeros(len([obj for obj in go.objects if obj['for_mean']]))
        mag_errors_formean = np.zeros(len(mags_formean))
        i=0
        i2=0
        bad = dict()
        for star in go.objects:
            try:
                mags[i] = star['mag_psf'] + p_vector[image_id_dict[star[id_string]]]
                mag_errors[i] = star['magerr_psf']
                if star['for_mean']:
                    mags_formean[i2] = star['mag_psf'] + p_vector[image_id_dict[star[id_string]]]
                    mag_errors_formean[i2] = star['magerr_psf']
                for p in precal:
                    [pre_vector, precal_id_string, pre_median] = p
                    if star[precal_id_string] in pre_vector:
                        n_good_e += 1
                        mags[i] += pre_vector[star[precal_id_string]]
                        if star['for_mean']:
                            mags_formean[i2] += pre_vector[star[precal_id_string]]
                            mag_errors_formean[i2] = star['magerr_psf']
                    else:
                        n_bad_e += 1
                        mags[i] += pre_median
                        if star['for_mean']:
                            mags_formean[i2] += pre_median
                if star['for_mean']:
                    i2 += 1
                i += 1
            except KeyError:
                bad[star[id_string]] = True
        if i>1:
            mags = mags[0:i]
            mag_errors = mag_errors[0:i]
            invsigma_array = 1.0/np.square(mag_errors)
            sum_invsigma2 = invsigma_array.sum()
            sum_m_i = (mags*invsigma_array).sum() / sum_invsigma2
            sum_rms += np.std(mags-sum_m_i)
            n_rms += 1
        if i2>1:
            mags_formean = mags_formean[0:i2]
            mag_errors_formean = mag_errors_formean[0:i2]
            invsigma_array = 1.0/np.square(mag_errors_formean)
            sum_invsigma2 = invsigma_array.sum()
            sum_m_i = (mags_formean*invsigma_array).sum() / sum_invsigma2
            sum_rms_formean += np.std(mags_formean-sum_m_i)
            n_rms_formean += 1
            
    if n_rms == 0:
        n_rms = 1
    if n_rms_formean == 0:
        n_rms_formean = 1
    
    print "Mean RMS after calibration: %e (%e with standards), %d/%d images calibrated" %(sum_rms_formean/n_rms_formean, sum_rms/n_rms, len(p_vector)-len(bad.keys()), len(p_vector))
    
    return image_id_dict, p_vector, has_precam



def nebencalibrate( band, nside, nside_file, id_string, precam_map, precam_stars, precal, globals_dir, require_standards=False, max_dets=1e9 ):
    
    if len(precam_stars) == 0 and require_standards:
        print "nebencalibrate: No standards, yet you are requiring standards!"
        raise Exception
    
    npix = healpy.nside2npix(nside)
    
    p_vector = []
    image_id_dict = []
    for pix in range(npix):
        p_vector.append(None)
        image_id_dict.append(dict())
        
    # nebencalibrate the pixels!
    has_precam = dict()
    pix_wobjs = []
    inputs= []
    for p in range(npix):
        inputs.append( [p, nside, nside_file, band, precal, precam_stars, precam_map, globals_dir, id_string, max_dets] )
        image_id_dict[p], p_vector[p], has_precam[p] = nebencalibrate_pixel(inputs[p])
        if require_standards and not has_precam[p] and p_vector[p] is not None:
            print "No standards found in pixel %d for nside=%d.  Degrading!" %(p,nside)
            return 'degrade'
        if p_vector[p] is not None:
            pix_wobjs.append(p)
    npix_wobjs = len(pix_wobjs)
    
    # now do another ubercalibration to make sure the pixels are normalized to each other
    # the measurements are now the difference between two zps calculated for the same image, during different pixels' ubercalibrations.
    
    # first, get a superlist of image ids
    imagelist = []
    for pix in range(npix):
        if image_id_dict[pix] is not None:
            imagelist.extend(image_id_dict[pix].keys())
    imagelist = list(set(imagelist))
    print "There are %d images in total." %len(imagelist)
    
    sum_for_zps = np.zeros(npix_wobjs)
    b_vector = []
    a_matrix = None
    c_vector = []
    sum_rms = 0.
    n_rms = 0
    matrix_size = 0
    max_matrix_size = len(imagelist)*npix_wobjs
    a_matrix_xs = np.zeros(max_matrix_size)
    a_matrix_ys = np.zeros(max_matrix_size)
    a_matrix_vals = np.zeros(max_matrix_size)
    pix_id_dict2 = dict()
    nzps_tot = 0
    pix_id_count = -1
    
    for imageid in imagelist:
        zps = []
        pix_id_matrix2 = np.zeros([npix_wobjs,npix_wobjs], dtype=np.int)
        pix_ids2 = np.zeros(npix_wobjs, dtype=np.int)
        
        nzps = 0
        for pix_id in range(npix_wobjs):
            pix = pix_wobjs[pix_id]
            if imageid in image_id_dict[pix] and (imageid == 1 or p_vector[pix][image_id_dict[pix][imageid]] != 0.):
                zps.append(p_vector[pix][image_id_dict[pix][imageid]])
                
                try:
                    pix_ids2[nzps] = pix_id_dict2[pix_id]
                    pix_id_matrix2[pix_ids2[nzps],nzps] = 1
                    
                except KeyError:
                    # the current image id is not in the image_ids dictionary yet
                    pix_id_count += 1
                    pix_id_dict2[pix_id] = pix_id_count
                    pix_ids2[nzps] = pix_id_dict2[pix_id]
                    pix_id_matrix2[pix_ids2[nzps],nzps] = 1
                
                nzps += 1

        if nzps == 0:
            print "image %d has no zps" %imageid
            continue

        pix_id_matrix2 = pix_id_matrix2[:,0:nzps]
        
        zps = np.array(zps)
        zp_errors = np.ones(nzps)
        invsigma_array = 1.0/np.square(zp_errors)
        sum_invsigma2 = invsigma_array.sum()
        sum_zps_i = (zps*invsigma_array).sum() / sum_invsigma2
        

        invsigma_matrix = np.tile(invsigma_array, (npix_wobjs,1))
        sum_for_zps = (pix_id_matrix2*invsigma_matrix).sum(axis=1) / sum_invsigma2
        b_vector = np.append( b_vector, zps - sum_zps_i )
        
        a_submatrix = np.tile(sum_for_zps, (nzps,1) )
        a_submatrix[range(nzps),pix_ids2[range(nzps)]] -= 1.0
        a_submatrix = coo_matrix(a_submatrix)

        indices = np.where(a_submatrix.data != 0.)[0]

        if( matrix_size+len(indices) > max_matrix_size ):
            a_matrix_xs = np.hstack((a_matrix_xs,np.zeros(max_matrix_size)))
            a_matrix_ys = np.hstack((a_matrix_ys,np.zeros(max_matrix_size)))
            a_matrix_vals = np.hstack((a_matrix_vals,np.zeros(max_matrix_size)))
            max_matrix_size += max_matrix_size

        a_matrix_xs[matrix_size:(matrix_size+len(indices))] = a_submatrix.col[indices]
        a_matrix_ys[matrix_size:(matrix_size+len(indices))] = a_submatrix.row[indices]+nzps_tot
        a_matrix_vals[matrix_size:(matrix_size+len(indices))] = a_submatrix.data[indices]
        matrix_size += len(indices)

        c_vector = np.append(c_vector, invsigma_array)
        
        # add up some stats
        sum_rms += np.std(zps-sum_zps_i)
        n_rms += 1
        nzps_tot += nzps
        
    print "Looped through the images"
    print "Mean RMS of the zps: %e" %(sum_rms/n_rms)
    print "Matrix size is %d out of %d" %(matrix_size, max_matrix_size)
    
    a_matrix = coo_matrix((a_matrix_vals[0:matrix_size], (a_matrix_ys[0:matrix_size], a_matrix_xs[0:matrix_size])), shape=(nzps_tot,npix_wobjs))

    c_matrix = lil_matrix((nzps_tot,nzps_tot))
    c_matrix.setdiag(1.0/c_vector)

    print "Calculating intermediate matrices..."
    subterm = (a_matrix.transpose()).dot(c_matrix)
    termA = subterm.dot(a_matrix)
    termB = subterm.dot(b_vector)
    print "Solving!"
    # p_vector = linalg.bicgstab(termA,termB)[0]
    # p_vector = linalg.spsolve(termA,termB)
    p1_vector = linalg.minres(termA,termB)[0]
    print "Solution:"
    print p1_vector

    # how did we do with this?
    sum_rms = 0.
    n_rms = 0
    for imageid in imagelist:
        zps = []
        pix_id_matrix2 = np.zeros([npix_wobjs,npix_wobjs], dtype=np.int)
        pix_ids2 = np.zeros(npix_wobjs, dtype=np.int) # *len(imagelist)

        for pix_id in range(npix_wobjs):
            pix = pix_wobjs[pix_id]
            if imageid in image_id_dict[pix]:
                zps.append(p_vector[pix][image_id_dict[pix][imageid]]+p1_vector[pix_id_dict2[pix_id]])

        zps = np.array(zps)
        zp_errors = np.ones(len(zps))
        invsigma_array = 1.0/np.square(zp_errors)
        sum_invsigma2 = invsigma_array.sum()
        sum_zps_i = (zps*invsigma_array).sum() / sum_invsigma2

        sum_rms += np.std(zps-sum_zps_i)
        n_rms += 1
        
    print "Mean RMS of the zps after correction: %e" %(sum_rms/n_rms)
    
    
    # if using precam standards, normalize these zps to the pixels that had precam objects in them
    if len(precam_stars) > 0:
        precam_zp = 0.
        precam_nzp = 0
        for pix_id in pix_id_dict2.keys():
            p1_vector_index = pix_id_dict2[pix_id]
            pix = pix_wobjs[pix_id]
            if pix in has_precam:
                if has_precam[pix]:
                    precam_zp += p1_vector[p1_vector_index]
                    precam_nzp += 1
        if precam_nzp > 0:
            precam_zp /= precam_nzp
        print "Normalizing all zps by the PreCam mean of %f from %d zps" %(precam_zp, precam_nzp)
        p1_vector -= precam_zp
    
    
    # now compile all the info into one zp per image.
    # for each image, find the p_vector zps and add the p1_vector zps, then take the (TBD: WEIGHTED?) mean
    zeropoints = dict()
    for imageid in imagelist:
        zp_tot = 0.
        zp_n = 0
        for pix_id in range(npix_wobjs):
            pix = pix_wobjs[pix_id]
            if imageid in image_id_dict[pix]:
                zp_tot += p_vector[pix][image_id_dict[pix][imageid]] + p1_vector[pix_id_dict2[pix_id]] 
                zp_n += 1
        zp_tot /= zp_n
        zeropoints[imageid] = zp_tot
    
    return zeropoints


def calibrate_by_filter(config):
    
    band = config['general']['filter']
    print "\nCalibrating filter " + band + "!\n"
    
    # read in the nominal ccd offsets (zp_phots), which will be our starting point
    zp_phot_file = open( config['general']['zp_phot_filename'], 'r' )
    zp_phots = dict()
    for line in zp_phot_file:
        entries = line.split()
        if entries[0][0] == '#':
            continue
        zp_phots[int(entries[0])] = -1.0*float(entries[4])
    # include precam fake entry
    zp_phots[0] = 0.
    
    # read in any catalogs we would like to treat as standards (i.e. precam, sdss...)
    precam_stars = []
    precam_map = index.Index()
    if config['general']['use_precam']:
        read_precam( precam_stars, precam_map, config, band )
    if config['general']['use_sdss']:
        read_sdss( precam_stars, precam_map, config, band )
    
    
    # calibrate!
    
    # use the zp_phots as a starting point to our calibration
    precal = [[zp_phots, 'ccd', np.median(zp_phots.values())]]
    
    for calibration in config['calibrations']:
        print "\nStarting calibration on %s\n" %calibration['id_string']
        new_zps = None
        success = False
        while not success:
            new_zps = nebencalibrate( band, calibration['nside'], config['general']['nside_file'], calibration['id_string'], precam_map, precam_stars, precal, config['general']['globals_dir'], require_standards=calibration['require_standards'], max_dets=calibration['max_dets'] )
            if new_zps == 'degrade':
                if calibration['nside'] == 1:
                    calibration['nside'] = 0
                calibration['nside'] /= 2
                print "Now trying nside=%d" %calibration['nside']
            else:
                success = True
                
        outfile = open( calibration['outfilename'], 'w' )
        for zp_id in new_zps.keys():
            outfile.write( "%d %e\n" %(zp_id, new_zps[zp_id]) );
        outfile.close()
        
        # so that we use these as input to the following calibration(s)
        precal.append([new_zps, calibration['id_string'], np.median(new_zps.values())])
        

def main():
    
    if len(sys.argv) != 2:
        print "Usage: nebencal.py config_filename"
        print "       To print out a default config file, run: nebencal.py default"
        exit(1)
    elif sys.argv[1] == 'default':
        config = open( os.path.dirname(os.path.realpath(__file__)) + '/configs/config_sample', 'r' )
        print "\n### DEFAULT EXAMPLE CONFIG FILE ###\n"
        for line in config:
            sys.stdout.write(line)
        exit(0)
    
    config_filename = sys.argv[1]
    print "Running nebencalibration using config file %s!" %config_filename
    
    config_file = open(config_filename)
    config = yaml.load(config_file)
    calibrate_by_filter( config )



if __name__ == '__main__':
    main()


