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
src_dir = '/Users/bauer/software/python'
sys.path.append(src_dir)
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


def nebencalibrate_pixel( inputs ):
    
    [p, nside, band, pix_wobjs, precal, precam_stars, precam_map, globals_dir, id_string] = inputs
    
    pix = pix_wobjs[p]
    
    use_precam = 0
    if len(precam_stars) > 0:
        use_precam = 1

    max_nepochs = 50 # ick!
    magerr_sys2 = 0.0004
    
    stars = []
    star_map = index.Index()
    imgids = dict()
    image_id_dict = dict()
    global_objects = []
    has_precam = 0
    p_vector = None
    
    print "Starting pixel %d" %(pix)
    
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
    
    
    # read the global objects in from the file
    filename = 'gobjs_' + str(band) + '_nside' + str(nside) + '_p' + str(pix)
    file = open(os.path.join(globals_dir,filename), 'rb')
    global_objs_list = cPickle.load(file)
    file.close()
    max_matrix_size = len(global_objs_list)*max_nepochs
    a_matrix_xs = np.zeros(max_matrix_size)
    a_matrix_ys = np.zeros(max_matrix_size)
    a_matrix_vals = np.zeros(max_matrix_size)

    # is it worth finding out max_nimgs correctly?  YES.
    imgids = dict()
    ndet_for_cal = 0
    ndet_for_summi = 0
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
            image_id = obj[id_string] #int(entries[i])
            imgids[image_id] = True

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
                precam_count += 1

    max_nimgs = len(imgids.keys())
    
    for go in global_objs_list:
        
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



def nebencalibrate( band, nside, id_string, precam_map, precam_stars, precal, globals_dir ):
    
    # first let's just see which pixels are populated so that we can loop over only those.
    npix = healpy.nside2npix(nside)
    pix_wobjs = []
    pix_dict = dict()

    # which pixels exist?  check file names.
    dir_files = os.listdir(globals_dir)
    for filename in dir_files:
        if re.match('gobjs_' + str(band) + '_nside' + str(nside) + '_p', filename) is not None:
            pixel_match = re.search('p(\d+)', filename)
            if pixel_match is None:
                print "Problem parsing pixel number"
            pixel = int(pixel_match.group()[1:])
            pix_dict[pixel] = True

    pix_wobjs = sorted(pix_dict.keys())
    npix_wobjs = len(pix_wobjs)
    print "%d pixels have data" %npix_wobjs
    
    p_vector = []
    image_id_dict = []
    for pix in range(npix):
        p_vector.append(None)
        image_id_dict.append(dict())
        
    has_precam = dict()
    inputs= []
    for p in range(npix_wobjs):
        inputs.append( [p, nside, band, pix_wobjs, precal, precam_stars, precam_map, globals_dir, id_string] )
        image_id_dict[pix_wobjs[p]], p_vector[pix_wobjs[p]], has_precam[pix_wobjs[p]] = nebencalibrate_pixel(inputs[p])
     
    
    # now do another ubercalibration to make sure the pixels are normalized to each other
    # the measurements are now the difference between two zps calculated for the same image, during different pixels' ubercalibrations.
    # what is the "correct" value?  the mean, since that will bring the different zps to the same value and that's what matters?

    # first, get a superlist of image ids
    imagelist = []
    for pix in range(npix):
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
    
    print "looping through imagelist, e.g."
    print imagelist[0]
    print "max matrix size = %d" %max_matrix_size

    for imageid in imagelist:
        zps = []
        pix_id_matrix2 = np.zeros([npix_wobjs,npix_wobjs], dtype=np.int)
        pix_ids2 = np.zeros(npix_wobjs, dtype=np.int)

        nzps = 0
        
        for pix_id in range(npix_wobjs):
            pix = pix_wobjs[pix_id]
            if imageid in image_id_dict[pix] and p_vector[pix][image_id_dict[pix][imageid]] != 0.:
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
        # print "indices has length %d, nzps = %d, matrix size = %d out of %d max" %(len(indices), nzps, matrix_size, max_matrix_size)

        if( matrix_size+len(indices) > max_matrix_size ):
            a_matrix_xs.extend(np.zeros(max_matrix_size))
            a_matrix_ys.extend(np.zeros(max_matrix_size))
            a_matrix_vals.extend(np.zeros(max_matrix_size))
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


    
    # now compile all the info into one zp per image.
    # for each image, find the p_vector zps and add the p1_vector zps, then take the (TBD: WEIGHTED) mean
    
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
    
    # outfile = open( outfilename, 'w' )
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
        # outfile.write( "%d %e\n" %(imageid, zp_tot) );
    # outfile.close()
    
    return zeropoints


def calibrate_by_filter(config, band, by_exposure=True, by_image=True):
    
    print "\nCalibrating filter " + band + "!\n"
    
    flux_name = "flux_" + "%s" %config['mag_type']
    flux_err_name = "flux_err_" + "%s" %config['mag_type']
    
    # read in the nominal ccd offsets (zp_phots), which will be our starting point
    zp_phot_file = open( config['zp_phot_filename'], 'r' )
    zp_phots = dict()
    for line in zp_phot_file:
        entries = line.split()
        if entries[0][0] == '#':
            continue
        zp_phots[int(entries[0])] = -1.0*float(entries[4])
    # include precam fake entry
    zp_phots[0] = 0.
    
    
    precam_stars = []
    precam_map = index.Index()
    if config['use_precam']:
        precam_file = open( config['precam_filename'], 'r' )
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
        
    if config['use_sdss']:
        sdssfile = open(config['sdss_filename'], 'r')
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
                precam_stars.append(sdss_obj)
                precam_map.insert( count, (sdss_obj['ra'],sdss_obj['dec'],sdss_obj['ra'],sdss_obj['dec']) )
                count += 1
        print "Read in %d SDSS standards" %count
    
    
    if by_exposure:
        print "\nStarting exposure calibration\n"
        precal = [[zp_phots,'ccd', np.median(zp_phots.values())]]
        exp_zps = nebencalibrate( band, config['nside_exp'], 'exposureid', precam_map, precam_stars, precal, config['globals_dir'] )
        
        outfile = open( config['exp_outfilename'], 'w' )
        for zp_id in exp_zps.keys():
            outfile.write( "%d %e\n" %(zp_id, exp_zps[zp_id]) );
        outfile.close()
        
    if by_image:
        print "\nStarting image calibration\n"
        precal.append([exp_zps,'exposure_id', np.median(exp_zps.values())])
        img_zps = nebencalibrate( band, config['nside_img'], 'image_id', precam_map, precam_stars, precal, config['globals_dir'] )
    
        outfile = open( config['img_outfilename'], 'w' )
        for zp_id in img_zps.keys():
            outfile.write( "%d %e\n" %(zp_id, img_zps[zp_id]) );
        outfile.close()
    

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
    for filt in config['filters']:
        calibrate_by_filter( config, filt, config['by_exposure'], config['by_image'] )



if __name__ == '__main__':
    main()


