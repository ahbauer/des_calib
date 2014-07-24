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
from scipy.sparse import cs_graph_components
from rtree import index
from scipy.stats import scoreatpercentile
import tables
import healpy
import yaml
# from pyspherematch import spherematch
from nebencal_utils import read_precam
from nebencal_utils import read_sdss
from nebencal_utils import global_object
colorterm_dir = '/Users/bauer/software/python/des/des_calib/ting/color-term'
sys.path.append(colorterm_dir)
import color_term_cal

"""
nebencal.py

Performs an ubercal-like calibration.

Reads global objects made by make_global_objs.py, using DESDB output.

For now, makes one ZP per exposure and one per image.  Can be generalized to solve for spatial dependence, non-linearity.

There are some inputs that might want changing in main() and calibrate_by_filter()
"""


# for now.....
good_quality_magerr = 0.02
def good_quality(star):
    if star['magerr_psf'] < good_quality_magerr:
        return True
    return False

def standard_quality(star):
    if star['image_id'] != 1 and star['magerr_psf'] < good_quality_magerr and star['gskyphot'] == 1 and star['x_image']>100. and star['x_image']<1900. and star['y_image']>100. and star['y_image']<3950.and star['cloud_nomad']<0.2:
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
    
    [pix, nside, nside_file, band, precal, precam_stars, precam_map, globals_dir, id_strings, operands, max_dets, require_standards, ra_min, ra_max, dec_min, dec_max, use_color_terms, mag_classes] = inputs
    
    use_precam = 0
    if len(precam_stars) > 0:
        use_precam = 1
        
    max_nepochs = 5 # ick!  just a starting point, though.
    magerr_sys2 = 0.0004
    
    precam_ids = dict()
    for nid, id_string in enumerate(id_strings):
        if len(precam_stars) > 0:
            precam_ids[nid] = precam_stars[0][id_string]
        else:
            precam_ids[nid] = None
    
    stars = []
    star_map = index.Index()
    global_objects = []
    has_precam = 0
    p_vector = None
    
    imgids = dict()
    image_id_dict = dict()
    for nid, id_string in enumerate(id_strings):
        image_id_dict[nid] = dict()
    
    # while we're at it, add stuff to the matrices so we don't have to loop over global objects.
    
    match_radius = 1.0/3600.
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
    
    image_id_count = dict()
    
    
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
    # print "parsed focal plane positions for %d ccds" %len(fp_xs)
    
    
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
        
        # if there weren't any objects for this low-res pixel, then we're done.
        if len(global_objs_list) == 0:
            return None, None, None
        
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
    
    obj_inarea = False
    for go in global_objs_list:
        
        if go.dec < dec_min or go.dec > dec_max:
            continue
        if ra_min > ra_max: # 300 60
            if go.ra > ra_max and go.ra < ra_min:
                continue
        else:
            if go.ra < ra_min or go.ra > ra_max:
                continue

        # cut out objects that are bad quality!
        go.objects = filter(good_quality, go.objects)
        if len(go.objects) < 2:
            continue
        
        obj_inarea = True
        
        for obj in go.objects:
            # if bad quality (or precam), don't use for calibration
            if standard_quality(obj):
                obj['for_mean'] = True
                ndet_for_summi += 1
            else:
                obj['for_mean'] = False
            ndet_for_cal += 1
            
            for nid, id_string in enumerate(id_strings):
                
                if nid not in img_ndets:
                    img_ndets[nid] = dict()
                    img_errs[nid] = dict()
                    worst_ok_err[nid] = dict()
                    imgids[nid] = dict()
                
                image_id = obj[id_string]
                imgids[nid][image_id] = True
                
                if obj[id_string] in img_ndets[nid]:
                    img_ndets[nid][obj[id_string]] += 1
                    if obj['magerr_psf'] < worst_ok_err[nid][obj[id_string]][1]:
                        # replace the current worst error in the list
                        img_errs[nid][obj[id_string]][worst_ok_err[nid][obj[id_string]][0]] = obj['magerr_psf']
                        # now what's the worst?
                        max_index = np.argmax(img_errs[nid][obj[id_string]])
                        worst_ok_err[nid][obj[id_string]] = [max_index, img_errs[nid][obj[id_string]][max_index]]
                else:
                    img_ndets[nid][obj[id_string]] = 0
                    img_errs[nid][obj[id_string]] = np.ones(max_dets) #[obj['magerr_psf']]
                    worst_ok_err[nid][obj[id_string]] = [0,1.0] # index, value
            
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
                for nid, id_string in enumerate(id_strings):
                    precam_id = precam_ids[nid]
                    if nid not in imgids:
                        imgids[nid] = dict()
                        worst_ok_err[nid] = dict()
                    imgids[nid][precam_id] = True
                    worst_ok_err[nid][precam_id] = [0,1.0] # don't cut any precam objects
                precam_count += 1

    if not obj_inarea:
        return None, None, None

    max_nimgs = dict()
    max_nimgs_total = 0
    for nid, id_string in enumerate(id_strings):
        if nid in imgids.keys():
            max_nimgs[nid] = len(imgids[nid].keys())
            max_nimgs_total += max_nimgs[nid]
    img_connectivity = None
    if require_standards:
        if len(id_strings) == 1:
            img_connectivity = np.zeros((max_nimgs[0], max_nimgs[0]), dtype='int')
    
    print "Finished first pass through global objects"
    
    for nid, id_string in enumerate(id_strings):
        n_to_clip = 0
        for i in worst_ok_err[nid].keys():
            if worst_ok_err[nid][i][1] < 1.:
                n_to_clip += 1
        print "%s: %d regions to calibrate, %d with a clipped number of detections" %(id_string, len(worst_ok_err[nid].keys()), n_to_clip)
    
    for go in global_objs_list:
        
        if len(go.objects) < 2:
            continue
        
        if go.dec < dec_min or go.dec > dec_max:
            continue
        if ra_min > ra_max: # 300 60
            if go.ra > ra_max and go.ra < ra_min:
                continue
        else:
            if go.ra < ra_min or go.ra > ra_max:
                continue
                
        # cut out the bad objects for ALL id strings!  (all calibration identifiers)
        for nid,id_string in enumerate(id_strings):
            go.objects = [obj for obj in go.objects if obj['magerr_psf'] < worst_ok_err[nid][obj[id_string]][1]]
        
        if len(go.objects) < 2:
            continue
        
        ndet_for_summi = len([obj for obj in go.objects if obj['for_mean'] == True] )
        ndet_for_cal = len(go.objects)
        ra = go.ra
        dec = go.dec
        
        if ndet_for_summi < 2:
            continue
        
        mags_for_summi = np.zeros(ndet_for_summi)
        mag_errors_for_summi = np.zeros(ndet_for_summi)
        mags_for_cal = np.zeros(ndet_for_cal)
        mag_errors_for_cal = np.zeros(ndet_for_cal)
        image_ids = dict()
        image_id_matrix = dict()
        operands_for_cal = dict()
        for nid,id_string in enumerate(id_strings):
            image_ids[nid] = np.zeros(ndet_for_cal, dtype=np.int)
            image_id_matrix[nid] = np.zeros([max_nimgs[nid],ndet_for_cal], dtype=np.int)
            operands_for_cal[nid] = np.zeros(ndet_for_cal)
        d_for_summi = 0
        d_for_cal = 0
        for star in go.objects:
            
            if star[id_strings[0]] != precam_ids[0]: # if not precam  CHEAP HACK
                star['magerr_psf'] = np.sqrt(star['magerr_psf']*star['magerr_psf'] + magerr_sys2)
            
            mags_for_cal[d_for_cal] = star['mag_psf']
            mag_errors_for_cal[d_for_cal] = star['magerr_psf']
            
            if use_color_terms and not ( use_precam and star['exposureid'] == precam_stars[0]['exposureid'] ):
                if star['exposureid'] not in mag_classes:
                    pwv = color_term_cal.get_pwv(star['mjd'])
                    mag_classes[star['exposureid']] = color_term_cal.color_term_correction(band,pwv,star['airmass'])
                fp_x = star['x_image'] + fp_xs[star['ccd']]
                fp_y = star['x_image'] + fp_ys[star['ccd']]
                fp_r = np.sqrt(fp_x*fp_x + fp_y*fp_y)/14880.1 # the largest fp_r i saw in an example run
                color_term = color_term_cal.fit_color_position(mag_classes[star['exposureid']],go.color,fp_r)
                mags_for_cal[d_for_cal] += color_term
                
            for p in precal:
                [pre_vector, precal_id_string, precal_operand, pre_median] = p
                if star[precal_id_string] in pre_vector:
                    coeff = None
                    if precal_operand == 'None' or precal_operand is None or precal_operand == 1:
                        coeff = 1.
                    else:
                        coeff = star[precal_operand]
                    mags_for_cal[d_for_cal] += pre_vector[star[precal_id_string]]*coeff
                    n_good_e += 1
                else:
                    # mags_for_cal[d_for_cal] += pre_median
                    # if there's no valid precal, then don't accept this!
                    n_bad_e += 1
                    continue
                
            for nid, id_string in enumerate(id_strings):
                if operands[nid] in [ 'None', None, 1 ]:
                    operands_for_cal[nid][d_for_cal] = 1
                else:
                    operands_for_cal[nid][d_for_cal] = star[operands[nid]]
                
                try:
                    image_ids[nid][d_for_cal] = image_id_dict[nid][star[id_string]]
                    image_id_matrix[nid][image_ids[nid][d_for_cal],d_for_cal] = 1
                except KeyError:
                    # the current image id is not in the image_ids dictionary yet
                    if nid not in image_id_count:
                        image_id_count[nid] = 0
                    else:
                        image_id_count[nid] += 1

                    if image_id_count[nid] >= max_nimgs[nid]:
                        print "WARNING, image_id_count = %d >= max_nimgs = %d" %(image_id_count[nid], max_nimgs[nid])

                    image_id_dict[nid][star[id_string]] = image_id_count[nid] # will need to concatenate these to get the indices for p_vector
                    image_ids[nid][d_for_cal] = image_id_dict[nid][star[id_string]]
                    image_id_matrix[nid][image_ids[nid][d_for_cal],d_for_cal] = 1
            
            
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
        
        b_vector = np.append( b_vector, mags_for_cal - sum_m_i )
        
        a_submatrix = None
        for nid,id_string in enumerate(id_strings):
            invsigma_matrix = np.tile(invsigma_array_for_cal*operands_for_cal[nid], (max_nimgs[nid],1))
            # print "invsigma_matrix shape: {0}".format(invsigma_matrix.shape)
            sum_for_zps_nid = (image_id_matrix[nid]*invsigma_matrix).sum(axis=1) / sum_invsigma2_for_cal
            # print "sum_for_zps_nid shape: {0}".format(sum_for_zps_nid.shape)
            a_submatrix_nid = np.tile(sum_for_zps_nid, (ndet_for_cal,1) )
            a_submatrix_nid[range(ndet_for_cal),image_ids[nid][range(ndet_for_cal)]] -= operands_for_cal[nid]
            if a_submatrix is None:
                a_submatrix = a_submatrix_nid
            else:
                a_submatrix = np.hstack((a_submatrix, a_submatrix_nid))
            
        # print a_submatrix.shape
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
        
        # connectivity matrix
        # only for one image_id....
        if require_standards and len(id_strings) == 1:
            for star in go.objects:
                for star2 in go.objects:
                    img_connectivity[image_id_dict[0][star[id_string]], image_id_dict[0][star2[id_string]]] = 1
                
    print "%d good previous zps, %d bad" %(n_good_e, n_bad_e)
    
    n_components = 1
    standard_component = None
    labels = None
    if require_standards:
        if len(id_strings) > 1:
            print "WARNING, not checking disjoing regions because you are using >1 id_string"
        else:
            n_components, labels = cs_graph_components(img_connectivity)
            print "# disjoint regions: %d" %(n_components)
        
        id_string = id_strings[0]
        if n_components > 1:
            print "Warning, %d disjoint regions in the data!!" %n_components
            if precam_ids[0] not in image_id_dict[0]:
                print "No standards in any of the regions!"
                has_precam = False
                p_vector = [0]
                return image_id_dict, p_vector, has_precam
            standard_component = labels[image_id_dict[0][precam_ids[0]]]
            labels[labels == -2] = np.max(labels)+1
            n_w_standards = np.bincount(labels)[standard_component]
            print "%d of %d images will be calibrated" %(n_w_standards, len(labels))
    if( ndets == 0 ):
        # continue
        return image_id_dict, p_vector, has_precam
    
    gcount = len(global_objs_list)
    print "Looped through %d global objects, %d measurements total, %d from precam" %( gcount, ndets, precam_count )
    print "Mean RMS of the stars: %e" %(sum_rms/n_rms)
    
    print "matrix size = {0}".format(matrix_size)
    print "ndets = {0}, max_nimgs_total = {1}".format(ndets,max_nimgs_total)
    print "len a_matrix vectors: {0}".format(len(a_matrix_ys))
    a_matrix = coo_matrix((a_matrix_vals[0:matrix_size], (a_matrix_ys[0:matrix_size], a_matrix_xs[0:matrix_size])), shape=(ndets,max_nimgs_total))

    c_matrix = lil_matrix((ndets,ndets))
    c_matrix.setdiag(1.0/c_vector)
    
    # print a_matrix
    # print b_vector

    # print "Calculating intermediate matrices..."
    subterm = (a_matrix.transpose()).dot(c_matrix)
    termA = subterm.dot(a_matrix)
    termB = subterm.dot(b_vector)
    # print "Solving!"
    # p_vector = linalg.bicgstab(termA,termB)[0]
    # p_vector = linalg.spsolve(termA,termB)
    p_vector = linalg.minres(termA,termB)[0]
    
    
    # normalize to the precam standards
    if use_precam:
        p_base = 0
        for nid, id_string in enumerate(id_strings):
            # only normalize the additive zero points...  not ideal, but not sure what's better.
            if operands[nid] not in [ 'None', None, 1 ]:
                continue
            # deal with the indices for each id_string separately
            p_vector_indices = image_id_dict[nid].values()
            if precam_ids[nid] in image_id_dict[nid]:
                print "%s: Normalizing to the PreCam standards!" %id_string
                p_vector[p_vector_indices] -= p_vector[p_base+image_id_dict[nid][precam_id]]
                has_precam = 1
            else:
                median = np.median(p_vector)
                print "%s: Normalizing to the median of the zeropoints: %e" %(id_string, median)
                p_vector[p_vector_indices] -= median
            p_base += max_nimgs[nid]
    
    print "Solution:"
    print p_vector.shape
    print p_vector
    
    if len(id_strings) == 1:
        if n_components > 1:
            for p in range(len(p_vector)):
                if labels[p] != standard_component:
                    p_vector[p] = np.nan
    
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
        
        if go.dec < dec_min or go.dec > dec_max:
            continue
        if ra_min > ra_max: # 300 60
            if go.ra > ra_max and go.ra < ra_min:
                continue
        else:
            if go.ra < ra_min or go.ra > ra_max:
                continue
        
        mags = np.zeros(len(go.objects))
        mag_errors = np.zeros(len(go.objects))
        mags_formean = np.zeros(len([obj for obj in go.objects if obj['for_mean']]))
        mag_errors_formean = np.zeros(len(mags_formean))
        i=0
        i2=0
        bad = dict()
        for nid,id_string in enumerate(id_strings):
            bad[nid] = dict()
        
        for star in go.objects:
            
            mags[i] = star['mag_psf']
            mag_errors[i] = star['magerr_psf']
            if star['for_mean']:
                mags_formean[i2] = star['mag_psf']
                mag_errors_formean[i2] = star['magerr_psf']
            
            try:
                p_base = 0
                for nid,id_string in enumerate(id_strings):
                    if (p_vector[p_base+image_id_dict[nid][star[id_string]]] is None) or np.isnan(p_vector[p_base+image_id_dict[nid][star[id_string]]]):
                        bad[nid][star[id_string]] = True
                        p_base += max_nimgs[nid]
                        continue
                    coeff = None
                    if operands[nid] in [ 'None', None, 1 ]:
                        coeff = 1.
                    else:
                        coeff = star[operands[nid]]
                    mags[i] += p_vector[p_base+image_id_dict[nid][star[id_string]]]*coeff
                    if star['for_mean']:
                        mags_formean[i2] += p_vector[p_base+image_id_dict[nid][star[id_string]]]*coeff
                    p_base += max_nimgs[nid]
                    
            except KeyError:
                bad[nid][star[id_string]] = True
                
            for p in precal:
                [pre_vector, precal_id_string, precal_operand, pre_median] = p
                coeff = None
                if precal_operand in [ 'None', None, 1 ]:
                    coeff = 1.
                else:
                    coeff = star[precal_operand]
                if star[precal_id_string] in pre_vector:
                    n_good_e += 1
                    mags[i] += pre_vector[star[precal_id_string]]*coeff
                    if star['for_mean']:
                        mags_formean[i2] += pre_vector[star[precal_id_string]]*coeff
                else:
                    n_bad_e += 1
                    mags[i] += pre_median*coeff
                    if star['for_mean']:
                        mags_formean[i2] += pre_median*coeff
                        
            if star['for_mean']:
                i2 += 1
            i += 1
            
        if i>2:
            mags = mags[0:i]
            mag_errors = mag_errors[0:i]
            invsigma_array = 1.0/np.square(mag_errors)
            sum_invsigma2 = invsigma_array.sum()
            sum_m_i = (mags*invsigma_array).sum() / sum_invsigma2
            sum_rms += np.std(mags-sum_m_i)
            n_rms += 1
        if i2>2:
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
    
    # split p_vector into the different n_id sections
    p_vectors = []
    p_base = 0
    for nid, id_string in enumerate(id_strings):
        p_vectors.append(p_vector[p_base:p_base+max_nimgs[nid]])
        p_base += max_nimgs[nid]
        
    return image_id_dict, p_vectors, has_precam



def nebencalibrate( band, nside, nside_file, id_strings, operands, precam_map, precam_stars, precal, globals_dir, require_standards=False, max_dets=1e9, ra_min=-360., ra_max=361., dec_min=91., dec_max=91., use_color_terms=False, mag_classes={} ):
    
    if len(precam_stars) == 0 and require_standards:
        print "nebencalibrate: No standards, yet you are requiring standards!"
        raise Exception
    
    npix = 1
    if nside>0:
        npix = healpy.nside2npix(nside)
    
    precam_ids = dict()
    if len(precam_stars) > 0:
        for nid,id_string in enumerate(id_strings):
            precam_ids[nid] = precam_stars[0][id_string]
    else:
        for nid,id_string in enumerate(id_strings):
            precam_ids[nid] = None
    
    p_vectors = []
    image_id_dicts = []
    for pix in range(npix):
        p_vectors.append(None)
        image_id_dicts.append(dict())
        
    # nebencalibrate the pixels!
    has_precam = dict()
    pix_wobjs = []
    inputs= []
    for p in range(npix):
        inputs.append( [p, nside, nside_file, band, precal, precam_stars, precam_map, globals_dir, id_strings, operands, max_dets, require_standards, ra_min, ra_max, dec_min, dec_max, use_color_terms, mag_classes] )
        image_id_dicts[p], p_vectors[p], has_precam[p] = nebencalibrate_pixel(inputs[p])
        if require_standards and not has_precam[p] and p_vectors[p] is not None:
            print "No standards found in pixel %d for nside=%d.  Degrading!" %(p,nside)
            return 'degrade'
        if p_vectors[p] is not None:
            pix_wobjs.append(p)
    npix_wobjs = len(pix_wobjs)
    
    # now do another ubercalibration to make sure the pixels are normalized to each other
    # the measurements are now the difference between two zps calculated for the same image, during different pixels' ubercalibrations.
    
    # If there is only one pixel with objects, let's just save the ZPs and be done.
    zeropoints_tot = []
    if npix_wobjs == 1:
        for nid,id_string in enumerate(id_strings):
            zeropoints = dict()
            imagelist = []
            for pix in range(npix):
                if image_id_dicts[pix][nid] is not None:
                    imagelist.extend(image_id_dicts[pix][nid].keys())
            imagelist = list(set(imagelist))
            print "%s: There are %d images in total, from %d pixels." %(id_string, len(imagelist), npix_wobjs)
            
            for imageid in imagelist:
                pix = pix_wobjs[0]
                if (imageid in image_id_dicts[pix][nid]) and (p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]] is not None) and (not np.isnan(p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]])):
                    zeropoints[imageid] = p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]]
            zeropoints_tot.append(zeropoints)
            
        return zeropoints_tot
    
    
    # do this only for the first of the id_strings (TBD: do it for each?)
    id_string = id_strings[0]
    nid = 0
    
    # first, get a superlist of image ids
    imagelist = []
    for pix in range(npix):
        if image_id_dicts[pix] is not None:
            imagelist.extend(image_id_dicts[pix][nid].keys())
    imagelist = list(set(imagelist))
    print "There are %d images in total, from %d pixels." %(len(imagelist), npix_wobjs)
        
    # if there was more than one pixel, we need to do the overall ubercal.
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
            if not nid in image_id_dicts[pix]:
                continue
            if (imageid in image_id_dicts[pix][nid]) and (imageid == precam_ids[nid] or p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]] != 0.) and (p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]] is not None) and not np.isnan(p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]]):
                zps.append(p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]])
                
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
            print "Image %d has no zps" %imageid
            continue

        pix_id_matrix2 = pix_id_matrix2[:,0:nzps]
        
        zps = np.array(zps)
        zp_errors = np.ones(nzps)
        invsigma_array = 1.0/np.square(zp_errors)
        sum_invsigma2 = invsigma_array.sum()
        sum_zps_i = (zps*invsigma_array).sum() / sum_invsigma2
        
        invsigma_matrix = np.tile(invsigma_array, (npix_wobjs,1))
        sum_for_zps = (pix_id_matrix2*invsigma_matrix).sum(axis=1) / sum_invsigma2
        
        a_submatrix = np.tile(sum_for_zps, (nzps,1) )
        a_submatrix[range(nzps),pix_ids2[range(nzps)]] -= 1.0
        a_submatrix = coo_matrix(a_submatrix)

        indices = np.where(a_submatrix.data != 0.)[0]
        if len(indices) == 0:
            continue

        while( matrix_size+len(indices) > max_matrix_size ):
            a_matrix_xs = np.hstack((a_matrix_xs,np.zeros(max_matrix_size)))
            a_matrix_ys = np.hstack((a_matrix_ys,np.zeros(max_matrix_size)))
            a_matrix_vals = np.hstack((a_matrix_vals,np.zeros(max_matrix_size)))
            max_matrix_size += max_matrix_size

        a_matrix_xs[matrix_size:(matrix_size+len(indices))] = a_submatrix.col[indices]
        a_matrix_ys[matrix_size:(matrix_size+len(indices))] = a_submatrix.row[indices]+nzps_tot
        a_matrix_vals[matrix_size:(matrix_size+len(indices))] = a_submatrix.data[indices]
        matrix_size += len(indices)

        b_vector = np.append( b_vector, zps - sum_zps_i )
        c_vector = np.append(c_vector, invsigma_array)
        
        # add up some stats
        sum_rms += np.std(zps-sum_zps_i)
        n_rms += 1
        nzps_tot += nzps
        
    print "Looped through the images"
    print "Mean RMS of the zps: %e with %d images with zps" %(sum_rms/n_rms, n_rms)
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
            if imageid in image_id_dicts[pix][nid] and pix_id in pix_id_dict2 and not np.isnan(p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]]):
                zps.append(p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]]+p1_vector[pix_id_dict2[pix_id]])
        
        sum_rms += np.std(zps)
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
    id_string = id_strings[0]
    for imageid in imagelist:
        zp_tot = 0.
        zp_n = 0
        for pix_id in range(npix_wobjs):
            pix = pix_wobjs[pix_id]
            if imageid in image_id_dicts[pix][nid] and pix_id in pix_id_dict2 and not np.isnan(p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]]):
                zp_tot += p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]] + p1_vector[pix_id_dict2[pix_id]] 
                zp_n += 1
        if zp_n > 0:
            zp_tot /= zp_n
            zeropoints[imageid] = zp_tot
    zeropoints_tot.append(zeropoints)
    
    # are there more id_strings?
    if len(id_strings)>1:
        for nid,id_string in enumerate(id_strings[1:]):
            nid1 = nid+1 # because enumerate starts from zero
            zeropoints = dict()
            imagelist1 = []
            for pix in range(npix):
                if image_id_dicts[pix] is not None:
                    imagelist1.extend(image_id_dicts[pix][nid1].keys())
            imagelist1 = list(set(imagelist1))
            for imageid in imagelist1: #image_id_dicts[pix][id_string].keys():
                zp_tot = 0.
                zp_n = 0
                for pix_id in range(npix_wobjs):
                    pix = pix_wobjs[pix_id]
                    if imageid in image_id_dicts[pix][nid1] and not np.isnan(p_vectors[pix][nid1][image_id_dicts[pix][nid1][imageid]]):
                        zp_tot += p_vectors[pix][nid1][image_id_dicts[pix][nid1][imageid]]
                        zp_n += 1
                if zp_n > 0:
                    zp_tot /= zp_n
                zeropoints[imageid] = zp_tot
            zeropoints_tot.append(zeropoints)
    
    return zeropoints_tot


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
        read_precam( precam_stars, precam_map, config['general']['precam_filename'], band )
    if config['general']['use_sdss']:
        read_sdss( precam_stars, precam_map, config['general']['sdss_filename'], band )
    if config['general']['use_betoule']:
        read_betoule( precam_stars, precam_map, config['general']['betoule_filename'], band )
    
    
    # calibrate!
    
    # use the zp_phots as a starting point to our calibration
    precal = [[zp_phots, 'ccd', 1, np.median(zp_phots.values())]]
    
    # initialize the color term structure, which will be filled up as needed during the calibration
    colorterm_class = {}
    
    for calibration in config['calibrations']:
        print "\nStarting calibration on ids %s, operands %s\n" %(calibration['id_strings'], calibration['operands'])
        standard_stars = []
        standard_map = None
        if calibration['use_standards']:
            standard_stars = precam_stars
            standard_map = precam_map
        new_zps = None
        success = False
        while not success:
            new_zps = nebencalibrate( band, calibration['nside'], config['general']['nside_file'], calibration['id_strings'], calibration['operands'], standard_map, standard_stars, precal, config['general']['globals_dir'], require_standards=calibration['require_standards'], max_dets=calibration['max_dets'], ra_min=config['general']['ra_min'], ra_max=config['general']['ra_max'], dec_min=config['general']['dec_min'], dec_max=config['general']['dec_max'], use_color_terms=config['general']['use_color_terms'], mag_classes=colorterm_class )
            if new_zps == 'degrade':
                if calibration['nside'] == 1:
                    calibration['nside'] = 0
                elif calibration['nside'] == 0:
                    print "Giving up!"
                    exit(1)
                calibration['nside'] /= 2
                print "Now trying nside=%d" %calibration['nside']
            else:
                success = True

        for i in range(len(calibration['id_strings'])):
            outfile = open( calibration['outfilenames'][i], 'w' )
            for zp_id in new_zps[i].keys():
                outfile.write( "%d %e\n" %(zp_id, new_zps[i][zp_id]) );
            outfile.close()
        
        # so that we use these as input to the following calibration(s)
        for i in range(len(calibration['id_strings'])):
            precal.append([new_zps[i], calibration['id_strings'][i], calibration['operands'][i], np.median(new_zps[i].values())])
        

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


