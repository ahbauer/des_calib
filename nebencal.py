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
src_dir = '/Users/bauer/software/python'
sys.path.append(src_dir)
from pyspherematch import spherematch

"""
ubercal.py

Read from the global_objects table, generate a sparse matrix equation that yields zero points 
that minimize the mag difference between measurements of the same objects at different times.  
Solve the equation, save the ZPs back into the database.

For now, make one ZP per image.  Can be generalized to solve for spatial dependence, non-linearity.
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

def make_exp_global_objs( filename, band, precam_stars ):
    
    print "Making global objects per exposure"
    
    max_objs_per_image = 50 # so that in super-dense areas like the LMC we don't spend ages, but instead use a subset of objects.
    
    table = None
    h5file = tables.openFile(filename, "r")
    if band == 'u':
        table = h5file.root.data.table_u
    elif band == 'g':
        table = h5file.root.data.table_g
    elif band == 'r':
        table = h5file.root.data.table_r
    elif band == 'i':
        table = h5file.root.data.table_i
    elif band == 'z':
        table = h5file.root.data.table_z
    elif band == 'y':
        table = h5file.root.data.table_y
    else:
        print "Hm, band = %s" %band
        exit(1)
    
    stars_by_img = dict()
    stars_by_exp = dict()
    for star in table.iterrows():
        
        if not good_quality(star):
            continue
        
        star2 = dict()
        star2['ra'] = star['ra']
        star2['dec'] = star['dec']
        star2['band'] = star['band']
        star2['mag_psf'] = star['mag_psf']
        star2['magerr_psf'] = star['magerr_psf']
        star2['x_image'] = star['x_image']
        star2['y_image'] = star['y_image']
        star2['image_id'] = star['imageid']
        star2['exposureid'] = star['exposureid']
        star2['ccd'] = star['ccd']
        star2['matched'] = False
        star2['count'] = 0
        star2['gskyphot'] = star['gskyphot']
        star2['lskyphot'] = star['lskyphot']
        star2['gskyhot'] = star['gskyhot']
        star2['lskyhot'] = star['lskyhot']
        
        if not star['imageid'] in stars_by_img:
            stars_by_img[star['imageid']] = []
        stars_by_img[star['imageid']].append(star2)
    
    count = 0
    for img in stars_by_img.keys():
        star_list = sorted(stars_by_img[img], key=itemgetter('magerr_psf'))
        star_list = star_list[0:max_objs_per_image]
        if not star_list[0]['exposureid'] in stars_by_exp:
            stars_by_exp[star_list[0]['exposureid']] = []
        stars_by_exp[star_list[0]['exposureid']].extend(star_list)
        count+=len(star_list)
    print "Parsed %d stars" %(count)
    
    if not 1 in stars_by_exp:
        stars_by_exp[1] = precam_stars
    else:
        stars_by_exp[1].extend(precam_stars)
    print "Added %d precam stars" %len(precam_stars)
    
    match_radius = 1.0/3600.
    gos = []
    exposures = stars_by_exp.keys()
    ne = len(exposures)
    print "%d exposures" %ne
    for e1 in range(ne):
        
        print " {0}/{1}\r".format(e1, ne),
        
        exposure1 = exposures[e1]
        global_list = dict()
        star1_ras = [o['ra'] for o in stars_by_exp[exposure1] if not o['matched']]
        star1_decs = [o['dec'] for o in stars_by_exp[exposure1] if not o['matched']]
        star1_indices = [o for o in range(len(stars_by_exp[exposure1])) if not stars_by_exp[exposure1][o]['matched']]
        for e2 in range(e1+1,ne):
            exposure2 = exposures[e2]
            
            star2_ras = [o['ra'] for o in stars_by_exp[exposure2] if not o['matched']]
            star2_decs = [o['dec'] for o in stars_by_exp[exposure2] if not o['matched']]
            star2_indices = [o for o in range(len(stars_by_exp[exposure2])) if not stars_by_exp[exposure2][o]['matched']]
            
            
            if len(star1_ras) == 0 or len(star2_ras) == 0:
                continue
            
            inds1, inds2, dists = spherematch( star1_ras, star1_decs, star2_ras, star2_decs, tol=match_radius )
            
            
            if len(inds1) < 2:
                continue
            
            for i in range(len(inds1)):
                try:
                    global_list[star1_indices[inds1[i]]].append(stars_by_exp[exposure2][star2_indices[inds2[i]]])
                    stars_by_exp[exposure2][star2_indices[inds2[i]]]['matched'] = True
                except:
                    global_list[star1_indices[inds1[i]]] = []
                    global_list[star1_indices[inds1[i]]].append(stars_by_exp[exposure1][star1_indices[inds1[i]]])
                    global_list[star1_indices[inds1[i]]].append(stars_by_exp[exposure2][star2_indices[inds2[i]]])
                    stars_by_exp[exposure1][star1_indices[inds1[i]]]['matched'] = True
                    stars_by_exp[exposure2][star2_indices[inds2[i]]]['matched'] = True
        

        # ok, now we have a set of global objects seen in exposure1
        for index in global_list.keys():
            objects = global_list[index]
            
            ra_mean = np.mean([star['ra'] for star in objects])
            dec_mean = np.mean([star['dec'] for star in objects])
            
            go = global_object()
            go.ra = ra_mean
            go.dec = dec_mean
            go.objects = objects
            gos.append(go)
                
    outfile = open( 'gobs_by_exp_' + band, 'wb' )
    cPickle.dump(gos, outfile, cPickle.HIGHEST_PROTOCOL)
    outfile.close()
    print "Wrote %d global objects by exposure" %(len(gos))
    
    h5file.close()
        
        

def ubercalibrate_exposure( filename, band, zp_phots, precam_stars ):
    
    redo_matches = False
    
    match_radius = 1.0/3600.
    max_nepochs = 50 # ick!
    magerr_sys2 = 0.0004
    
    max_for_expcal = 75 # max objs per image
    
    count = 0
    expids = dict()
    ras = dict() # by exposure
    decs = dict()
    image_ids = dict()
    errors = dict()
    indices = dict()
    image_exposure = dict()
    stars = []
    exp_id_count = -1
    exp_id_dict = dict()
    ndets = 0
    b_vector = []
    a_matrix = None
    c_vector = []
    global_objs_list = []
    n_global_objs = 0
    
    if redo_matches:
        make_exp_global_objs( filename, band, precam_stars )
    
    print "Reading from global file"
    matchfile = open('gobs_by_exp_' + band, 'rb' )
    global_objs_list = cPickle.load(matchfile)
    matchfile.close()
    n_global_objs = len(global_objs_list)
    
    for go in global_objs_list:
        for star2 in go.objects:
            image_exposure[star2['image_id']] = star2['exposureid']
            expids[star2['exposureid']] = True
    max_nexps = len(expids.keys())
    print "Read %d global objects from file, from %d exposures" %(n_global_objs, max_nexps)
    
    exposures = expids.keys()
    
    count = 0
    sum_rms = 0.
    n_rms = 0
    matrix_size = 0
    max_matrix_size = 5*n_global_objs #(len(stars)/max_nexps)*max_nepochs
    a_matrix_xs = np.zeros(max_matrix_size)
    a_matrix_ys = np.zeros(max_matrix_size)
    a_matrix_vals = np.zeros(max_matrix_size)
    
    p_vector = np.zeros(max_nexps)
    sum_for_zps = np.zeros(max_nexps)
    
    
    print "Starting loop over global objs"
    for go in global_objs_list:
        
        ndet_for_cal = len(go.objects)
        ndet_for_summi = len([obj for obj in go.objects if obj['image_id'] != 1])
        
        if ndet_for_summi < 2:
            continue
            
        # ok, now we have a "global object"
        count += 1
        
        if not count%1000:
            print " {0}/{1}\r".format(count, n_global_objs),
            
        mags_for_summi = np.zeros(ndet_for_summi)
        mags_for_cal = np.zeros(ndet_for_cal)
        mag_errors_for_summi = np.zeros(ndet_for_summi)
        mag_errors_for_cal = np.zeros(ndet_for_cal)
        exp_ids = np.zeros(ndet_for_cal, dtype=np.int)
        exp_id_matrix = np.zeros([max_nexps,ndet_for_cal], dtype=np.int)
        d_for_cal=0
        d_for_summi = 0
        # print "for object %f %f" %(star['ra'], star['dec'])
        for star2 in go.objects:
            # add up matrix info
            try:
                exp_ids[d_for_cal] = exp_id_dict[star2['exposureid']]
                exp_id_matrix[exp_ids[d_for_cal],d_for_cal] = 1
            except KeyError:
                # the current image id is not in the image_ids dictionary yet
                exp_id_count += 1
                exp_id_dict[star2['exposureid']] = exp_id_count
                exp_ids[d_for_cal] = exp_id_dict[star2['exposureid']]
                exp_id_matrix[exp_ids[d_for_cal],d_for_cal] = 1
            
            if star2['exposureid'] != 1: # if not precam  CHEAP HACK
                star2['magerr_psf'] = np.sqrt(star2['magerr_psf']*star2['magerr_psf'] + magerr_sys2)
            
            mags_for_cal[d_for_cal] = star2['mag_psf'] + zp_phots[star2['ccd']]
            mag_errors_for_cal[d_for_cal] = star2['magerr_psf']
            d_for_cal += 1
            # if this isn't precam, add it to the sums for the mean mag
            if star2['exposureid'] != 1:
                mag_errors_for_summi[d_for_summi] = star2['magerr_psf']
                mags_for_summi[d_for_summi] = star2['mag_psf'] + zp_phots[star2['ccd']]
                d_for_summi += 1
        if( ndet_for_cal != d_for_cal ):
            print "hm, cal ndet = %d, d = %d" %(ndet_for_cal, d_for_cal)
            exit(1)
        if( ndet_for_summi != d_for_summi ):
            print "hm, summi ndet = %d, d = %d" %(ndet_for_summi, d_for_summi)
            exit(1)
        
        invsigma_array_for_cal = 1.0/np.square(mag_errors_for_cal)
        invsigma_array_for_summi = 1.0/np.square(mag_errors_for_summi)
        sum_invsigma2_for_summi = invsigma_array_for_summi.sum()
        sum_invsigma2_for_cal = invsigma_array_for_cal.sum()
        sum_m_i = (mags_for_summi*invsigma_array_for_summi).sum() / sum_invsigma2_for_summi
        
        invsigma_matrix = np.tile(invsigma_array_for_cal, (max_nexps,1))
        sum_for_zps = (exp_id_matrix*invsigma_matrix).sum(axis=1) / sum_invsigma2_for_cal
        b_vector = np.append( b_vector, mags_for_cal - sum_m_i )
        
        a_submatrix = np.tile(sum_for_zps, (ndet_for_cal,1) )
        a_submatrix[range(ndet_for_cal),exp_ids[range(ndet_for_cal)]] -= 1.0
        a_submatrix = coo_matrix(a_submatrix)
        
        indicesa = np.where(a_submatrix.data != 0.)[0]
        if( matrix_size+len(indicesa) > max_matrix_size ):
            print "enlarging max_matrix_size from %d" %max_matrix_size
            a_matrix_xs = np.hstack((a_matrix_xs,np.zeros(max_matrix_size)))
            a_matrix_ys = np.hstack((a_matrix_ys,np.zeros(max_matrix_size)))
            a_matrix_vals = np.hstack((a_matrix_vals,np.zeros(max_matrix_size)))
            max_matrix_size += max_matrix_size
        a_matrix_xs[matrix_size:(matrix_size+len(indicesa))] = a_submatrix.col[indicesa]
        a_matrix_ys[matrix_size:(matrix_size+len(indicesa))] = a_submatrix.row[indicesa]+ndets
        a_matrix_vals[matrix_size:(matrix_size+len(indicesa))] = a_submatrix.data[indicesa]
        matrix_size += len(indicesa)
        
        c_vector = np.append(c_vector, invsigma_array_for_cal)
        
        # add up some stats
        sum_rms += np.std(mags_for_summi-sum_m_i)
        n_rms += 1
        ndets += ndet_for_cal
        
        
    print "Looped through %d global objects, %d measurements total" %( count, ndets )
    print "Mean RMS of the stars: %e" %(sum_rms/n_rms)
    
    
    a_matrix = coo_matrix((a_matrix_vals[0:matrix_size], (a_matrix_ys[0:matrix_size], a_matrix_xs[0:matrix_size])), shape=(ndets,max_nexps))
    
    c_matrix = lil_matrix((ndets,ndets))
    c_matrix.setdiag(1.0/c_vector)
    
    print "Calculating intermediate matrices..."
    subterm = (a_matrix.transpose()).dot(c_matrix)
    termA = subterm.dot(a_matrix)
    termB = subterm.dot(b_vector)
    print"Solving!"
    # p_vector = linalg.bicgstab(termA,termB)[0]
    # p_vector = linalg.spsolve(termA,termB)
    p_vector = linalg.minres(termA,termB)[0]
    print "Solution:"
    print p_vector
    
    
    # now calculate how we did
    sum_rms = 0.
    n_rms = 0
    for go in global_objs_list:
        mags = np.zeros(len(go.objects))
        mag_errors = np.zeros(len(go.objects))
        for i,star in enumerate(go.objects):
            mags[i] = star['mag_psf'] + zp_phots[star['ccd']] + p_vector[exp_id_dict[star['exposureid']]]
            mag_errors[i] = star['magerr_psf']
        invsigma_array = 1.0/np.square(mag_errors)
        sum_invsigma2 = invsigma_array.sum()
        sum_m_i = (mags*invsigma_array).sum() / sum_invsigma2
        sum_rms += np.std(mags-sum_m_i)
        n_rms += 1
    
    print "Mean RMS after exposure calibration: %e" %(sum_rms/n_rms)
    
    
    # renormalize so that the precam zp is zero
    if 1 in exp_id_dict:
        precam_zp = p_vector[exp_id_dict[1]]
        for e in range(len(p_vector)):
            p_vector[e] -= precam_zp
    
    
    # write out the solution
    outfile = open( 'nebencal_exposurezps_' + band, 'w' )
    for exposureid in exposures:
        if exposureid in exp_id_dict:
            outfile.write( "%d %e\n" %(exposureid, p_vector[exp_id_dict[exposureid]]) );
    outfile.close()
    
    outfile2 = open( 'nebencal_exp_output_' + band, 'wb' )
    cPickle.dump([image_exposure, exp_id_dict, p_vector], outfile2, cPickle.HIGHEST_PROTOCOL)
    outfile2.close()
    
    return image_exposure, exp_id_dict, p_vector



def nebencalibrate_pixel( inputs ):
    
    [p, nside, band, pix_wobjs, exp_id_dict, e_vector, precam_stars, precam_map, zp_phots, globals_dir] = inputs
    
    pix = pix_wobjs[p]
    
    use_precam = 0
    if len(precam_stars) > 0:
        use_precam = 1

    max_nepochs = 50 # ick!
    magerr_sys2 = 0.0004
    
    e_median = 0.
    if e_vector is not None:
        e_median = np.median(e_vector)
    
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
    # filelines = file.readlines()
    # max_matrix_size = len(filelines)*max_nepochs
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
            image_id = obj['image_id'] #int(entries[i])
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
            if star['image_id'] != 1: # if not precam  CHEAP HACK
                star['magerr_psf'] = np.sqrt(star['magerr_psf']*star['magerr_psf'] + magerr_sys2)
            try:
                image_ids[d_for_cal] = image_id_dict[star['image_id']]
                image_id_matrix[image_ids[d_for_cal],d_for_cal] = 1
                image_ras[star['image_id']] += star['ra']
                image_decs[star['image_id']] += star['dec']
                image_ns[star['image_id']] += 1
            except KeyError:
                # the current image id is not in the image_ids dictionary yet
                image_id_count += 1

                if image_id_count >= max_nimgs:
                    print "WARNING, image_id_count = %d > max_nimgs = %d" %(image_id_count, max_nimgs)

                image_id_dict[star['image_id']] = image_id_count
                image_ids[d_for_cal] = image_id_dict[star['image_id']]
                image_id_matrix[image_ids[d_for_cal],d_for_cal] = 1
                image_ras[star['image_id']] = star['ra']
                image_decs[star['image_id']] = star['dec']
                image_ns[star['image_id']] = 1
            
            mags_for_cal[d_for_cal] = star['mag_psf'] + zp_phots[star['ccd']]
            
            if e_vector is not None:
                if star['exposureid'] in exp_id_dict:
                    mags_for_cal[d_for_cal] += e_vector[exp_id_dict[star['exposureid']]]
                    n_good_e += 1
                else:
                    mags_for_cal[d_for_cal] += e_median
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
    print "%d good exposure zps, %d bad" %(n_good_e, n_bad_e)
    
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
            # p_vector[pix] -= p_vector[pix][image_id_dict[pix][1]]
            p_vector -= p_vector[image_id_dict[1]]
            has_precam = 1 # [pix] = 1

    print "Solution:"
    print p_vector #[pix]
    
    
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
                mags[i] = star['mag_psf'] + zp_phots[star['ccd']] + p_vector[image_id_dict[star['image_id']]]
                mag_errors[i] = star['magerr_psf']
                if star['for_mean']:
                    mags_formean[i2] = star['mag_psf'] + zp_phots[star['ccd']] + p_vector[image_id_dict[star['image_id']]]
                    mag_errors_formean[i2] = star['magerr_psf']
                if e_vector is not None:
                    if star['exposureid'] in exp_id_dict:
                        n_good_e += 1
                        mags[i] += e_vector[exp_id_dict[star['exposureid']]]
                        if star['for_mean']:
                            mags_formean[i2] += e_vector[exp_id_dict[star['exposureid']]]
                            mag_errors_formean[i2] = star['magerr_psf']
                    else:
                        n_bad_e += 1
                        mags[i] += e_median
                        if star['for_mean']:
                            mags_formean[i2] += e_median
                if star['for_mean']:
                    i2 += 1
                i += 1
            except KeyError:
                bad[star['image_id']] = True
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



def nebencalibrate_by_filter(filename, band, makestats=True):
    
    
    mag_type = 'psf'
    flux_name = "flux_" + "%s" %mag_type
    flux_err_name = "flux_err_" + "%s" %mag_type
    
    use_precam = True

    recalibrate_by_exposure = True
    read_expzps = False

    max_nims = 1e6 # ick!
    nside = 32 # 2: 30 degrees per side, 4: 15 degrees per side, 8:  7.3 degrees per side, 16 = 3.7 degrees, 32 = 1.8 degrees
    max_objs_per_image = 500 # so that in super-dense areas like the LMC we don't spend ages, but instead use a subset of objects.
    
    use_global_objs = True
    globals_dir = '/Users/bauer/surveys/DES/y1p1/equatorial'
    use_hdf5 = True
    
    npix = healpy.nside2npix(nside)
    pixelMap = np.arange(npix)

    n_ccds = 63
    n_xspx = 8
    n_yspx = 16
    spxsize = 256
    p_vector = []
    image_id_dict = []
    for pix in range(npix):
        p_vector.append(None)
        image_id_dict.append(dict())
    

    precam_stars = []
    precam_map = index.Index()
    if use_precam:
        # read in the precam standards and make an index
        precam_path = "/Users/bauer/surveys/DES/precam/PreCamStandarStars"
        precam_name = band.upper() + ".Stand1percent.s"
        precam_file = open( os.path.join(precam_path, precam_name), 'r' )
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

        sdssfile = open("/Users/bauer/surveys/DES/y1p1/equatorial/sdss/SDSSDR10_SouthGalCap/stripe82_sample1.csv", 'r')
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
        print "Now %d including SDSS objects acting as more PreCam" %count


    # read in the nominal ccd offsets (zp_phots), which will be our starting point
    zp_phot_filename = "/Users/bauer/surveys/DES/zp_phots/" + band + ".dat"
    zp_phot_file = open( zp_phot_filename, 'r' )
    zp_phots = np.zeros(n_ccds+1)
    for line in zp_phot_file:
        entries = line.split()
        if entries[0][0] == '#':
            continue
        zp_phots[int(entries[0])] = -1.0*float(entries[4])
    # include precam fake entry
    zp_phots[0] = 0.
    
    
    exp_id_dict = None
    e_vector = None
    image_exposure = None
    if recalibrate_by_exposure:
        hdf5_filename = os.path.join(globals_dir, 'finalcutout.h5')
        image_exposure, exp_id_dict, e_vector = ubercalibrate_exposure( hdf5_filename, band, zp_phots, precam_stars )
    elif read_expzps:
        expfile = open('nebencal_exp_output_' + band, 'rb' )
        [image_exposure, exp_id_dict, e_vector] = cPickle.load(expfile)
        expfile.close()
        print "exp_id_dict has %d keys, e.g. %d: %d, %d: %d" %(len(exp_id_dict.keys()), exp_id_dict.keys()[0], exp_id_dict[exp_id_dict.keys()[0]], exp_id_dict.keys()[1], exp_id_dict[exp_id_dict.keys()[1]])

    e_median = 0.
    if e_vector is not None:
        e_median = np.median(e_vector)

    
    # first let's just see which pixels are populated so that we can loop over only those.
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

    has_precam = dict()

    inputs= []
    for p in range(npix_wobjs):
        inputs.append( [p, nside, band, pix_wobjs, exp_id_dict, e_vector, precam_stars, precam_map, zp_phots, globals_dir] )
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
        pix_ids2 = np.zeros(npix_wobjs, dtype=np.int) # *len(imagelist)

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

#            a_submatrix = np.tile(sum_for_zps, (ndet,1) )
#            a_submatrix[range(ndet),image_ids[range(ndet)]] -= 1.0
#            a_submatrix = coo_matrix(a_submatrix)

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
    if use_precam:
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
    
    outfilename = 'nebencal_zps_' + band
    outfile = open( outfilename, 'w' )
    for imageid in imagelist:
        zp_tot = 0.
        zp_n = 0
        for pix_id in range(npix_wobjs):
            pix = pix_wobjs[pix_id]
            if imageid in image_id_dict[pix]:
                zp_tot += p_vector[pix][image_id_dict[pix][imageid]] + p1_vector[pix_id_dict2[pix_id]] 
                # if e_vector is not None:
                #     if imageid in image_exposure:
                #         exposureid = image_exposure[imageid]
                #         if exposureid in exp_id_dict:
                #             zp_tot += e_vector[exp_id_dict[exposureid]]
                #         else:
                #             zp_tot += e_median
                zp_n += 1

        zp_tot /= zp_n
        outfile.write( "%d %e\n" %(imageid, zp_tot) );
    outfile.close()



def main():
    filt = 'g'
    print "Running nebencalibration for DES, filter %s!" %filt
    
    filename = None # "/Users/bauer/surveys/DES/sva1/finalcut/finalcutout.h5"
    # filename = "/Users/bauer/surveys/DES/sva1/finalcut/finalcutout_subsetimg.h5"
    # filename = "/Users/bauer/surveys/DES/sva1/finalcut/test"
    nebencalibrate_by_filter( filename, filt )



if __name__ == '__main__':
    main()


