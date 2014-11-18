import sys
import os
import math
import re
import copy
from operator import itemgetter
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
from nebencal_utils import read_precam
from nebencal_utils import read_sdss
from nebencal_utils import read_tertiaries
from nebencal_utils import global_object
from nebencal_utils import read_global_objs
from nebencal_utils import read_ccdpos
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

class Matrix_Inputs(object):
    
    def __init__(self, mms):
        self.max_matrix_size = mms
        self.a_matrix_xs = np.zeros(self.max_matrix_size)
        self.a_matrix_ys = np.zeros(self.max_matrix_size)
        self.a_matrix_vals = np.zeros(self.max_matrix_size)
        self.b_vector = []
        self.c_vector = []
        self.mag_vector = []
        self.matrix_size = 0
        self.ndets = 0
    
    def update(self, meas_info, mags, mag_errors, operands, max_nimgs=None):
        invsigma_array = 1.0/np.square(mag_errors)
        sum_invsigma2 = invsigma_array.sum()
        sum_m_i = (mags*invsigma_array).sum() / sum_invsigma2
        
        self.b_vector = np.append( self.b_vector, mags - sum_m_i )
        self.mag_vector = np.append( self.mag_vector, mags )
        
        if max_nimgs is None:
            max_nimgs = meas_info.max_nimgs
        
        a_submatrix = None
        ndet = len(mags)
        for nid,id_string in enumerate(meas_info.id_strings):
            invsigma_matrix = np.tile(invsigma_array*operands[nid], (max_nimgs[nid],1))
            sum_for_zps_nid = (meas_info.image_id_matrix[nid][0:max_nimgs[nid],0:ndet]*invsigma_matrix).sum(axis=1) / sum_invsigma2
            a_submatrix_nid = np.tile(sum_for_zps_nid, (ndet,1) )
            a_submatrix_nid[range(ndet),meas_info.image_ids[nid][range(ndet)]] -= operands[nid]
            if a_submatrix is None:
                a_submatrix = a_submatrix_nid
            else:
                a_submatrix = np.hstack((a_submatrix, a_submatrix_nid))
        
        # print a_submatrix.shape
        a_submatrix = coo_matrix(a_submatrix)
    
        indices = np.where(a_submatrix.data != 0.)[0]
        if( self.matrix_size+len(indices) > self.max_matrix_size ):
            self.a_matrix_xs = np.hstack((self.a_matrix_xs,np.zeros(self.max_matrix_size)))
            self.a_matrix_ys = np.hstack((self.a_matrix_ys,np.zeros(self.max_matrix_size)))
            self.a_matrix_vals = np.hstack((self.a_matrix_vals,np.zeros(self.max_matrix_size)))
            self.max_matrix_size += self.max_matrix_size
    
        self.a_matrix_xs[self.matrix_size:(self.matrix_size+len(indices))] = a_submatrix.col[indices]
        self.a_matrix_ys[self.matrix_size:(self.matrix_size+len(indices))] = a_submatrix.row[indices]+self.ndets
        self.a_matrix_vals[self.matrix_size:(self.matrix_size+len(indices))] = a_submatrix.data[indices]
        self.matrix_size += len(indices)
    
        self.c_vector = np.append(self.c_vector, invsigma_array)
        self.ndets += ndet
        
        return np.std(mags-sum_m_i)
    
    def solve(self, max_nimgs_total):
        
        a_matrix = coo_matrix((self.a_matrix_vals[0:self.matrix_size], (self.a_matrix_ys[0:self.matrix_size], self.a_matrix_xs[0:self.matrix_size])), shape=(self.ndets,max_nimgs_total))
        
        # (a_matrix, b_vector, c_vector) = tie_idvals(a_matrix, b_vector, c_vector, mag_vector)
        ndets = len(self.c_vector)
        
        c_matrix = lil_matrix((ndets,ndets))
        c_matrix.setdiag(1.0/self.c_vector)
        
        subterm = (a_matrix.transpose()).dot(c_matrix)
        termA = subterm.dot(a_matrix)
        termB = subterm.dot(self.b_vector)
        
        # p_vector = linalg.bicgstab(termA,termB)[0]
        # p_vector = linalg.spsolve(termA,termB)
        p_vector = linalg.minres(termA,termB)[0]
        return p_vector
    
    def solve_iterative(self, max_nimgs_total):
        
        a_matrix = coo_matrix((self.a_matrix_vals[0:self.matrix_size], (self.a_matrix_ys[0:self.matrix_size], self.a_matrix_xs[0:self.matrix_size])), shape=(self.ndets,max_nimgs_total))
        
        # dense!  only do this for small problems!
        a_matrix = a_matrix.todense()
        
        ndets = len(self.c_vector)
        
        c_matrix = np.zeros((ndets,ndets))
        c_matrix[range(ndets), range(ndets)] = 1.0/self.c_vector
        
        previous_zp = 0.
        new_zp = 100.
        iteration = 0
        p_vector = None
        while( (abs(new_zp - previous_zp) > abs(0.01*previous_zp)) or (abs(new_zp - previous_zp) > 0.005) ): # stop when zp is stable (rel and abs).  arbitrary convergence criteria.....
            iteration += 1
            previous_zp = new_zp
            
            subterm = (a_matrix.transpose()).dot(c_matrix)
            termA = subterm.dot(a_matrix)
            termB = np.array(subterm.dot(self.b_vector))[0]
            
            p_vector = linalg.minres(termA,termB)[0]
            
            print 'Outlier iteration {0} solution: {1}'.format(iteration, p_vector)
            # normalize by the calibrated data, save only the idval's result
            p_vector = p_vector[0]-p_vector[1]
            new_zp = p_vector
            
            # get the rms from the b_vector (mags - sum_m_i)
            mean_b = np.mean(abs(self.b_vector))
            rms = np.std(abs(self.b_vector))
            good_indices = np.where(abs(abs(self.b_vector)-mean_b) < 2.0*rms)[0]
            
            # print 'Outlier fit iteration {0}: {1}/{2} good measurements given rms {3}'.format(iteration,len(good_indices),len(self.b_vector), rms)
            if len(good_indices) < 10:
                print 'Giving up on this idval, no convergence.'
                return None
            a_matrix = a_matrix[good_indices,:]
            c_matrix = c_matrix[good_indices][:,good_indices]
            self.b_vector = self.b_vector[good_indices]
        
        return p_vector



class Measurement_Info(object):
    
    def __init__(self, id_s, ndet, max_ni):
        self.id_strings = id_s
        self.max_nimgs = max_ni
        
        self.image_ids = {}
        self.image_id_matrix = {}
        
        for nid,id_string in enumerate(self.id_strings):
            self.image_ids[nid] = np.zeros(ndet, dtype=np.int)
            self.image_id_matrix[nid] = np.zeros([self.max_nimgs[nid],ndet], dtype=np.int)
    
    def add_to_dicts(self, star, d, image_id_dict, image_id_count):
        for nid, id_string in enumerate(self.id_strings):
            try:
                self.image_ids[nid][d] = image_id_dict[nid][star[id_string]]
                self.image_id_matrix[nid][self.image_ids[nid][d],d] = 1
            except KeyError:
                # the current image id is not in the image_ids dictionary yet
                if nid not in image_id_count:
                    image_id_count[nid] = 0
                else:
                    image_id_count[nid] += 1
                
                if image_id_count[nid] >= self.max_nimgs[nid]:
                    print "WARNING, image_id_count = %d >= max_nimgs = %d" %(image_id_count[nid], self.max_nimgs[nid])
                
                image_id_dict[nid][star[id_string]] = image_id_count[nid] # will need to concatenate these to get the indices for p_vector
                self.image_ids[nid][d] = image_id_dict[nid][star[id_string]]
                self.image_id_matrix[nid][self.image_ids[nid][d],d] = 1
        return image_id_dict, image_id_count
    
    def assign(self, iids, idm):
        self.image_ids = {0:iids}
        self.image_id_matrix = {0:idm}
    
        
# for now.....
good_quality_magerr = 0.02
def good_quality(star):
    if star['magerr_psf'] < good_quality_magerr:
        return True
    return False



def standard_quality(star):
    if star['image_id'] != 1 and star['magerr_psf'] < good_quality_magerr and star['gskyphot'] == 1 and star['x_image']>100. and star['x_image']<1900. and star['y_image']>100. and star['y_image']<3950. and star['cloud_nomad']<0.2:
        return True
    else:
        return False


def ok_quality(star):
    if star['magerr_psf'] < good_quality_magerr*2.0 and star['x_image']>100. and star['x_image']<1900. and star['y_image']>100. and star['y_image']<3950.:
        return True
    return False


def unmatched(star):
    if star['matched']:
        return False
    return True


def match_radius():
    return 1.0/3600.


def normalize_p(p_vector, id_strings, image_id_dict, operands_in, max_nimgs):
    p_base = 0
    for nid, id_string in enumerate(id_strings):
        # only normalize the additive zero points...  not ideal, but not sure what's better.
        if operands_in[nid] not in [ 'None', None, 1 ]:
            continue
        # deal with the indices for each id_string separately
        p_vector_indices = image_id_dict[nid].values()
        # normalize to the max of the ZPs (the best-quality obs)
        # median = np.median(p_vector[p_vector_indices])
        maximum = np.amax(p_vector[p_vector_indices])
        print "%s: Normalizing to the maximum of the zeropoints: %e" %(id_string, maximum)
        # replace zeros with nans
        p_vector[p_vector == 0.0] = np.nan
        p_vector[p_vector_indices] -= maximum
        p_base += max_nimgs[nid]
    return p_vector


def tie_idvals(a_matrix, b_vector, c_vector, mag_vector):
    
    print 'Tying id_vals together to regularize the solution'
    
    # sort the mag_vector (keep indices)
    # for each mag, use it and its forward neighbor if:
    # - they are not connected directly
    # - we do not have this connection already in our linking sample
    linked = {}
    indices = np.argsort(mag_vector)
    i1=0
    i2=1
    n_target = 0.05*len(b_vector) # add 5% to the problem
    n_added = 0
    while i2<len(indices)-1:
        index1 = indices[i1]
        index2 = indices[i2]
        nonzero1 = (a_matrix.getrow(index1).toarray()!=0)[0]
        nonzero2 = (a_matrix.getrow(index2).toarray()!=0)[0]
        if np.sum(nonzero1 & nonzero2) == 0: # they are not connected by anything in p_vector
            first_nonzero1 = np.nonzero(nonzero1)[0][0]
            first_nonzero2 = np.nonzero(nonzero2)[0][0]
            
            # if we already have linked these zps (just the first one in the idval list), continue.
            already = False
            if (first_nonzero1 in linked.keys()) and (first_nonzero2 in linked[first_nonzero1]):
                already = True
            if (first_nonzero2 in linked.keys()) and (first_nonzero1 in linked[first_nonzero2]):
                already = True
            
            if not already:
                # add this to our linked dictionary for bookkeeping
                if not first_nonzero1 in linked:
                    linked[first_nonzero1] = []
                linked[first_nonzero1].append(first_nonzero2)
                if not first_nonzero2 in linked:
                    linked[first_nonzero2] = []
                linked[first_nonzero2].append(first_nonzero1)
            
                # add two entries to b_vector, with normal individual A and c but their mean in b
                # stop after i get X% of the original vector length.  5%?
                mean_mag = 0.5*(mag_vector[index1] + mag_vector[index2])
                b_vector = np.hstack((b_vector, [mag_vector[index1]-mean_mag, mag_vector[index2]-mean_mag]))
                c_vector = np.hstack((c_vector, [0.02, 0.02])) # arbitrary, same as systematic error added elsewhere
            
                a_matrix = sparsevstack([a_matrix, a_matrix.getrow(index1)])
                a_matrix = sparsevstack([a_matrix, a_matrix.getrow(index2)])
            
                n_added += 2
            
                if n_added >= n_target:
                    print 'tie_idvals: Added {0}/{1} objects to tie together different regions'.format(n_added/2, len(b_vector)/2)
                    break
        i2 += 1
        if (mag_vector[indices[i2]]-mag_vector[index1] > 0.005) or (i2 >= len(indices)-1): # arbitrary but close to relative calibration floor
            i1 += 1
            i2 = i1+1
    
    if n_added < n_target:
        print 'tie_idvals: Warning, only found {0}/{1} similar objects to tie together the field'.format(n_added/2, len(b_vector)/2)
            
    return (a_matrix, b_vector, c_vector)


def std_ubercal( band, precal, config_general ):
    # can put in a dependence (RA and Dec?) but for now, just do a ZP.
    # do this all at once;  assuming we don't have so many standards we'll overload the memory
    
    print 'Starting calibration to standards'
    
    # read in the standard catalog
    # read in any catalogs we would like to treat as standards (i.e. precam, sdss...)
    std_stars = []
    std_map = index.Index()
    if config_general['use_precam']:
        read_precam( std_stars, std_map, config_general['precam_filename'], band )
    if config_general['use_sdss']:
        read_sdss( std_stars, std_map, config_general['sdss_filename'], band )
    if config_general['use_betoule']:
        read_betoule( std_stars, std_map, config_general['betoule_filename'], band )
    if config_general['use_tertiaries']:
        read_tertiaries( std_stars, std_map, config_general['tertiary_filename'], band )
    print 'Read in {0} standard stars'.format(len(std_stars))
    
    
    # let's figure out how many different labels we have
    # use precal[1] (the first calibration post-zp_phot) lables
    # each label number corresponds to a disjoint region in the data
    pre_labels = precal[1][5]
    unique_labels = np.unique(pre_labels.values())
    unique_labels = unique_labels[(unique_labels != 999)]
    n_labels = len(unique_labels)
    
    max_matrix_size = 1000
    a_matrix_xs = {}
    a_matrix_ys = {}
    a_matrix_vals = {}
    for nl in range(n_labels):
        a_matrix_xs[nl] = np.zeros(max_matrix_size)
        a_matrix_ys[nl] = np.zeros(max_matrix_size)
        a_matrix_vals[nl] = np.zeros(max_matrix_size)
    matrix_size = np.zeros(n_labels, dtype='int')
    ndets = np.zeros(n_labels,dtype='int')
    b_vector = []
    c_vector = []
    for nl in range(n_labels):
        b_vector.append([])
        c_vector.append([])
    
    # read in the objs and match to the standards, one pixel at a time.
    npix = 1
    if config_general['nside_file']>0:
        npix = healpy.nside2npix(config_general['nside_file'])
    
    for pix in range(npix):
        
        global_objs_list = read_global_objs(band, config_general['globals_dir'], config_general['nside_file'], config_general['nside_file'], pix)
        if global_objs_list is None:
            continue
        
        for go in global_objs_list:
        
            if go.dec < config_general['dec_min'] or go.dec > config_general['dec_max']:
                continue
            if config_general['ra_min'] > config_general['ra_max']: # 300 60
                if go.ra > config_general['ra_max'] and go.ra < config_general['ra_min']:
                    continue
            else:
                if go.ra < config_general['ra_min'] or go.ra > config_general['ra_max']:
                    continue
        
            # cut out objects that are bad quality!
            go.objects = filter(ok_quality, go.objects)
            if len(go.objects) == 0:
                continue
            
            # match to standards
            ra_match_radius = match_radius()/math.cos(go.dec*math.pi/180.)
            dec_match_radius = match_radius()
            match_area = (go.ra-ra_match_radius, go.dec-dec_match_radius,
                            go.ra+ra_match_radius, go.dec+dec_match_radius)
            det_indices = list(std_map.intersection(match_area))
            if len(det_indices) == 1:
                std_star = std_stars[det_indices[0]]
                
                # calibrate the matches and average to get one best mag per obj
                # do one ubercal per label in the FIRST non-zp_phot calibration (precal[1])
                calib_mags = []
                label = None
                for star in go.objects:
                    mag = star['mag_psf']
                    good = True
                    for p in precal:
                        [pre_vector, precal_id_string, precal_operand, pre_median, pre_outliers, pre_labels] = p
                        coeff = None
                        if precal_operand in [ 'None', None, 1 ]:
                            coeff = 1.
                        else:
                            coeff = star[precal_operand]
                        if star[precal_id_string] in pre_vector:
                            mag += pre_vector[star[precal_id_string]]*coeff
                        else:
                            good = False
                    if good:
                        calib_mags.append(mag)
                        label0 = precal[1][5][star[precal[1][1]]]
                        if label0 != 999:
                            label = label0
                    
                if len(calib_mags) == 0:
                    continue
                
                if label is None:
                    print 'WARNING, std calibration finds no good label, skipping object.'
                    continue
            
                # add the matches to the matrices
                ndet = 2
                image_ids = np.array([0,1])
                image_id_matrix = np.array([[1,0],[0,1]])
                a_submatrix = {}
            
                mags = np.array([np.mean(calib_mags), std_star['mag_psf']])
                mag_errors = np.array([0.02, 0.02]) # arbitrary...
            
                invsigma_array = 1.0/np.square(mag_errors)
                sum_invsigma2 = invsigma_array.sum()
                sum_m_i = (mags*invsigma_array).sum() / sum_invsigma2
            
                invsigma_matrix = np.tile(invsigma_array, (2,1))
                sum_for_zps_nid = (image_id_matrix*invsigma_matrix).sum(axis=1) / sum_invsigma2
                a_submatrix = np.tile(sum_for_zps_nid, (ndet,1) )
                a_submatrix[range(ndet),image_ids[range(ndet)]] -= 1.0
                a_submatrix = coo_matrix(a_submatrix)
            
                b_vector[label] = np.append( b_vector[label], mags - sum_m_i )
            
                indices = np.where(a_submatrix.data != 0.)[0]
                if( matrix_size[label]+len(indices) > len(a_matrix_xs[label]) ):
                    a_matrix_xs[label] = np.hstack((a_matrix_xs[label],np.zeros(max_matrix_size)))
                    a_matrix_ys[label] = np.hstack((a_matrix_ys[label],np.zeros(max_matrix_size)))
                    a_matrix_vals[label] = np.hstack((a_matrix_vals[label],np.zeros(max_matrix_size)))
                    max_matrix_size += max_matrix_size
                
                a_matrix_xs[label][matrix_size[label]:(matrix_size[label]+len(indices))] = a_submatrix.col[indices]
                a_matrix_ys[label][matrix_size[label]:(matrix_size[label]+len(indices))] = a_submatrix.row[indices]+ndets[label]
                a_matrix_vals[label][matrix_size[label]:(matrix_size[label]+len(indices))] = a_submatrix.data[indices]
                matrix_size[label] += len(indices)
            
                c_vector[label] = np.append(c_vector[label], invsigma_array)
            
                ndets[label] += ndet
                
        # end loop over pixels
    
    print 'Read in data from all pixels'
    
    if np.sum(ndets) == 0:
        print 'WARNING: No standards match the sample!!'
        return {0:0.0}
    else:
        print '{0} objects matched to the standards'.format(ndets/2)
    
    for label in a_matrix_xs.keys():
        a_matrix_xs[label] = a_matrix_xs[label][0:ndets[label]]
        a_matrix_ys[label] = a_matrix_ys[label][0:ndets[label]]
        a_matrix_vals[label] = a_matrix_vals[label][0:ndets[label]]
    
    # Now do the solution for each label (disjoint area)
    result = {}
    for label in range(n_labels):
        
        if ndets[label] == 0:
            continue
        
        b_vector0 = np.array(b_vector[label])
        a_matrix = coo_matrix((a_matrix_vals[label], (a_matrix_ys[label], a_matrix_xs[label])), shape=(ndets[label],2))
        a_matrix = a_matrix.todense() #so that we can clip out bad values;  this should be small anyway, if we don't have a gazillion standards?
        c_matrix = np.zeros((ndets[label],ndets[label]))
        c_matrix[range(ndets[label]), range(ndets[label])] = 1.0/c_vector[label]
    
        previous_zp = 0.
        new_zp = 100.
        iteration = 0
        while( (abs(new_zp - previous_zp) > abs(0.01*previous_zp)) or (abs(new_zp - previous_zp) > 0.005) ): # stop when zp is stable (rel and abs).  arbitrary convergence criteria.....
            iteration += 1
            previous_zp = new_zp
        
            subterm = (a_matrix.transpose()).dot(c_matrix)
            termA = subterm.dot(a_matrix)
            termB = np.array(subterm.dot(b_vector0))[0]
            p_vector = linalg.minres(termA,termB)[0]
        
            print 'Standard calibration iteration {0} solution: {1}'.format(iteration, p_vector)
            # normalize by the calibrated data, save only the idval's result
            p_vector = p_vector[0]-p_vector[1]
            new_zp = p_vector
        
            # get the rms from the b_vector (mags - sum_m_i)
            mean_b = np.mean(abs(b_vector0))
            rms = np.std(abs(b_vector0))
            good_indices = np.where(abs(abs(b_vector0)-mean_b) < 2.0*rms)[0]
        
            print 'Standard calibration iteration {0}: {1}/{2} good measurements given rms {3}'.format(iteration,len(good_indices),len(b_vector0), rms)
            if len(good_indices) < 2:
                print 'Giving up on this calibration, no convergence.'
                break
            a_matrix = a_matrix[good_indices,:]
            c_matrix = c_matrix[good_indices][:,good_indices]
            b_vector0 = b_vector0[good_indices]
        
        result[label] = p_vector
    
    return result
    


def nebencalibrate_pixel( inputs ):
    
    [pix, nside, nside_file, band, precal, globals_dir, id_strings, operands_in, max_dets, ra_min, ra_max, dec_min, dec_max, use_color_terms, mag_classes] = inputs
        
    max_nepochs = 5 # ick!  just a starting point, though.
    magerr_sys2 = 0.0004
    
    n_iters = 3 
    
    # read in the CCD positions in the focal plane    
    fp_xs, fp_ys = read_ccdpos()
    
    # read the global objects in from the files.
    global_objs_list = read_global_objs(band, globals_dir, nside, nside_file, pix)
    if global_objs_list is None:
        return None, None, None, None
        
    print "Starting pixel %d: %d global objects" %(pix, len(global_objs_list))
    
    bad_idvals = []
    for nid, id_string in enumerate(id_strings):
        bad_idvals.append({})
    
    nimgs_flagged = 1
    iteration = 0
    image_id_dict = None
    while( nimgs_flagged != 0 ):
        iteration += 1
        
        mag_vector = []
        gcount = 0
        sum_rms = 0.
        n_rms = 0
        matrix_size = 0
        n_good_e = 0
        n_bad_e = 0
        
        
        stars = []
        star_map = index.Index()
        p_vector = None
        
        imgids = dict()
        
        image_id_count = dict()
        image_id_dict = dict()
        for nid, id_string in enumerate(id_strings):
            image_id_dict[nid] = dict()
        
        matrix_info = Matrix_Inputs(len(global_objs_list)*max_nepochs)
        
        # is it worth finding out max_nimgs correctly?  YES.
        # also, check to see if there are too many objects per image/exposure
        imgids = dict()
        img_ndets = dict()
        img_errs = dict()
        max_errors = dict()
        ndet = 0
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
        
            obj_inarea = True
        
            # cut out objects that are bad quality!
            go.objects = filter(ok_quality, go.objects)
            if len(go.objects) < 2:
                continue
        
            for obj in go.objects:
            
                # if bad quality, don't use for calibration
                # (don't use at all!)
                ok_idval = True
                for nid, id_string in enumerate(id_strings):
                    if obj[id_string] in bad_idvals[nid]:
                        ok_idval = False
                if standard_quality(obj) and ok_idval:
                    obj['for_mean'] = True
                    ndet += 1
                else: # don't use at all!
                    obj['for_mean'] = False
                    continue
            
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
            
        
        if not obj_inarea:
            return None, None, None, None
            
        max_nimgs = dict()
        max_nimgs_total = 0
        skip = False
        for nid, id_string in enumerate(id_strings):
            if nid in imgids.keys():
                max_nimgs[nid] = len(imgids[nid].keys())
                max_nimgs_total += max_nimgs[nid]
            else:
                print 'WARNING, calibration string {0} has no acceptable data'
                skip = True
        if skip:
            continue
        img_connectivity = None
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
            # also only do the decent objects ("for_mean")
            for nid,id_string in enumerate(id_strings):
                go_objects = [obj for obj in go.objects if obj['for_mean'] == True and obj[id_string] in worst_ok_err[nid] and obj['magerr_psf'] < worst_ok_err[nid][obj[id_string]][1]]
        
            if len(go_objects) < 2:
                continue
        
            ndet = len(go_objects)
            ra = go.ra
            dec = go.dec
            
            if ndet < 2:
                continue
        
            mags = np.zeros(ndet)
            mag_errors = np.zeros(ndet)
            operands = dict()
            for nid,id_string in enumerate(id_strings):
                operands[nid] = np.zeros(ndet)
            
            meas_info = Measurement_Info(id_strings, ndet, max_nimgs)
            
            d = 0
            for star in go_objects:
            
                mags[d] = star['mag_psf']
                mag_errors[d] = np.sqrt(star['magerr_psf']*star['magerr_psf'] + magerr_sys2)
            
                if use_color_terms:
                    mags[d] = apply_color_term(mags[d], go, fp_xs, fp_ys, mag_classes)
                
                for p in precal:
                    [pre_vector, precal_id_string, precal_operand, pre_median, pre_outliers, pre_labels] = p
                    if star[precal_id_string] in pre_vector:
                        coeff = None
                        if precal_operand == 'None' or precal_operand is None or precal_operand == 1:
                            coeff = 1.
                        else:
                            coeff = star[precal_operand]
                        mags[d] += pre_vector[star[precal_id_string]]*coeff
                        n_good_e += 1
                    else:
                        n_bad_e += 1
                        continue
                
                for nid, id_string in enumerate(id_strings):
                    if operands_in[nid] in [ 'None', None, 1 ]:
                        operands[nid][d] = 1
                    else:
                        operands[nid][d] = star[operands_in[nid]]
                        
                image_id_dict, image_id_count = meas_info.add_to_dicts(star, d, image_id_dict, image_id_count)
            
                d += 1
            
            sum_rms += matrix_info.update(meas_info, mags, mag_errors, operands)
            n_rms += 1
            
            # connectivity matrix
            # only for one image_id....
            if len(id_strings) == 1:
                for star in go_objects:
                    if star[id_string] in image_id_dict[0]:
                        for star2 in go_objects:
                            if star2[id_string] in image_id_dict[0]:
                                img_connectivity[image_id_dict[0][star[id_string]], image_id_dict[0][star2[id_string]]] = 1
                
        print "%d good previous zps, %d bad" %(n_good_e, n_bad_e)
        print '{0} images in image_id_dict[0]'.format(len(image_id_dict[0].keys()))
        n_components = 1
        labels = None
        if len(id_strings) > 1:
            print "WARNING, not checking disjoing regions because you are using >1 id_string"
        else:
            n_components, labels = cs_graph_components(img_connectivity)
            print "# disjoint regions: %d" %(n_components)
        
        id_string = id_strings[0]
        if n_components > 1:
            print "Warning, %d disjoint regions in the data!!" %n_components
        
        if( matrix_info.ndets == 0 ):
            # continue
            return image_id_dict, p_vector, None, labels
        
        gcount = len(global_objs_list)
        print "Looped through %d global objects, %d measurements total" %( gcount, matrix_info.ndets )
        print "Mean RMS of the stars: %e" %(sum_rms/n_rms)
        
        p_vector = matrix_info.solve(max_nimgs_total)
        
        p_vector = normalize_p(p_vector, id_strings, image_id_dict, operands_in, max_nimgs)
        
        print "Solution:"
        print p_vector.shape
        print p_vector
        
        # now calculate how we did
        sum_rms = 0.
        n_rms = 0
        sum_rms_formean = 0.
        n_rms_formean = 0.
        n_good_e = 0
        n_bad_e = 0
        dmag_by_img = []
        dmag2_by_img = []
        n_by_img = []
        nmeas_flagged = 0
        nimgs_flagged = 0
        
        for nid,id_string in enumerate(id_strings):
            dmag_by_img.append({})
            dmag2_by_img.append({})
            n_by_img.append({})
            
        for go in global_objs_list:
            
            go_objects = [obj for obj in go.objects if obj[id_string] in worst_ok_err[nid] and obj['magerr_psf'] < worst_ok_err[nid][obj[id_string]][1]]
            
            if len(go_objects) < 2:
                continue
        
            if go.dec < dec_min or go.dec > dec_max:
                continue
            if ra_min > ra_max: # 300 60
                if go.ra > ra_max and go.ra < ra_min:
                    continue
            else:
                if go.ra < ra_min or go.ra > ra_max:
                    continue
        
            mags = np.zeros(len(go_objects))
            mag_errors = np.zeros(len(go_objects))
            mags_formean = np.zeros(len([obj for obj in go_objects if obj['for_mean']]))
            mag_errors_formean = np.zeros(len(mags_formean))
            id_strings_tally = []
            for nid in range(len(id_strings)):
                id_strings_tally.append(np.zeros(len(go_objects)))
            sids = np.zeros(len(go_objects),dtype=int)
            i=0
            i2=0
            bad = dict()
            for nid,id_string in enumerate(id_strings):
                bad[nid] = dict()
        
            for s,star in enumerate(go_objects):
            
                mags[i] = star['mag_psf']
                mag_errors[i] = star['magerr_psf']
                sids[i] = s
                for nid,id_string in enumerate(id_strings):
                    id_strings_tally[nid][i] = star[id_string]
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
                        if operands_in[nid] in [ 'None', None, 1 ]:
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
                    [pre_vector, precal_id_string, precal_operand, pre_median, pre_outliers, pre_labels] = p
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
                    stddev = np.std(mags_formean-sum_m_i)
                    sum_rms_formean += stddev
                    n_rms_formean += 1
            
                    # if there were enough objects to get a decent mean, look for outliers.
                    for j in range(i):
                        if abs(mags[j] - sum_m_i) > 3.0*stddev:
                            # mark j as an outlier and mark it as bad.
                            # mark its mag_error as > good_mag_error
                            go_objects[sids[j]]['magerr_psf'] = 10.0*good_quality_magerr
                            nmeas_flagged += 1
                        for nid,id_string in enumerate(id_strings):
                            if id_strings_tally[nid][j] not in dmag_by_img[nid]:
                                dmag_by_img[nid][id_strings_tally[nid][j]] = 0.
                                dmag2_by_img[nid][id_strings_tally[nid][j]] = 0.
                                n_by_img[nid][id_strings_tally[nid][j]] = 0.
                            dmag_by_img[nid][id_strings_tally[nid][j]] += mags[j]-sum_m_i
                            dmag2_by_img[nid][id_strings_tally[nid][j]] += (mags[j]-sum_m_i)**2
                            n_by_img[nid][id_strings_tally[nid][j]] += 1.0
                        
        if n_rms == 0:
            n_rms = 1
        if n_rms_formean == 0:
            n_rms_formean = 1
    
        print "Mean RMS after calibration: %e (%e with everything), %d/%d images calibrated" %(sum_rms_formean/n_rms_formean, sum_rms/n_rms, len(p_vector)-len(bad.keys()), len(p_vector))
    
        p_base = 0
        for nid,id_string in enumerate(id_strings):
            # calculate the mean rms and find outliers.
            rms_by_img = {}
            for idval in dmag_by_img[nid].keys():
                rms_by_img[idval] = np.sqrt(dmag2_by_img[nid][idval]/n_by_img[nid][idval] - (dmag_by_img[nid][idval]/n_by_img[nid][idval])*(dmag_by_img[nid][idval]/n_by_img[nid][idval]))
            mean_rms = np.mean(rms_by_img.values())
                
            # also cut out from the mean images with outlier zps
            good_zps = []
            for idval in image_id_dict[nid].keys():
                if not idval in bad_idvals[nid].keys():
                    if np.isnan(p_vector[p_base+image_id_dict[nid][idval]]):
                        bad_idvals[nid][idval] = True
                        nimgs_flagged += 1
                    else:
                        good_zps.append(p_vector[p_base+image_id_dict[nid][idval]])
            mean_zp = np.mean(good_zps)
            rms_zp = np.std(good_zps)
            print 'zp mean {0} rms {1}'.format(mean_zp,rms_zp)
            for idval in dmag_by_img[nid].keys():
                # cut at 2.5 sigma
                if (idval in image_id_dict[nid]) and (rms_by_img[idval] > 2.5*mean_rms or abs(p_vector[p_base+image_id_dict[nid][idval]]-mean_zp) > 2.5*rms_zp):
                    # mark idval as bad
                    bad_idvals[nid][idval] = True
                    nimgs_flagged += 1
            # let's renormalize by the mean of the good zps!
            p_vector[p_base:p_base+max_nimgs[nid]] -= mean_zp
            p_base += max_nimgs[nid]
                    
        print 'Iteration {0}: {1} measurement outliers, {2} image outliers'.format(iteration,nmeas_flagged,nimgs_flagged)
        
    # end loop over iterations!
    
    # make a massive "image" of all the calibrated data
    # loop over uncalibrated outlier images and pairwise calibrate them to the master
    # normalize the result to the master's zp.
    # this pairwise business will only work if there is one id_string at a time.
    p_vector_outlier = None
    outlier_list = []
    labels_outliers = {}
    if len(id_strings) == 1:
        print 'Pairwise calibrating the outliers'
            
        outlier_idvals = {}
        p_vector_outlier = {}
        nid = 0
        id_string = id_strings[0]
        
        a_matrix_xs = {}
        a_matrix_ys = {}
        a_matrix_vals = {}
        b_vector = {}
        c_vector = {}
        matrix_size = {}
        ndets = {}
        
        max_matrix_size = 100
        
        matrix_infos = {}
        
        # always the same.
        meas_info = Measurement_Info([id_string], 2, [2])
        image_ids_outlier = np.array([0,1])
        image_id_matrix_outlier = np.array([[1,0],[0,1]])
        meas_info.assign(image_ids_outlier, image_id_matrix_outlier)
        
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
            go_objects = [obj for obj in go.objects if (obj[id_string] in worst_ok_err[nid] and obj['magerr_psf'] < worst_ok_err[nid][obj[id_string]][1]) or (obj[id_string] not in worst_ok_err[nid])]

            if len(go_objects) < 2:
                continue

            ndet = len(go_objects)

            if ndet < 2:
                continue
    
            # are there both good and bad idvals in this object?
            calibrated = [obj[id_string] in image_id_dict[nid] for obj in go_objects]
            if True not in calibrated or False not in calibrated:
                continue
    
            # jimmy the measurements so that all the calibrated ones are averaged into one value.            
        
            ndet = 2 # for each id val separately
            ndet_calib = np.sum(calibrated)
            
            mags = {} # these are used differently from before, with one per idval
            mag_errors = {}
            operands = {}
            mags_calib = np.zeros(ndet_calib)
            mag_errors_calib = np.zeros(ndet_calib)
            dc = 0
            calib_label = None
            uncalib_ids = []
            for s, star in enumerate(go_objects):
                
                if calibrated[s]:
                    mags_calib[dc] = star['mag_psf']
                    mag_errors_calib[dc] = np.sqrt(star['magerr_psf']*star['magerr_psf'] + magerr_sys2)            
                    calib_label = labels[image_id_dict[0][star[id_strings[0]]]]
                else:
                    outlier_idvals[star[id_string]] = True
                    mags[star[id_string]] = [star['mag_psf']]
                    mag_errors[star[id_string]] = [np.sqrt(star['magerr_psf']*star['magerr_psf'] + magerr_sys2)]
                    uncalib_ids.append(star[id_string])
                if use_color_terms:
                    if calibrated[s]:
                        mags_calib[dc] = apply_color_term(mags_calib[dc], go, s, fp_xs, fp_ys, mag_classes)
                    else:
                        mags[star[id_string]][0] = apply_color_term(mags[star[id_string]][0], go, s, fp_xs, fp_ys, mag_classes)
                
                for p in precal:
                    [pre_vector, precal_id_string, precal_operand, pre_median, pre_outliers, pre_labels] = p
                    if star[precal_id_string] in pre_vector:
                        coeff = None
                        if precal_operand == 'None' or precal_operand is None or precal_operand == 1:
                            coeff = 1.
                        else:
                            coeff = star[precal_operand]
                        if calibrated[s]:
                            mags_calib[dc] += pre_vector[star[precal_id_string]]*coeff
                        else:
                            mags[star[id_string]][0] += pre_vector[star[precal_id_string]]*coeff
                        n_good_e += 1
                    else:
                        # if there's no valid precal, then don't accept this!
                        n_bad_e += 1
                        continue
        
                if not calibrated[s]:
                    if operands_in[nid] in [ 'None', None, 1 ]:
                        operands[star[id_string]] = 1
                    else:
                        operands[star[id_string]] = star[operands_in[nid]]
                else:
                    dc += 1
            
            mean_calib = (mags_calib*(1.0/np.square(mag_errors_calib))).sum() / (1.0/np.square(mag_errors_calib)).sum()
            error_calib = np.median(mag_errors_calib)
            
            for uncalib_id in uncalib_ids:
                labels_outliers[uncalib_id] = calib_label
            
            # now add the data to the a_matrices, with an a_matrix per idval.
            nid = 0
            a_submatrix = {}
            for idval in mags.keys():
                
                if idval in ndets and ndets[idval] > max_dets:
                    # not ideal, because we're randomly picking the first bunch.
                    continue
                
                # add the calibrated data to the array
                mags[idval].append(mean_calib)
                mag_errors[idval].append(error_calib)
                
                if not idval in matrix_infos.keys():
                    matrix_infos[idval] = Matrix_Inputs(max_matrix_size)
                
                matrix_infos[idval].update(meas_info, np.array(mags[idval]), np.array(mag_errors[idval]), {0:[1.,1.]})
                
        
        n_outliers = len(matrix_infos.keys())
        print '{0} outliers'.format(n_outliers)
        for i, idval in enumerate(matrix_infos.keys()):
            p_vector_outlier[idval] = matrix_infos[idval].solve_iterative(2)
        
        outlier_list.append(outlier_idvals.keys())
    else:
        print 'More than one id_string to calibrate simultaneously: can\'t pairwise recalibrate the outliers.'
        outlier_list = [len(id_strings)*[]]
        
    
    # split p_vector into the different n_id sections
    p_vectors = []
    p_base = 0
    for nid, id_string in enumerate(id_strings):
        p_vectors.append(p_vector[p_base:p_base+max_nimgs[nid]])
        p_base += max_nimgs[nid]
        
    if p_vector_outlier is not None:
        # add the outlier zps to the output
        for idval in p_vector_outlier.keys():
            image_id_dict[0][idval] = len(p_vectors[0])
            labels = np.hstack((labels, [labels_outliers[idval]]))
            p_vectors[0] = np.hstack((p_vectors[0],p_vector_outlier[idval]))
    
    return image_id_dict, p_vectors, outlier_list, labels



def nebencalibrate( band, nside, nside_file, id_strings, operands, precal, globals_dir, max_dets=1e9, ra_min=-360., ra_max=361., dec_min=91., dec_max=91., use_color_terms=False, mag_classes={} ):
    
    npix = 1
    if nside>0:
        npix = healpy.nside2npix(nside)
    
    p_vectors = []
    image_id_dicts = []
    outliers = []
    labels = []
    for pix in range(npix):
        p_vectors.append(None)
        image_id_dicts.append(dict())
        outliers.append(None)
        labels.append(None)
    
    
    # nebencalibrate the pixels!
    pix_wobjs = []
    inputs= []
    for p in range(npix):
        inputs.append( [p, nside, nside_file, band, precal, globals_dir, id_strings, operands, max_dets, ra_min, ra_max, dec_min, dec_max, use_color_terms, mag_classes] )
        image_id_dicts[p], p_vectors[p], outliers[p], labels[p] = nebencalibrate_pixel(inputs[p])
        if p_vectors[p] is not None:
            pix_wobjs.append(p)
    npix_wobjs = len(pix_wobjs)
    
    img_connectivity = None
    max_nlabels = 10
    if len(id_strings) == 1:
        img_connectivity = np.zeros((max_nlabels*npix_wobjs, max_nlabels*npix_wobjs), dtype='int')
    
    
    # now do another ubercalibration to make sure the pixels are normalized to each other
    # the measurements are now the difference between two zps calculated for the same image, during different pixels' ubercalibrations.
    
    # If there is only one pixel with objects, let's just save the ZPs and be done.
    zeropoints_tot = []
    outliers_tot = []
    labels_tot = []
    if npix_wobjs == 1:
        for nid,id_string in enumerate(id_strings):
            zeropoints = dict()
            outliers_nid = []
            imagelist = []
            labels0 = dict()
            for pix in range(npix):
                if image_id_dicts[pix][nid] is not None:
                    imagelist.extend(image_id_dicts[pix][nid].keys())
            imagelist = list(set(imagelist))
            print "%s: There are %d images in total, from %d pixels." %(id_string, len(imagelist), npix_wobjs)
            
            pix = pix_wobjs[0]
            for imageid in imagelist:
                if (imageid in image_id_dicts[pix][nid]) and (p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]] is not None) and (not np.isnan(p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]])):
                    zeropoints[imageid] = p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]]
                    if nid == 0:
                        labels0[imageid] = labels[pix][image_id_dicts[pix][nid][imageid]]
                    else:
                        labels0[imageid] = 1
                    if imageid in outliers[pix][nid]:
                        outliers_nid.append(imageid)
            labels_tot.append(labels0)
            zeropoints_tot.append(zeropoints)
            outliers_tot.append(outliers_nid)
        
        return zeropoints_tot, outliers_tot, labels_tot
    
    
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
    sum_rms = 0.
    n_rms = 0
    pix_id_dict2 = [dict()]
    pix_id_count = {} 
    outliers_nid = []
    
    max_matrix_size = len(imagelist)*npix_wobjs
    matrix_info = Matrix_Inputs(max_matrix_size)
    
    for imageid in imagelist: # like for go
        
        zps = []
        pix_id_matrix2 = np.zeros([npix_wobjs,npix_wobjs], dtype=np.int)
        pix_ids2 = np.zeros(npix_wobjs, dtype=np.int)
        
        # is the image not an outlier in at least one pixel?
        # if it's an outlier in all pixels than use the outlier value, but flag it.
        use_outlier = True
        for pix_id in range(npix_wobjs):
            pix = pix_wobjs[pix_id]
            if not nid in image_id_dicts[pix]:
                continue
            if (imageid in image_id_dicts[pix][nid]) and not (imageid in outliers[pix][nid]):
                use_outlier = False
        if use_outlier:
            outliers_nid.append(imageid)
        
        
        meas_info = Measurement_Info(['pix_zp'], npix_wobjs, [npix_wobjs])
        
        
        
        nzps = 0
        for pix_id in range(npix_wobjs):
            pix = pix_wobjs[pix_id]
            if not nid in image_id_dicts[pix]:
                continue
            if (imageid in image_id_dicts[pix][nid]) and (p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]] != 0.) and (p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]] is not None) and not np.isnan(p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]]):
                if (not use_outlier) and (imageid in outliers[pix][nid]):
                    continue
                zps.append(p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]])
                instance = {'pix_zp' : pix_id}
                
                pix_id_dict2, pix_id_count = meas_info.add_to_dicts( instance, nzps, pix_id_dict2, pix_id_count )
                nzps += 1
                
                for pix_id2 in range(npix_wobjs):
                    pix2 = pix_wobjs[pix_id2]
                    if not nid in image_id_dicts[pix2]:
                        continue
                    if (imageid in image_id_dicts[pix2][nid]) and (p_vectors[pix2][nid][image_id_dicts[pix2][nid][imageid]] != 0.) and (p_vectors[pix2][nid][image_id_dicts[pix2][nid][imageid]] is not None) and not np.isnan(p_vectors[pix2][nid][image_id_dicts[pix2][nid][imageid]]):
                        if (not use_outlier) and (imageid in outliers[pix2][nid]):
                            continue
                        # link up the same images across pixels
                        img_connectivity[max_nlabels*pix_id + labels[pix][image_id_dicts[pix][nid][imageid]], max_nlabels*pix_id2 + labels[pix2][image_id_dicts[pix2][nid][imageid]]] = 1
        
        
        if nzps == 0:
            print "Image %d has no zps" %imageid
            continue
        
        
        pix_id_matrix2 = pix_id_matrix2[:,0:nzps]
        
        zps = np.array(zps)
        zp_errors = np.ones(nzps)
        
        sum_rms += matrix_info.update( meas_info, zps, zp_errors, {0:np.ones(len(zps))} )
        
        n_rms += 1
        
    print "Looped through the images"
    print "Mean RMS of the zps: %e with %d images with zps" %(sum_rms/n_rms, n_rms)
    
    p1_vector = matrix_info.solve(npix_wobjs)
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
            if imageid in image_id_dicts[pix][nid] and pix_id in pix_id_dict2[0] and p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]] is not None and (not np.isnan(p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]])):
                zps.append(p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]]+p1_vector[pix_id_dict2[0][pix_id]])
        sum_rms += np.std(zps)
        n_rms += 1
        
    print "Mean RMS of the zps after correction: %e" %(sum_rms/n_rms)
    
    n_components2 = 1
    labels2 = None
    if len(id_strings) > 1:
        print "WARNING, not checking disjoing regions because you are using >1 id_string"
    else:
        n_components2, labels2 = cs_graph_components(img_connectivity)
        if n_components2 > max_nlabels:
            msg =  'Error dealing with disjoint regions, there are more than {0} of them.'.format(max_nlabels)
            raise Exception(msg)
        print "# disjoint regions: %d" %(n_components2)
    if n_components2 > 1:
        print "Warning, %d disjoint regions in the data!!" %n_components2
    
    
    superlabels = {}
    for imageid in imagelist:
        for pix_id in range(npix_wobjs):
            pix = pix_wobjs[pix_id]
            if not nid in image_id_dicts[pix]:
                continue
            if (imageid in image_id_dicts[pix][nid]) and (p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]] != 0.) and (p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]] is not None) and not np.isnan(p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]]):
                if (not use_outlier) and (imageid in outliers[pix][nid]):
                    continue
                superlabel = labels2[max_nlabels*pix_id+labels[pix][image_id_dicts[pix][nid][imageid]]]
                if imageid in superlabels:
                    if superlabels[imageid] != superlabel:
                        print "WARNING, conflicting labels when combining pixel results"
                else:
                    superlabels[imageid] = superlabel
    
    
    # now compile all the info into one zp per image.
    # for each image, find the p_vector zps and add the p1_vector zps, then take the (TBD: WEIGHTED?) mean
    print 'Compiling ZPs'
    zeropoints = dict()
    id_string = id_strings[0]
    for imageid in imagelist:
        zp_tot = 0.
        zp_n = 0
        for pix_id in range(npix_wobjs):
            pix = pix_wobjs[pix_id]
            if imageid in image_id_dicts[pix][nid] and pix_id in pix_id_dict2[0] and (p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]] is not None) and not np.isnan(p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]]):
                zp_tot += p_vectors[pix][nid][image_id_dicts[pix][nid][imageid]] + p1_vector[pix_id_dict2[0][pix_id]] 
                zp_n += 1
        if zp_n > 0:
            zp_tot /= zp_n
            zeropoints[imageid] = zp_tot
            # if there's no label for this data, give it #999
            if imageid not in superlabels:
                superlabels[imageid] = 999
            
    # if there is no p1_vector solution (if all instances of an imagid were outliers) then there is no saved zp.
    zeropoints_tot.append(zeropoints)
    labels_tot = [superlabels]
    
    # are there more id_strings?
    # make all labels the same
    if len(id_strings)>1:
        for nid,id_string in enumerate(id_strings[1:]):
            nid1 = nid+1 # because enumerate starts from zero
            zeropoints = dict()
            labels = dict()
            imagelist1 = []
            for pix in range(npix):
                if image_id_dicts[pix] is not None:
                    imagelist1.extend(image_id_dicts[pix][nid1].keys())
            imagelist1 = list(set(imagelist1))
            for imageid in imagelist1:
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
                labels[imageid] = 0
            zeropoints_tot.append(zeropoints)
            labels_tot.append(labels)
    
    # put the outlier info in the right format
    outliers_tot.append(outliers_nid)
    for nid in range(len(id_strings)-1):
        outliers_tot.append([])
    
    return zeropoints_tot, outliers_tot, labels_tot


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
    
    # calibrate!
    
    # use the zp_phots as a starting point to our calibration
    zp_phot_labels = {}
    for nc in range(63):
        zp_phot_labels[nc] = 0
    precal = [[zp_phots, 'ccd', 1, np.median(zp_phots.values()), [], zp_phot_labels]]
    
    # initialize the color term structure, which will be filled up as needed during the calibration
    colorterm_class = {}
    
    for calibration in config['calibrations']:
        print "\nStarting calibration on ids %s, operands %s\n" %(calibration['id_strings'], calibration['operands'])
        success = False
        results = None
        while not success:
            results = nebencalibrate( band, calibration['nside'], config['general']['nside_file'], calibration['id_strings'], calibration['operands'], precal, config['general']['globals_dir'], max_dets=calibration['max_dets'], ra_min=config['general']['ra_min'], ra_max=config['general']['ra_max'], dec_min=config['general']['dec_min'], dec_max=config['general']['dec_max'], use_color_terms=config['general']['use_color_terms'], mag_classes=colorterm_class )
            if results == 'degrade':
                if calibration['nside'] == 1:
                    calibration['nside'] = 0
                elif calibration['nside'] == 0:
                    print "Giving up!"
                    exit(1)
                calibration['nside'] /= 2
                print "Now trying nside=%d" %calibration['nside']
            else:
                success = True
        
        new_zps, new_outliers, new_labels = results
        for i in range(len(calibration['id_strings'])):
            outfile = open( calibration['outfilenames'][i], 'w' )
            for zp_id in new_zps[i].keys():
                outfile.write( "%d %e %d\n" %(zp_id, new_zps[i][zp_id], new_labels[i][zp_id]) );
            outfile.close()
        
        # so that we use these as input to the following calibration(s)
        for i in range(len(calibration['id_strings'])):
            precal.append([new_zps[i], calibration['id_strings'][i], calibration['operands'][i], np.median(new_zps[i].values()), new_outliers[i], new_labels[i]])
    
    # now do an ubercalibration to the standards.
    if config['general']['use_standards']:
        std_zps = std_ubercal( band, precal, config['general'] )
        
        outfile = open( config['general']['stdzp_outfilename'], 'w' )
        for label in std_zps:
            outfile.write( "%d %e\n" %(label, std_zps[label] ) )
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
    calibrate_by_filter( config )



if __name__ == '__main__':
    main()


