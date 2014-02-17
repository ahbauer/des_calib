import os
import time
import copy
import shutil
import math
import datetime
import re

import sys
#sys.path.append("/Users/bauer/software/PAUdm/trunk/src/")

import numpy
from numpy.linalg import solve
from numpy.linalg import norm
from scipy.sparse import linalg
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy import weave
import pyfits

from rtree import index

#from pipeline.pixelsim import wcsUtils

init_cat = 0
make_fits = 0
resid_map = 1
stats_pre = 1
stats_post = 1
obj_zps = 0

use_colorterms = 0
#filters = ['r','g','i','z']  # in rough order of depth
filters = ['g']  # in rough order of depth
filt = 'g'

C0_A62     = False
CR1_A62    = False
C9_A62     = False
CCDXY_A62  = False
CCDXY9_A62 = False
CR4_A62    = False
SUPERPIX   = True

input_path = "/Users/bauer/surveys/DES/starflats/december"


# We want to fit iteratively, always starting with 1 ZP per CCD (C0_A62)
fits = []
C0_A62_FIT     = False
CR1_A62_FIT    = False
C9_A62_FIT     = False
CCDXY_A62_FIT  = False
CCDXY9_A62_FIT = False
CR4_A62_FIT    = False
SUPERPIX_FIT   = False

if C0_A62:
    fits.append('C0_A62')
    C0_A62_FIT = True
if CR1_A62:
    fits.append('C0_A62')
    fits.append('CR1_A62')
    C0_A62_FIT = True
    CR1_A62_FIT = True
if C9_A62:
    fits.append('C0_A62')
    fits.append('C9_A62')
    C0_A62_FIT = True
    C9_A62_FIT = True
if CCDXY_A62:
    fits.append('C0_A62')
    fits.append('CCDXY_A62')
    C0_A62_FIT = True
    CCDXY_A62_FIT = True
if CCDXY9_A62:
    fits.append('C0_A62')
    fits.append('CCDXY_A62')
    fits.append('CCDXY9_A62')
    C0_A62_FIT = True
    CCDXY_A62_FIT = True
    CCDXY9_A62_FIT = True
if CR4_A62:
    fits.append('C0_A62')
    fits.append('CR4_A62')
    C0_A62_FIT = True
    CR4_A62_FIT = True
if SUPERPIX:
    fits.append('SUPERPIX')
    SUPERPIX_FIT = True

find_spix = """
using namespace std;
spix = (int)(4.0*(int)(y/512.) + (int)(x/512.) + 32.0*ccd);
"""
    
class global_obj(object):
    
    def __init__(self, detection_obj, f):
        if f not in filters:
            print "Error, global object's detection's filter = %s" %filter
            exit(1)
        self.detections = dict()
        for f1 in filters:
            self.detections[f1] = []
        self.detections[f].append(detection_obj)
        self.ra = detection_obj.ra
        self.dec = detection_obj.dec
        self.mags = dict()
        self.mag_errors = dict()
        for f in filters:
            self.mags[f] = 0.0
            self.mag_errors[f] = 0.0

class detection_obj(object):
    
    def __init__(self, r, d, m, em, i, j, n, fi, fj, img):
        self.ra = r
        self.dec = d
        self.mag = m
        self.mag_error = em
        if( self.mag_error < 0.01 ):
            self.mag_error = 0.01
        self.x = i
        self.y = j
        self.ccd_num = n
        self.amp_num = n
        if( self.x > 1024. ):
            self.amp_num *= 2
        self.fp_x = fi
        self.fp_y = fj
        self.img_num = img
        self.zp = 0
        self.spix = 0

class global_objs_list(object):
    
    def __init__(self):
        self.global_objs = {}
        self.global_list = []
        
    def add(self, obj, f):
        
        # not paying attention to objs right near degree borders
        # we have the need for speed!
        bin1 = int(4*obj.ra)
        bin2 = int(4*obj.dec)
        if bin1 not in self.global_objs:
            self.global_objs[bin1] = {}
        if bin2 not in self.global_objs[bin1]:
            self.global_objs[bin1][bin2] = []

        found = 0
        ra_cutoff = 0.00028*math.cos(obj.dec*math.pi/180.)
        mid = int(len(self.global_objs[bin1][bin2])/2)
        while( mid > 0 and obj.ra < self.global_objs[bin1][bin2][mid].ra ):
            mid = int(mid/2)
        for g in range(mid, len(self.global_objs[bin1][bin2])):
            go = self.global_objs[bin1][bin2][g]
            if( go.ra-obj.ra > ra_cutoff ):
                break
            if( abs(obj.dec-go.dec) < 0.00028 ):
                if( abs(obj.ra-go.ra) < ra_cutoff ):
                    raterm = (obj.ra-go.ra)/math.cos(go.dec*math.pi/180.)
                    if( raterm*raterm + (obj.dec-go.dec)*(obj.dec-go.dec) < 7.716e-8 ): # 1 arcsec
                        found = 1
                        go.detections[f].append(obj)
                        #print "match %f %f %f %f" %(obj.ra, obj.dec, go.ra, go.dec)
        if found == 0:
            new_global_obj = global_obj(obj, f)
            #if len(self.global_objs) == 0:
            #    self.global_objs.append(new_global_obj)
            #    return
            mid = int(len(self.global_objs[bin1][bin2])/2)
            while( mid > 0 and obj.ra < self.global_objs[bin1][bin2][mid].ra ):
                mid = int(mid/2)
            for g in range(mid, len(self.global_objs[bin1][bin2])):
                if( obj.ra < self.global_objs[bin1][bin2][g].ra ):
                    self.global_objs[bin1][bin2].insert(g, new_global_obj)
                    return 0
            self.global_objs[bin1][bin2].append(new_global_obj)            
            return 0
        return 1
            
    def consolidate(self):
        for bin1 in self.global_objs:
            for bin2 in self.global_objs[bin1]:
                for g in self.global_objs[bin1][bin2]:
                    self.global_list.append(g)



def match_objs( filt ):

    # read in the CCD positions in the focal plane
    posfile = open("/Users/bauer/surveys/DES/ccdPos-v2.par", 'r')
    posarray = posfile.readlines()
    fp_xs = []
    fp_ys = []
    for i in range(21,83,1):
        entries = posarray[i].split(" ")
        fp_xs.append(66.6667*(float(entries[4])-211.0605) - 1024) # in pixels
        fp_ys.append(66.6667*float(entries[5]) - 2048)
    print "parsed focal plane positions for %d ccds" %len(fp_xs)

    print "reading from directory %s" %input_path
    dir_list = os.listdir(input_path)
    files = []
    for file in dir_list:
        if re.match("DECam", file) is not None and re.search("fits$", file) is not None:
            files.append(file)

    global_objs = global_objs_list()
    file_index = 0
    filter_line = 45 # november = 42
    for file in files:
        hdulist = pyfits.open(os.path.join(input_path,file))
        print "file %s has %d extensions:" %(file, len(hdulist))
        #print hdulist[1].data[0]
        header = ''
        for char in hdulist[1].data[0]:
            header += char
        #print header[42]
        filt_matches = re.match("FILTER\s+=\s+\'(\S)", header[filter_line])
        f = filt_matches.group(1)
        #print "filter %s" %f
        if( f != filt ):
            continue
        for i in range( 2,len(hdulist), 2 ):
            ngood = 0
            nmatch = 0
            for row in hdulist[i].data:
                if( row[11] > 4.5 or row[17] > 0.1 or row[14] > 3 or row[33]<0.9 ): # flux_radius magerr_auto flags class_star
                    continue
                ccd_num = (i-2)/2
                detection = detection_obj(row[1],row[2],row[16],row[17],row[7],row[8],ccd_num,row[7]+fp_xs[ccd_num],row[8]+fp_ys[ccd_num],file_index)
                nmatch += global_objs.add(detection, f)
                ngood += 1
            print "done with extension %d, %d new detections, %d new matches" %(i, ngood, nmatch)
        file_index = file_index+1
        #if file_index > 4:
        #    break

    print "consolidating..."
    global_objs.consolidate()
    print "done!"

    outfilename = "starflatout_%s" %filt
    outfile = open( outfilename, 'w' )
    for go in global_objs.global_list:
        if len(go.detections[filt]) > 1:
            outfile.write( "%f %f " %(go.ra, go.dec) )
            for do in go.detections[filt]:
                outfile.write( "%f %f %d %f %f %f %f %d " %(do.x, do.y, do.ccd_num, do.fp_x, do.fp_y, do.mag, do.mag_error, do.img_num) )
            outfile.write( "\n" )



def read_list( filt, go_list, min_d=2 ):
    listname = "starflatout_%s" %filt
    print "reading from %s" %listname
    listfile = open(listname, 'r')
    #listarray = listfile.readlines()
    n_imgs = 0
    for line in listfile:
#    for line in listarray:
        vals = line.split(" ")
        n_d = (len(vals) - 2)/8
        if( int(n_d) != n_d ):
            print "Problem, number of detections = %f, not an integer..." %n_d
            exit(1)
        if( n_d < min_d ):
            continue
        img_num = int(vals[9])
        det = detection_obj( float(vals[0]), float(vals[1]), float(vals[7]), float(vals[8]), float(vals[2]), float(vals[3]), float(vals[4]), float(vals[5]), float(vals[6]), img_num )
        go = global_obj(det, filt)
        if( img_num > n_imgs ):
            n_imgs = img_num
        for i in range(1,n_d,1):
            img_num = int(vals[1+8*i+8])
            if( img_num > n_imgs ):
                n_imgs = img_num
            det = detection_obj( float(vals[0]), float(vals[1]), float(vals[1+8*i+6]), float(vals[1+8*i+7]), float(vals[1+8*i+1]), float(vals[1+8*i+2]), float(vals[1+8*i+3]), float(vals[1+8*i+4]), float(vals[1+8*i+5]), img_num )
            go.detections[filt].append(det)
        go_list.global_list.append(go)
    return (n_imgs+1)


def print_stats( go_list, filt ):
    for go in go_list.global_list:
        mean_mag = 0
        for det in go.detections[filt]:
            mean_mag += det.mag
        mean_mag /= len(go.detections)
        for det in go.detections[filt]:
            print "%f %f" %(det.mag-mean_mag, det.mag_error)


def calc_stats( go_list, filt ):
    rms = 0.0
    n = 0.0
    for go in go_list.global_list:
        mean_mag = 0
        for det in go.detections[filt]:
            mean_mag += det.mag
        mean_mag /= len(go.detections[filt])
        for det in go.detections[filt]:
            rms += (det.mag-mean_mag)*(det.mag-mean_mag)
            n += 1.0
    rms = numpy.sqrt(rms/n)
    print "RMS = %e" %rms


def clip_objs( go_list, filt ):
    for g, go in enumerate(go_list.global_list):
        mean_mag = 0
        for det in go.detections[filt]:
            mean_mag += det.mag
        mean_mag /= len(go.detections[filt])
        for d, det in enumerate(go.detections[filt]):
            if( abs(det.mag-mean_mag) > 0.5 ):
                #det.mag = -1
                go.detections[filt].pop(d)
        if len(go.detections[filt]) < 2:
            go_list.global_list.pop(g)


def print_mags( go_list ):
    for go in go_list.global_list:
        for f in ['g', 'r', 'i', 'z']:
            outstring = "%f %f " %(go.mags[f], go.mag_errors[f])
            sys.stdout.write(outstring)
        sys.stdout.write("\n")


def apply_colorterms( go_list ):
    for go in go_list.global_list:
        go.mags['g'] -= 0.4
        go.mags['r'] -= 0.1
        go.mags['z'] -= 0.35
        
        gmr = go.mags['g'] - go.mags['r']
        imz = go.mags['i'] - go.mags['z']
        
        # # average DECal
        go.mags['g'] += -0.0122*gmr*gmr + 0.115*gmr + 0.00388
        go.mags['r'] += 0.00185*gmr*gmr + 0.0798*gmr + 0.00039
        go.mags['i'] += -0.0603*imz*imz + 0.456*imz + 0.00687
        go.mags['z'] += 0.0515*imz*imz + 0.105*imz - 0.00554


def combine_filts( global_objects_lists ):

    match_radius = 1.0/3600.

    # start with the global objects from the first filter in the list. (should be deep.)
    final_global_objects_list = copy.deepcopy(global_objects_lists[filters[0]])

    # make an rtree for the first filter in the list of filters 
    objmap = index.Index()
    for gid, go in enumerate(global_objects_lists[filters[0]].global_list):
        objmap.insert(gid, (go.ra,go.dec,go.ra,go.dec))

    print "Made the tree for filter %s, with %d global objects" %(filters[0], len(global_objects_lists[filters[0]].global_list))

    # now match the other filters' data to the tree.
    for fid in range(1,len(filters)):
        f = filters[fid]
        print "Starting filter %s, with %d global objects" %(f, len(global_objects_lists[f].global_list))
        for go in global_objects_lists[f].global_list:
            ra_match_radius = match_radius/math.cos(go.dec*math.pi/180.)
            dec_match_radius = match_radius
            match_area = (go.ra-ra_match_radius, go.dec-dec_match_radius,
                           go.ra+ra_match_radius, go.dec+dec_match_radius)

            go_candidates = list(objmap.intersection(match_area, objects=True)) #go_candidates is a list of map ids

            if len(go_candidates) == 0:
                continue
            if len(go_candidates) == 1:
                glo_candidate = go_candidates[0]
            if len(go_candidates)>1:
                #Among all the detections, select the nearest one to the global object
                glo_candidate = list(objmap.nearest((go.ra,go.dec,go.ra,go.dec), objects=True))[0]
           
            final_global_objects_list.global_list[glo_candidate.id].detections[f] = go.detections[f]
        print "Finished adding filter %s" %f

    print "Averaging magnitudes"
    m_tot = dict()
    n_tot = dict()
    for g in range(len(final_global_objects_list.global_list)):
        for f in filters:
            m_tot[f] = 0.
            n_tot[f] = 0.
            for det in final_global_objects_list.global_list[g].detections[f]:
                if( det.mag_error < 0.02 ):
                    det.mag_error = 0.02
                # actually, just take the flat-out mean.
                det.mag_error = 0.02
                m_tot[f] += det.mag/(det.mag_error*det.mag_error)
                n_tot[f] += 1.0/(det.mag_error*det.mag_error)
            if n_tot[f] > 0.:
                final_global_objects_list.global_list[g].mags[f] = m_tot[f]/n_tot[f]
                final_global_objects_list.global_list[g].mag_errors[f] = numpy.sqrt(1.0/n_tot[f])
    
    print "Finished combining filters"
    
    return copy.deepcopy(final_global_objects_list)


def calc_star_flat( global_objects, n_imgs, f ):

    print "Calculating star flat from %d images for filt %s" %(n_imgs, f)

    n_ccds = 62
    n_amps = 2*n_ccds
    counts_per_ccd = numpy.zeros(n_ccds)
    
    # make the matrices that we want to use in the ubercal
    # we will use a slightly different model from that of sdss
    # also, this is the part that is different between our star flat code and our ubercal
    # m_true = m_obs + f(x,y) + A(chip)
    # (will this capture the offsets for different filters?)
    # where x,y are coordinates on the focal plane
    # the "weighing the giants" code uses the product of 3rd order chebyshev polynomials for f(x,y)
    # but we can't because that makes this not a linear system!  instead we must do
    # f(x,y) = c0*x + c1*y + c2*x*y + c3*x^2 + c4*y^2 + c5*x^2*y + c6*x*y^2 + c7*x^3 + c8*y^3
    # the most general would be
    # m_true = m_obs + f(x,y,chip)
    # but that may be too many degrees of freedom, and wouldn't take advantage of any spherical 
    # symmetry in the focal plane distortion.  and wouldn't leave room for zp_phot...
    
    # matrix A:
    # has n_obs rows and n_par columns
    
    # matrix b:
    # b_j = sum_k( m_k * I_k ) - m_j
    # where there are j different observations (n_obs) (total, for any objects)        
    
    # matrix p: fit parameters: p^T = ( c0, c1, .. c8, A1, A2, .. An_ccds )
    # c are the x,y polynomial coefficients
    # A are the constants, one per chip, which are added to f(x,y)
    # so, n_par = 9 + n_ccds 
          
    # how big might our array and matrix get?  i.e., what's a maximum n_obs?
    # this overestimates because it doesn't take into account the filter tray of the members.
    # this makes this faster but will waste space in memory.  the right choice?
    max_n_obs = 0
    for global_object in global_objects:
        max_n_obs = max_n_obs + len(global_object.detections[f])
    
    print "Maximum possible number of observations: %d" %max_n_obs
    
    n_par = dict()
    n_par['C9_A62'] = 9 + n_amps
    n_par['C0_A62'] = 1 + n_amps
    n_par['CCDXY_A62'] = n_amps + 2*n_ccds
    n_par['CCDXY9_A62'] = 9 + n_amps + 2*n_ccds
    n_par['CR4_A62'] = 4 + n_amps
    n_par['CR1_A62'] = 1 + n_amps
    n_par['SUPERPIX'] = 1 + 32*n_ccds # 512pix boxes
    
    
    p_guess = dict()
    a_matrix = dict()
    for fit in fits:
        p_guess[fit] = numpy.zeros(n_par[fit]+n_imgs)
        a_matrix[fit] = lil_matrix((max_n_obs, n_par[fit]+n_imgs))
    b_vector = numpy.zeros(max_n_obs)
    c_matrix = lil_matrix((max_n_obs, max_n_obs))
    
    obs_count = 0
    array = dict()
    for gindex, global_object in enumerate(global_objects):
        
        if( gindex % 5000 == 0 ):
            print "Working on global object #%d..." %gindex
        
        # calculate the sums over all measurements needed in the individual matrix rows
        sum_invsigma2 = 0.0
        #for det in global_object.detections[f]:
        #    # print "%f %f" %(det.mag, det.mag_error)
        #    sum_invsigma2 += 1.0/(det.mag_error*det.mag_error)
        sum_m_i = 0.0

        sum_for_cs = dict()
        #sum_for_cs_i = dict()
        sum_for_as = numpy.zeros(n_amps)
        #sum_for_as_i = None
        for fit in fits:
            sum_for_cs[fit] = numpy.zeros(n_par[fit] - n_amps)
        if SUPERPIX_FIT:
            sum_for_cs['SUPERPIX'] = numpy.zeros(1)
            sum_for_as = numpy.zeros(32*n_ccds)

        sum_for_zps = numpy.zeros(n_imgs)
        
        # if SUPERPIX_FIT:
        #     mags = numpy.array([det.mag for det in global_object.detections[f]])
        #     mag_errs = numpy.array([det.mag_error for det in global_object.detections[f]])
        #     i = (1.0 /(mag_errs**2))
        #     sum_m_i = (mags.sum() * i).sum()
        #     sum_for_cs['SUPERPIX'][0] = i.sum()
        #     sum_invsigma2 = i.sum()
        #     xs = numpy.array([det.x for det in global_object.detections[f]])
        #     ys = numpy.array([det.y for det in global_object.detections[f]])
        #     ccd_num = numpy.array([det.ccd_num for det in global_object.detections[f]]).astype(int)
        #     counts_per_ccd[ccd_num] += 1
        #     spixs = (4*numpy.floor(ys/512.) + numpy.floor(xs/512.) + 32*ccd_num).astype(int)
        #     sum_for_as[spixs] += i
        #     imgs = numpy.array([det.img_num for det in global_object.detections[f]]).astype(int)
        #     sum_for_zps[imgs] += i
        # 
        # 
        # if not SUPERPIX_FIT:

        for det in global_object.detections[f]:
                
            counts_per_ccd[det.ccd_num] += 1

            sum_invsigma2 += 1.0/(det.mag_error*det.mag_error)
    
            i = (1.0/(det.mag_error*det.mag_error)) #/sum_invsigma2
            sum_m_i += det.mag*i

            if C9_A62_FIT:
                fp_x2 = det.fp_x*det.fp_x
                fp_y2 = det.fp_y*det.fp_y
                sum_for_cs['C9_A62'] += numpy.array([det.fp_x*i, det.fp_y*i, det.fp_x*det.fp_y*i, fp_x2*i, fp_y2*i, fp_x2*det.fp_y*i, det.fp_x*fp_y2*i, fp_x2*det.fp_x*i, fp_y2*det.fp_y*i])
            if CR4_A62_FIT:
                fp_r2 = det.fp_x*det.fp_x + det.fp_y*det.fp_y
                fp_r = numpy.sqrt(fp_r2)
                sum_for_cs['CR4_A62'] += numpy.array([fp_r*i, fp_r2*i, fp_r2*fp_r*i, fp_r2*fp_r2*i])
            if CR1_A62_FIT:
                fp_r2 = det.fp_x*det.fp_x + det.fp_y*det.fp_y
                fp_r = numpy.sqrt(fp_r2)
                sum_for_cs['CR1_A62'] += numpy.array([fp_r*i])
            if C0_A62_FIT:
                sum_for_cs['C0_A62'][0] += i
            if CCDXY_A62_FIT:
                sum_for_cs['CCDXY_A62'][2*det.ccd_num] += det.x*i;
                sum_for_cs['CCDXY_A62'][2*det.ccd_num+1] += det.y*i;
            if CCDXY9_A62_FIT:
                fp_x2 = det.fp_x*det.fp_x
                fp_y2 = det.fp_y*det.fp_y
                sum_for_cs['CCDXY9_A62'][0] += det.fp_x*i
                sum_for_cs['CCDXY9_A62'][1] += det.fp_y*i
                sum_for_cs['CCDXY9_A62'][2] += det.fp_x*det.fp_y*i
                sum_for_cs['CCDXY9_A62'][3] += fp_x2*i
                sum_for_cs['CCDXY9_A62'][4] += fp_y2*i
                sum_for_cs['CCDXY9_A62'][5] += fp_x2*det.fp_y*i
                sum_for_cs['CCDXY9_A62'][6] += det.fp_x*fp_y2*i
                sum_for_cs['CCDXY9_A62'][7] += fp_x2*det.fp_x*i
                sum_for_cs['CCDXY9_A62'][8] += fp_y2*det.fp_y*i
                sum_for_cs['CCDXY9_A62'][9 + 2*det.ccd_num] += det.x*i;
                sum_for_cs['CCDXY9_A62'][9 + 2*det.ccd_num+1] += det.y*i
            if SUPERPIX_FIT:
                sum_for_cs['SUPERPIX'][0] += i #numpy.array([i])

        
            if SUPERPIX_FIT:
                det.spix = int(4*numpy.floor(det.y/512.) + numpy.floor(det.x/512.) + 32*det.ccd_num)
                sum_for_as[det.spix] += i
            else:
                sum_for_as[det.amp_num] += i

            sum_for_zps[det.img_num] += i


        # now divide everything by sum_invsigma2 since that should be part of "i"
        for fit in fits:
            sum_for_cs[fit] /= sum_invsigma2
            #sum_for_cs_i[fit] += sum_for_cs[fit][numpy.newaxis,:]

        sum_for_as /= sum_invsigma2
        sum_for_zps /= sum_invsigma2
        sum_m_i /= sum_invsigma2

        # now make the matrix rows using the info from this measurement
        for det in global_object.detections[f]:
            
            # make an element of the b vector
            b_vector[obs_count] = (det.mag - sum_m_i)
            #print "%f %f %f" %(x, y, b_vector[obs_count])
            
        
            sum_for_cs_i = dict()
            if C9_A62_FIT:
                fp_x2 = det.fp_x*det.fp_x
                fp_y2 = det.fp_y*det.fp_y
                sum_for_cs_i['C9_A62'] = sum_for_cs['C9_A62'] - numpy.array([det.fp_x, det.fp_y, det.fp_x*det.fp_y, fp_x2, fp_y2, fp_x2*det.fp_y, det.fp_x*fp_y2, fp_x2*det.fp_x, fp_y2*det.fp_y])                
            if CR4_A62_FIT:
                fp_r2 = fp_x2 + fp_y2
                fp_r = numpy.sqrt(fp_r2)
                sum_for_cs_i['CR4_A62'] = sum_for_cs['CR4_A62'] - numpy.array([fp_r, fp_r2, fp_r2*fp_r, fp_r2*fp_r2])
            if CR1_A62_FIT:
                fp_r2 = fp_x2 + fp_y2
                fp_r = numpy.sqrt(fp_r2)
                sum_for_cs_i['CR1_A62'] = sum_for_cs['CR1_A62'] - numpy.array([fp_r])
            if C0_A62_FIT:
                sum_for_cs_i['C0_A62'] = sum_for_cs['C0_A62'] - numpy.array([1])
            if CCDXY_A62_FIT:
                sum_for_cs_i['CCDXY_A62'] = sum_for_cs['CCDXY_A62'].copy()
                sum_for_cs_i['CCDXY_A62'][2*det.ccd_num] = sum_for_cs_i['CCDXY_A62'][2*det.ccd_num] - det.x;
                sum_for_cs_i['CCDXY_A62'][2*det.ccd_num+1] = sum_for_cs_i['CCDXY_A62'][2*det.ccd_num+1] - det.y;
            if CCDXY9_A62_FIT:
                fp_x2 = det.fp_x*det.fp_x
                fp_y2 = det.fp_y*det.fp_y
                sum_for_cs_i['CCDXY9_A62'] = sum_for_cs['CCDXY9_A62'].copy()
                sum_for_cs_i['CCDXY9_A62'][0] = sum_for_cs_i['CCDXY9_A62'][0] - det.fp_x
                sum_for_cs_i['CCDXY9_A62'][1] = sum_for_cs_i['CCDXY9_A62'][1] - det.fp_y
                sum_for_cs_i['CCDXY9_A62'][2] = sum_for_cs_i['CCDXY9_A62'][2] - det.fp_x*det.fp_y
                sum_for_cs_i['CCDXY9_A62'][3] = sum_for_cs_i['CCDXY9_A62'][3] - fp_x2
                sum_for_cs_i['CCDXY9_A62'][4] = sum_for_cs_i['CCDXY9_A62'][4] - fp_y2
                sum_for_cs_i['CCDXY9_A62'][5] = sum_for_cs_i['CCDXY9_A62'][5] - fp_x2*det.fp_y
                sum_for_cs_i['CCDXY9_A62'][6] = sum_for_cs_i['CCDXY9_A62'][6] - det.fp_x*fp_y2
                sum_for_cs_i['CCDXY9_A62'][7] = sum_for_cs_i['CCDXY9_A62'][7] - fp_x2*det.fp_x
                sum_for_cs_i['CCDXY9_A62'][8] = sum_for_cs_i['CCDXY9_A62'][8] - fp_y2*det.fp_y
                sum_for_cs_i['CCDXY9_A62'][9 + 2*det.ccd_num] = sum_for_cs_i['CCDXY9_A62'][9 + 2*det.ccd_num] - det.x;
                sum_for_cs_i['CCDXY9_A62'][9 + 2*det.ccd_num+1] = sum_for_cs_i['CCDXY9_A62'][9 + 2*det.ccd_num+1] - det.y;
            if SUPERPIX_FIT:
                sum_for_cs_i['SUPERPIX'] = sum_for_cs['SUPERPIX'] - numpy.array([1]) # [sum_for_cs['SUPERPIX'][0] - 1]
            
            sum_for_as_i = sum_for_as.copy()
            if SUPERPIX_FIT:
                sum_for_as_i[det.spix] -= 1.0
            else:
                sum_for_as_i[det.amp_num] -= 1.0
            sum_for_zps_i = sum_for_zps.copy()
            sum_for_zps_i[det.img_num] -= 1.0
            # if abs(sum_for_as_i[det.amp_num]) < 1.0e-10:
            #     sum_for_as_i[det.amp_num] = 0.
            # if abs(sum_for_zps_i[det.img_num]) < 1.0e-10:
            #     sum_for_zps_i[det.img_num] = 0.
        
            # make a row of A, with length n_par
            for fit in fits:
                array[fit] = numpy.append(sum_for_cs_i[fit], sum_for_as_i)
                array[fit] = numpy.append(array[fit], sum_for_zps_i)
                # array[fit] = numpy.zeros(len(sum_for_cs_i[fit]) + len(sum_for_as_i) + len(sum_for_zps_i))
                # array[fit][0:len(sum_for_cs_i[fit])] = sum_for_cs_i[fit]
                # array[fit][len(sum_for_cs_i[fit]):len(sum_for_cs_i[fit])+len(sum_for_as_i)] = sum_for_as_i
                # array[fit][len(sum_for_cs_i[fit])+len(sum_for_as_i):len(array[fit])] = sum_for_zps_i

                # for i in range(len(array[fit])):
                #     a_matrix[fit][obs_count,i] = array[fit][i]
                a_matrix[fit][obs_count,:] = array[fit]
            
            c_matrix[obs_count,obs_count] = 1.0/(det.mag_error*det.mag_error)
            
            obs_count = obs_count+1
        
    print "Number of observations used: %d" %obs_count
    print "Number per CCD:"
    print counts_per_ccd
    
    # now truncate the vectors and matrices to be of size obs_count
    if( obs_count < max_n_obs ):
        for fit in sum_for_cs_i:
            a_matrix[fit] = a_matrix[fit][0:obs_count,0:(n_par[fit]+n_imgs)]
        b_vector = b_vector[0:obs_count]
        c_matrix = c_matrix[0:obs_count,0:obs_count]
    
    # now solve for p!
    # following the appropach in the padmanabhan ubercal paper, equation 14
    # put this in terms of ax = b, solve for x
    # since we know that c_matrix is diagonal, let's do a dumb inversion
    for k in range(obs_count):
        c_matrix[k,k] = 1.0/c_matrix[k,k]
    #subterm = numpy.dot(a_matrix.transpose(), numpy.linalg.inv(c_matrix))
    p_vector = dict()
    for fit in fits:
        
        # fill in the 62 ZPs from C0_A62 as the initial guess
        if fit in ( 'C9_A62', 'CCDXY_A62', 'CR1_A62', 'CR4_A62' ):
            print "Using p_guess from C0_A62"
            p_guess[fit][n_par[fit]-n_amps:(n_par[fit]+n_imgs)] = p_vector['C0_A62'][1:n_par['C0_A62']+n_imgs]
        # fill in the results from C9_A62 (could use CCDXY_A62 instead, but we expect FP terms to dominate?)
        if( fit == 'CCDXY9_A62' ):
            print "Using p_guess from C9_A62"
            p_guess[fit][9:(n_par[fit]+n_imgs)] = p_vector['C9_A62'][1:n_par['C9_A62']+n_imgs]
        
        print "Calculating fit for %s" %fit
        subterm = (a_matrix[fit].transpose()).dot(c_matrix)
        # print subterm
        termA = subterm.dot(a_matrix[fit])
        termB = subterm.dot(b_vector)
        print "Shapes:"
        print termA.shape
        print termB.shape
        # p_vector = linalg.spsolve(termA.tocsc(),termB)
        if( fit == 'C0_A62' or fit == 'SUPERPIX' ):
            p_vector[fit] = linalg.bicgstab(termA,termB)[0]
        else:
            p_vector[fit] = linalg.bicgstab(termA,termB,p_guess[fit])[0]
    # err = norm(p_vector-p_vector2)
        print "Solution:"
        print p_vector[fit].shape
        print p_vector[fit]
    # print "error:"
    # print err
    
    # now delete things to free up memory
    p_guess = []
    a_matrix = []
    b_vector = []
    c_matrix = []

    fit = fits[len(fits)-1]

    if stats_post:
        rms = 0.0
        n = 0.0
        for gindex, global_object in enumerate(global_objects):
            mean_mag = 0.
            mags = []
            for det in global_object.detections[f]:
                zp = 0.0
                fp_r2 = det.fp_x*det.fp_x + det.fp_y*det.fp_y
                fp_r = numpy.sqrt(fp_r2)
                term9 = p_vector[fit][n_par[fit]-n_amps+det.amp_num]
                if( C9_A62 ):
                    zp = p_vector[fit][0]*det.fp_x + p_vector[fit][1]*det.fp_y + p_vector[fit][2]*det.fp_x*det.fp_y + p_vector[fit][3]*det.fp_x*det.fp_x + p_vector[fit][4]*det.fp_y*det.fp_y + p_vector[fit][5]*det.fp_x*det.fp_x*det.fp_y + p_vector[fit][6]*det.fp_x*det.fp_y*det.fp_y + p_vector[fit][7]*det.fp_x*det.fp_x*det.fp_x + p_vector[fit][8]*det.fp_y*det.fp_y*det.fp_y + term9
                elif( CR4_A62 ):
                    zp = p_vector[fit][0]*fp_r + p_vector[fit][1]*fp_r2 + p_vector[fit][2]*fp_r2*fp_r + p_vector[fit][3]*fp_r2*fp_r2 + term9
                elif( CR1_A62 ):
                    zp = p_vector[fit][0]*fp_r + term9
                elif( C0_A62 ):
                    zp = term9
                elif( CCDXY_A62 ):
                    zp = p_vector[fit][2*det.ccd_num]*det.x + p_vector[fit][2*det.ccd_num+1]*det.y + term9
                elif( CCDXY9_A62 ):
                    zp = p_vector[fit][0]*det.fp_x + p_vector[fit][1]*det.fp_y + p_vector[fit][2]*det.fp_x*det.fp_y + p_vector[fit][3]*det.fp_x*det.fp_x + p_vector[fit][4]*det.fp_y*det.fp_y + p_vector[fit][5]*det.fp_x*det.fp_x*det.fp_y + p_vector[fit][6]*det.fp_x*det.fp_y*det.fp_y + p_vector[fit][7]*det.fp_x*det.fp_x*det.fp_x + p_vector[fit][8]*det.fp_y*det.fp_y*det.fp_y + p_vector[fit][9+2*det.ccd_num]*det.x + p_vector[fit][9+2*det.ccd_num+1]*det.y + term9
                elif( SUPERPIX ):
                    zp = p_vector[fit][1+det.spix]
                
                det.zp = zp
                mag = det.mag + det.zp
                mags.append(mag)
                mean_mag += mag
            mean_mag /= len(global_object.detections[f])
            for det in global_object.detections[f]:
                # print "%f %f" %(det.mag+det.zp-mean_mag, det.mag_error)
                rms += (det.mag+det.zp-mean_mag)*(det.mag+det.zp-mean_mag)
                n += 1.0
        rms = numpy.sqrt(rms/n)
        print "RMS after flat = %e" %rms
        
    if obj_zps:
        for gindex, global_object in enumerate(global_objects):
            mean_mag = 0.
            mags = []
            for det in global_object.detections[f]:
                zp = 0.0
                fp_r2 = det.fp_x*det.fp_x + det.fp_y*det.fp_y
                fp_r = numpy.sqrt(fp_r2)
                term9 = p_vector[fit][n_par[fit]-n_amps+det.amp_num]
                if( C9_A62 ):
                    zp = p_vector[fit][0]*det.fp_x + p_vector[fit][1]*det.fp_y + p_vector[fit][2]*det.fp_x*det.fp_y + p_vector[fit][3]*det.fp_x*det.fp_x + p_vector[fit][4]*det.fp_y*det.fp_y + p_vector[fit][5]*det.fp_x*det.fp_x*det.fp_y + p_vector[fit][6]*det.fp_x*det.fp_y*det.fp_y + p_vector[fit][7]*det.fp_x*det.fp_x*det.fp_x + p_vector[fit][8]*det.fp_y*det.fp_y*det.fp_y + term9
                elif( CR4_A62 ):
                    zp = p_vector[fit][0]*fp_r + p_vector[fit][1]*fp_r2 + p_vector[fit][2]*fp_r2*fp_r + p_vector[fit][3]*fp_r2*fp_r2 + term9
                elif( CR1_A62 ):
                    zp = p_vector[fit][0]*fp_r + term9
                elif( C0_A62 ):
                    zp = term9
                elif( CCDXY_A62 ):
                    zp = p_vector[fit][2*det.ccd_num]*det.x + p_vector[fit][2*det.ccd_num+1]*det.y + term9
                elif( CCDXY9_A62 ):
                    zp = p_vector[fit][0]*det.fp_x + p_vector[fit][1]*det.fp_y + p_vector[fit][2]*det.fp_x*det.fp_y + p_vector[fit][3]*det.fp_x*det.fp_x + p_vector[fit][4]*det.fp_y*det.fp_y + p_vector[fit][5]*det.fp_x*det.fp_x*det.fp_y + p_vector[fit][6]*det.fp_x*det.fp_y*det.fp_y + p_vector[fit][7]*det.fp_x*det.fp_x*det.fp_x + p_vector[fit][8]*det.fp_y*det.fp_y*det.fp_y + p_vector[fit][9+2*det.ccd_num]*det.x + p_vector[fit][9+2*det.ccd_num+1]*det.y + term9
                elif( SUPERPIX ):
                    zp = p_vector[fit][1+det.spix]
                
                print "%f %f %f %f %f %f" %(det.fp_x, det.fp_y, fp_r, det.mag, zp, term9)
        
                
    if make_fits:

        # now make the fits file that is the flat field
        starflatname = "starflat.fits"
        new_hdu = pyfits.PrimaryHDU()
        new_hdu.header.update("NEXTEND", n_ccds)
        pyfits.writeto(starflatname, data = numpy.array([]), header = new_hdu.header)

        # read in the CCD positions in the focal plane
        posfile = open("/Users/bauer/surveys/DES/ccdPos-v2.par", 'r')
        posarray = posfile.readlines()
        fp_xs = []
        fp_ys = []
        for i in range(21,83,1):
            entries = posarray[i].split(" ")
            fp_xs.append(66.6667*(float(entries[4])-211.0605) - 1024) # in pixels
            fp_ys.append(66.6667*float(entries[5]) - 2048)
        print "parsed focal plane positions for %d ccds" %len(fp_xs)
    
        print "reading from directory %s for WCS info" %input_path
        dir_list = os.listdir(input_path)
        for file in dir_list:
            if re.match("DECam", file) is not None and re.search("fits$", file) is not None:
                break

        hdulist = pyfits.open(os.path.join(input_path,file))
        for i in range( 1,len(hdulist), 2 ): # 1,5,2 ):
            ccd_num = (i-1)/2 # zero ordered
            header_array = ''
            for char in hdulist[i].data[0]:
                header_array += char
            header_string = header_array.tostring()
            # print header_string
            # print type(header_string)
            matches = re.search( "NAXIS1\s+=\s+(\S+)", header_string )
            width = matches.group(1)
            matches = re.search( "NAXIS2\s+=\s+(\S+)", header_string )
            height = matches.group(1)
            matches = re.search( "CRVAL1\s+=\s+(\S+)", header_string )
            crval1 = matches.group(1)
            matches = re.search( "CRVAL2\s+=\s+(\S+)", header_string )
            crval2 = matches.group(1)
            matches = re.search( "CRPIX1\s+=\s+(\S+)", header_string )
            crpix1 = matches.group(1)
            matches = re.search( "CRPIX2\s+=\s+(\S+)", header_string )
            crpix2 = matches.group(1)
            matches = re.search( "CD1_1\s+=\s+(\S+)", header_string )
            cd1_1 = matches.group(1)
            matches = re.search( "CD1_2\s+=\s+(\S+)", header_string )
            cd1_2 = matches.group(1)
            matches = re.search( "CD2_1\s+=\s+(\S+)", header_string )
            cd2_1 = matches.group(1)
            cd2_2 = (re.search( "CD2_2\s+=\s+(\S+)", header_string )).group(1)
            # print "wcs: %s %s %s %s %s %s %s %s %s %s" %(width, height, crval1, crval2, crpix1, crpix2, cd1_1, cd1_2, cd2_1, cd2_2)
            
            ccd_image = pyfits.ImageHDU()
            ccd_image.header = pyfits.Header(txtfile = "/Users/bauer/software/PAUdm/trunk/src/pipeline/pixelsim/config/header_extension.template")
            # now update the CCD header with this information
            ccd_image.header.update('NAXIS1', width)
            ccd_image.header.update('NAXIS2', height)
            ccd_image.header.update('WCSDIM', 2)
            # ccd_image.header.update('CTYPE1', (re.search( "CTYPE1\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('CTYPE2', (re.search( "CTYPE2\s+=\s+(\S+)", header_string )).group(1))
            ccd_image.header.update('CRVAL1', crval1)
            ccd_image.header.update('CRVAL2', crval2)
            ccd_image.header.update('CRPIX1', crpix1)
            ccd_image.header.update('CRPIX2', crpix2)
            ccd_image.header.update('CD1_1', cd1_1)
            ccd_image.header.update('CD1_2', cd1_2)
            ccd_image.header.update('CD2_1', cd2_1)
            ccd_image.header.update('CD2_2', cd2_2)
            # ccd_image.header.update('PV1_0', (re.search( "PV1_0\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV1_1', (re.search( "PV1_1\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV1_2', (re.search( "PV1_2\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV1_3', (re.search( "PV1_3\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV1_4', (re.search( "PV1_4\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV1_5', (re.search( "PV1_5\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV1_6', (re.search( "PV1_6\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV1_7', (re.search( "PV1_7\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV1_8', (re.search( "PV1_8\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV1_9', (re.search( "PV1_9\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV1_10', (re.search( "PV1_10\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV2_0', (re.search( "PV2_0\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV2_1', (re.search( "PV2_1\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV2_2', (re.search( "PV2_2\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV2_3', (re.search( "PV2_3\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV2_4', (re.search( "PV2_4\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV2_5', (re.search( "PV2_5\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV2_6', (re.search( "PV2_6\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV2_7', (re.search( "PV2_7\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV2_8', (re.search( "PV2_8\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV2_9', (re.search( "PV2_9\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('PV2_10', (re.search( "PV2_10\s+=\s+(\S+)", header_string )).group(1))
        
            width_int = int(width)
            height_int = int(height)
            print "dimensions: %d %d" %(width_int, height_int)

            ccd_image.data = numpy.zeros([height_int, width_int])
            data_amp1 = numpy.zeros([height_int, width_int])
            data_amp2 = numpy.zeros([height_int, width_int])

            pix_x_array = numpy.arange(width_int)
            pix_y_array = numpy.arange(height_int)
            pix_xx,pix_yy= numpy.meshgrid(pix_x_array,pix_y_array)
            x_array = pix_x_array + fp_xs[ccd_num]
            y_array = pix_y_array + fp_ys[ccd_num]
            xx,yy = numpy.meshgrid(x_array,y_array)
            xxyy = xx*yy
            xx2 = xx*xx
            yy2 = yy*yy
            xx3 = xx2*xx
            yy3 = yy2*yy
            xx2yy = xx2*yy
            xxyy2 = xx*yy2
            fp_r2 = xx2 + yy2
            fp_r = numpy.sqrt(fp_r2)
            term9_a1 = p_vector[fit][n_par[fit]-n_amps+ccd_num]
            term9_a2 = p_vector[fit][n_par[fit]-n_amps+ccd_num*2]
        
            # don't include the term9s in the plot, to make it smoother looking on a global scale.
            # (unless we're doing c0_a62)
            
            if( C9_A62 ):
                data_amp1 = 10.0**( (p_vector[fit][0]*xx + p_vector[fit][1]*yy + p_vector[fit][2]*xxyy + p_vector[fit][3]*xx2 + p_vector[fit][4]*yy2 + p_vector[fit][5]*xx2yy + p_vector[fit][6]*xxyy2 + p_vector[fit][7]*xx3 + p_vector[fit][8]*yy3)/2.5 ) # + term9_a1)/2.5 )
                data_amp2 = 10.0**( (p_vector[fit][0]*xx + p_vector[fit][1]*yy + p_vector[fit][2]*xxyy + p_vector[fit][3]*xx2 + p_vector[fit][4]*yy2 + p_vector[fit][5]*xx2yy + p_vector[fit][6]*xxyy2 + p_vector[fit][7]*xx3 + p_vector[fit][8]*yy3)/2.5 ) # + term9_a2)/2.5 )
            elif( CR4_A62 ):
                data_amp1 = 10.0**( (p_vector[fit][0]*fp_r + p_vector[fit][1]*fp_r2 + p_vector[fit][2]*fp_r2*fp_r + p_vector[fit][3]*fp_r2*fp_r2)/2.5 ) # + term9_a1)/2.5 )
                data_amp2 = 10.0**( (p_vector[fit][0]*fp_r + p_vector[fit][1]*fp_r2 + p_vector[fit][2]*fp_r2*fp_r + p_vector[fit][3]*fp_r2*fp_r2)/2.5 ) # + term9_a2)/2.5 )
            elif( CR1_A62 ):
                data_amp1 = 10.0**( (p_vector[fit][0]*fp_r)/2.5 ) # + term9_a1)/2.5 )
                data_amp2 = 10.0**( (p_vector[fit][0]*fp_r)/2.5 ) # + term9_a2)/2.5 )
            elif( C0_A62 ):
                data_amp1 = ccd_image.data + 10.0**( term9_a1/2.5 )
                data_amp1 = ccd_image.data + 10.0**( term9_a2/2.5 )
            elif( CCDXY_A62 ):
                data_amp1 = 10.0**( (p_vector[fit][2*ccd_num]*pix_xx + p_vector[fit][2*ccd_num+1]*pix_yy)/2.5 ) # + term9_a1)/2.5 )
                data_amp2 = 10.0**( (p_vector[fit][2*ccd_num]*pix_xx + p_vector[fit][2*ccd_num+1]*pix_yy)/2.5 ) # + term9_a2)/2.5 )
            elif( CCDXY9_A62 ):
                data_amp1 = 10.0**( (p_vector[fit][0]*xx + p_vector[fit][1]*yy + p_vector[fit][2]*xxyy + p_vector[fit][3]*xx2 + p_vector[fit][4]*yy2 + p_vector[fit][5]*xx2yy + p_vector[fit][6]*xxyy2 + p_vector[fit][7]*xx3 + p_vector[fit][8]*yy3 + p_vector[fit][9+2*ccd_num]*pix_xx + p_vector[fit][9+2*ccd_num+1]*pix_yy)/2.5 ) # + term9_a1)/2.5 )
                data_amp2 = 10.0**( (p_vector[fit][0]*xx + p_vector[fit][1]*yy + p_vector[fit][2]*xxyy + p_vector[fit][3]*xx2 + p_vector[fit][4]*yy2 + p_vector[fit][5]*xx2yy + p_vector[fit][6]*xxyy2 + p_vector[fit][7]*xx3 + p_vector[fit][8]*yy3 + p_vector[fit][9+2*ccd_num]*pix_xx + p_vector[fit][9+2*ccd_num+1]*pix_yy)/2.5 ) # + term9_a2)/2.5 )
            elif( SUPERPIX ):
                for x in range(4):
                    for y in range(8):
                        spix = 4*y + x + 32*ccd_num
                        zp = 10.0**( p_vector[fit][1+spix]/2.5 )
                        superpix = zp * numpy.ones((512,512))
                        data_amp1[512*y:512*(y+1),512*x:512*(x+1)] = superpix
                        data_amp2[512*y:512*(y+1),512*x:512*(x+1)] = superpix
            ccd_image.data[:,0:(width_int/2)-1] = data_amp1[:,0:(width_int/2)-1]
            ccd_image.data[:,(width_int/2):width_int] = data_amp2[:,(width_int/2):width_int]
            
            # for pix_x in range(width_int):
            #     x = pix_x + fp_xs[ccd_num]
            #     x2 = x*x
            #     term0 = p_vector[fit][0]*x
            #     term2 = p_vector[fit][2]*x
            #     term3 = p_vector[fit][3]*x2
            #     term5 = p_vector[fit][5]*x2
            #     term6 = p_vector[fit][6]*x
            #     term7 = p_vector[fit][7]*x2*x
            #     for pix_y in range(height_int):
            #         # the position dependence is in terms of the focal plane (x,y)
            #         y = pix_y + fp_ys[ccd_num]
            #         y2 = y*y
            #         exponent = 0.0
            #         if( C9_A62 ):
            #             exponent = (term0 + p_vector[fit][1]*y + term2*y + term3 + p_vector[fit][4]*y2 + term5*y + term6*y2 + term7 + p_vector[fit][8]*y2*y + term9)/2.5
            #         elif( C0_A62 ):
            #             exponent = term9/2.5
            #         elif( CCDXY_A62 ):
            #             exponent = (p_vector[fit][2*ccd_num]*pix_x + p_vector[fit][2*ccd_num+1]*pix_y + term9)/2.5
            #         elif( CCDXY9_A62 ):
            #             exponent = (term0 + p_vector[fit][1]*y + term2*y + term3 + p_vector[fit][4]*y2 + term5*y + term6*y2 + term7 + p_vector[fit][8]*y2*y + term9 + p_vector[fit][9 + 2*ccd_num]*pix_x + p_vector[fit][9 + 2*ccd_num+1]*pix_y)/2.5
            #         # print "%f %f %d %f %e" %(x, y, pix_y, fp_ys[ccd_num], exponent)
            #         ccd_image.data[pix_y,pix_x] = 10.0**(exponent)

            # Save image to tmp file
            # print "Finished writing ccd %d" %ccd_num
            pyfits.append(starflatname, data=ccd_image.data, header=ccd_image.header)
            print "Finished appending ccd %d" %ccd_num
    
        print "Finished writing %s" %starflatname
        
    if resid_map:

        from scipy import interpolate

        mapname = "resid.fits"
        ccd_arrays = []
        ccd_counts = []

        print "reading from directory %s for WCS info" %input_path
        dir_list = os.listdir(input_path)
        for file in dir_list:
            if re.match("DECam", file) is not None and re.search("fits$", file) is not None:
                break

        hdulist = pyfits.open(os.path.join(input_path,file))
        for i in range( 1,len(hdulist), 2 ): # 1,5,2 ):
            ccd_num = (i-1)/2 # zero ordered
            header_array = ''
            for char in hdulist[i].data[0]:
                header_array += char
            header_string = header_array.tostring()
            # print header_string
            # print type(header_string)
            matches = re.search( "NAXIS1\s+=\s+(\S+)", header_string )
            width = matches.group(1)
            matches = re.search( "NAXIS2\s+=\s+(\S+)", header_string )
            height = matches.group(1)
            width_int = int(width)
            height_int = int(height)
            data = numpy.zeros([height_int, width_int])
            counts = numpy.zeros([height_int, width_int])
            ccd_arrays.append(data)
            ccd_counts.append(counts)
        print "initialized %d ccds worth of data" %len(ccd_arrays)

        # now add up the stats
        for gindex, global_object in enumerate(global_objects):
            mean_mag = 0.
            for det in global_object.detections[f]:
                zp = 0.0
                fp_r2 = det.fp_x*det.fp_x + det.fp_y*det.fp_y
                fp_r = numpy.sqrt(fp_r2)
                term9 = p_vector[fit][n_par[fit]-n_amps+det.amp_num]
                if( C9_A62 ):
                    zp = p_vector[fit][0]*det.fp_x + p_vector[fit][1]*det.fp_y + p_vector[fit][2]*det.fp_x*det.fp_y + p_vector[fit][3]*det.fp_x*det.fp_x + p_vector[fit][4]*det.fp_y*det.fp_y + p_vector[fit][5]*det.fp_x*det.fp_x*det.fp_y + p_vector[fit][6]*det.fp_x*det.fp_y*det.fp_y + p_vector[fit][7]*det.fp_x*det.fp_x*det.fp_x + p_vector[fit][8]*det.fp_y*det.fp_y*det.fp_y + term9
                elif( CR4_A62 ):
                    zp = p_vector[fit][0]*fp_r + p_vector[fit][1]*fp_r2 + p_vector[fit][2]*fp_r2*fp_r + p_vector[fit][3]*fp_r2*fp_r2 + term9
                elif( CR1_A62 ):
                    zp = p_vector[fit][0]*fp_r + term9
                elif( C0_A62 ):
                    zp = term9
                elif( CCDXY_A62 ):
                    zp = p_vector[fit][2*det.ccd_num]*det.x + p_vector[fit][2*det.ccd_num+1]*det.y + term9
                elif( CCDXY9_A62 ):
                    zp = p_vector[fit][0]*det.fp_x + p_vector[fit][1]*det.fp_y + p_vector[fit][2]*det.fp_x*det.fp_y + p_vector[fit][3]*det.fp_x*det.fp_x + p_vector[fit][4]*det.fp_y*det.fp_y + p_vector[fit][5]*det.fp_x*det.fp_x*det.fp_y + p_vector[fit][6]*det.fp_x*det.fp_y*det.fp_y + p_vector[fit][7]*det.fp_x*det.fp_x*det.fp_x + p_vector[fit][8]*det.fp_y*det.fp_y*det.fp_y + p_vector[fit][9+2*det.ccd_num]*det.x + p_vector[fit][9+2*det.ccd_num+1]*det.y + term9
                elif( SUPERPIX ):
                    zp = p_vector[fit][1+det.spix]

                det.zp = zp
                mag = det.mag + det.zp
                mean_mag += mag
            mean_mag /= len(global_object.detections[f])
            for det in global_object.detections[f]:
                # print "adding %d %d %d %f" %(int(det.amp_num), int(det.y),int(det.x), det.mag+det.zp-mean_mag)
                ccd_arrays[int(det.ccd_num)][int(det.y),int(det.x)] += (det.mag+det.zp-mean_mag)
                ccd_counts[int(det.ccd_num)][int(det.y),int(det.x)] += 1.0

        print "done compiling info..."

        # make the fits file!
        new_hdu = pyfits.PrimaryHDU()
        new_hdu.header.update("NEXTEND", n_ccds)
        pyfits.writeto(mapname, data = numpy.array([]), header = new_hdu.header)

        print "interpolating..."
        for ccd_num in range(len(ccd_arrays)):
            print "starting %d" %ccd_num
            xs = []
            ys = []
            zs = []
            for y in range(ccd_arrays[ccd_num].shape[0]):
                for x in range(ccd_arrays[ccd_num].shape[1]):
                    if ccd_counts[ccd_num][y,x] > 0.0:
                        xs.append(x)
                        ys.append(y)
                        zs.append(ccd_arrays[ccd_num][y,x]/ccd_counts[ccd_num][y,x])
                    #if ccd_counts[ccd_num][y,x] == 0.0:
                    #    ccd_counts[ccd_num][y,x] = 1.0
            #ccd_arrays[ccd_num] = ccd_arrays[ccd_num]/ccd_counts[ccd_num]
            print "ccd %d read in data %d %d %d" %(ccd_num, len(xs), len(ys), len(zs) )
            # define grid.
            xi = numpy.arange(ccd_arrays[ccd_num].shape[1])
            yi = numpy.arange(ccd_arrays[ccd_num].shape[0])
            # grid the data.
            xarray = numpy.array(xs)
            yarray = numpy.array(ys)
            zarray = numpy.array(zs)
            zi = numpy.array(interpolate.griddata((xarray, yarray), zarray,(xi[None,:], yi[:,None]),method='nearest'))
            ccd_arrays[ccd_num] = zi
            # f = interpolate.interp2d(xs, ys, zs, kind='linear')
            # pix_x_array = numpy.arange(width_int)
            # pix_y_array = numpy.arange(height_int)
            # pix_xx,pix_yy = numpy.meshgrid(pix_x_array,pix_y_array)
            # ccd_arrays[ccd_num] = f(pix_xx.flatten(),pix_yy.flatten()).reshape(height_int,width_int)
            # for y in range(ccd_arrays[ccd_num].shape[0]):
            #     for x in range(ccd_arrays[ccd_num].shape[1]):
            #         ccd_arrays[ccd_num][y,x] = f(x,y)

        print "Making the residual map."

        for i in range( 1,len(hdulist), 2 ): # 1,5,2 ):
            ccd_num = (i-1)/2 # zero ordered
            header_array = ''
            for char in hdulist[i].data[0]:
                header_array += char
            header_string = header_array.tostring()
            # print header_string
            # print type(header_string)
            matches = re.search( "NAXIS1\s+=\s+(\S+)", header_string )
            width = matches.group(1)
            matches = re.search( "NAXIS2\s+=\s+(\S+)", header_string )
            height = matches.group(1)
            matches = re.search( "CRVAL1\s+=\s+(\S+)", header_string )
            crval1 = matches.group(1)
            matches = re.search( "CRVAL2\s+=\s+(\S+)", header_string )
            crval2 = matches.group(1)
            matches = re.search( "CRPIX1\s+=\s+(\S+)", header_string )
            crpix1 = matches.group(1)
            matches = re.search( "CRPIX2\s+=\s+(\S+)", header_string )
            crpix2 = matches.group(1)
            matches = re.search( "CD1_1\s+=\s+(\S+)", header_string )
            cd1_1 = matches.group(1)
            matches = re.search( "CD1_2\s+=\s+(\S+)", header_string )
            cd1_2 = matches.group(1)
            matches = re.search( "CD2_1\s+=\s+(\S+)", header_string )
            cd2_1 = matches.group(1)
            cd2_2 = (re.search( "CD2_2\s+=\s+(\S+)", header_string )).group(1)
            # print "wcs: %s %s %s %s %s %s %s %s %s %s" %(width, height, crval1, crval2, crpix1, crpix2, cd1_1, cd1_2, cd2_1, cd2_2)

            ccd_image = pyfits.ImageHDU()
            ccd_image.header = pyfits.Header(txtfile = "/Users/bauer/software/PAUdm/trunk/src/pipeline/pixelsim/config/header_extension.template")
            # now update the CCD header with this information
            ccd_image.header.update('NAXIS1', width)
            ccd_image.header.update('NAXIS2', height)
            ccd_image.header.update('WCSDIM', 2)
            # ccd_image.header.update('CTYPE1', (re.search( "CTYPE1\s+=\s+(\S+)", header_string )).group(1))
            # ccd_image.header.update('CTYPE2', (re.search( "CTYPE2\s+=\s+(\S+)", header_string )).group(1))
            ccd_image.header.update('CRVAL1', crval1)
            ccd_image.header.update('CRVAL2', crval2)
            ccd_image.header.update('CRPIX1', crpix1)
            ccd_image.header.update('CRPIX2', crpix2)
            ccd_image.header.update('CD1_1', cd1_1)
            ccd_image.header.update('CD1_2', cd1_2)
            ccd_image.header.update('CD2_1', cd2_1)
            ccd_image.header.update('CD2_2', cd2_2)

            ccd_image.data = ccd_arrays[int(ccd_num)]

            pyfits.append(mapname, data=ccd_image.data, header=ccd_image.header)
            print "Finished appending ccd %d" %ccd_num

        print "Finished writing %s" %mapname

        

# def finish_starflat():
#     
#     infile = open("starflat_tempa", 'rb')
#     a_matrix = lilmatrix
#     if( r = infile.read(4) ):
#         row = float(r)
#         col = float(infile.read(4));
#         val = float(infile.read(32));
        
    
    
"""
def test_star_flat():
    
        
        # load up the star flat fits file
        starflatname = "starflat_" + filter_tray + ".fits"
        starflat = []
        for n_ccd in range(1,19):
            starflat.append( pyfits.getdata(os.path.join(config['common']['general']['TMP_PATH'], starflatname), n_ccd) )
            #print starflat[n_ccd-1].shape
        log.debug( "Read in the star flat from %s" %os.path.join(config['common']['general']['TMP_PATH'], starflatname) )
    
        # load up the global objects
        global_objects = model.session.query(model.Global_Object).filter(model.Global_Object.production_id == db_production.id)
    
        # calculate the variances of the global objects
        for index, global_object in enumerate(global_objects):
        
            # make arrays of the detection's measurements in each filter
            measurements = dict()
            mags1 = dict()
            mags2 = dict()
            for detection in global_object.detections:
                mosaic = model.session.query(model.Mosaic).join(model.Image).filter(model.Image.id == detection.image_id).one()
                if mosaic.filtertray != filter_tray:
                    continue
                if not mosaic.id in phot_params:
                    phot_params[mosaic.id] = photometry.get_phot_params(mosaic)
                image = model.session.query(model.Image).filter(model.Image.id == detection.image_id).one()
                filt = image.filter
                if not filt in measurements:
                    measurements[filt] = []
                    mags1[filt] = []
                    mags2[filt] = []
                (mag, mag_error) = photometry.pau_flux_to_mag(detection.flux_auto, detection.flux_err_auto, phot_params[mosaic.id], image.ccd_num)
                if mag_error > 0.05:
                    continue
                (x, y) = ccd_to_focal_plane_xy(detection.x, detection.y, image.ccd_num)
                x = round(x)
                y = round(y)
                measurements[filt].append( (mag, mag_error, detection.x, detection.y, image.ccd_num) )
                mags1[filt].append(mag)
                mags2[filt].append(mag - 2.5*math.log10(starflat[image.ccd_num-1][detection.y,detection.x]))
                #log.info( "%d %f (%f), %f" %(index, mags1[filt][-1], mag_error, mags2[filt][-1]) )
            
            for filt in mags1.keys():
                if len(mags1[filt]) > 1:
                    print "%f %f" %(numpy.std(mags1[filt]), numpy.std(mags2[filt])) 
"""


if init_cat:
    match_objs(filt)

go_lists = dict()
n_imgs = dict()
for filt in filters:
    go_lists[filt] = global_objs_list()
    n_imgs[filt] = read_list( filt, go_lists[filt] )
    clip_objs( go_lists[filt], filt )
    print "%d global objects read for filter %s" %(len(go_lists[filt].global_list), filt)
if stats_pre:
    for filt in filters:
        calc_stats( go_lists[filt], filt )
if stats_post or make_fits or resid_map or obj_zps:
    go_list = combine_filts( go_lists )
    #print_mags( go_list )
    if use_colorterms:
        apply_colorterms( go_list )

    # remove other filters' info, to free memory.
    for f in filters:
        if f == filt:
            continue
        else:
            for g, go in enumerate(go_list.global_list):
                go_list.global_list[g].detections[f] = []

    calc_star_flat( go_lists[filt].global_list, n_imgs[filt], filt )



#finish_starflat()

