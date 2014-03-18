import sys
import os
import math
import cPickle
from operator import itemgetter
import numpy as np
from scipy.sparse import vstack as sparsevstack
from scipy.sparse import linalg
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from rtree import index
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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


class global_object:
    def __init__(self):
        self.ra = None
        self.dec = None
        self.objects = []

# for now.....
def good_quality(star):
    if star['magerr_psf'] < 0.05:
        return True
    return False

def unmatched(star):
    if star['matched']:
        return False
    return True


def make_gos(filename, nside, band):
    
    
    mag_type = 'psf'
    flux_name = "flux_" + "%s" %mag_type
    flux_err_name = "flux_err_" + "%s" %mag_type
    
    max_nims = 1e6 # ick!
    max_nepochs = 20 # ick!
    max_objs_per_image = 500 # so that in super-dense areas like the LMC we don't spend ages, but instead use a subset of objects.
    
    npix = healpy.nside2npix(nside)
    pixelMap = np.arange(npix)
    
    # hdf5 
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
        star2['matched'] = 0
        star2['count'] = 0
        star2['gskyphot'] = star['gskyphot']
        star2['lskyphot'] = star['lskyphot']
        star2['gskyhot'] = star['gskyhot']
        star2['lskyhot'] = star['lskyhot']
        star2['superpix'] = int(4*np.floor(star2['y_image']/512.) + np.floor(star2['x_image']/512.) + 32*star2['ccd'])
        
        phi = star['ra']*3.1415926/180.
        theta = (90.-star['dec'])*3.1415926/180.
        pix = healpy.pixelfunc.ang2pix(nside, theta, phi)
        if not pix in stars_by_img:
            stars_by_img[pix] = dict()
            stars_by_exp[pix] = dict()
        
        if not star['imageid'] in stars_by_img[pix]:
            stars_by_img[pix][star['imageid']] = []
        stars_by_img[pix][star['imageid']].append(star2)
    
    
    print "%d pixels" %len(stars_by_img.keys())
    for pix in stars_by_img.keys():
        count = 0
        for img in stars_by_img[pix].keys():
            star_list = sorted(stars_by_img[pix][img], key=itemgetter('magerr_psf'))
            star_list = star_list[0:max_objs_per_image]
            if not star_list[0]['exposureid'] in stars_by_exp[pix]:
                stars_by_exp[pix][star_list[0]['exposureid']] = []
            stars_by_exp[pix][star_list[0]['exposureid']].extend(star_list)
            count+=len(star_list)
        print "Pixel %d, %d objects" %(pix, count)
        
        
        match_radius = 1.0/3600.
        gos = []
        exposures = stars_by_exp[pix].keys()
        ne = len(exposures)
        for e1 in range(ne):
            exposure1 = exposures[e1]
            global_list = dict()
            star1_ras = [o['ra'] for o in stars_by_exp[pix][exposure1] if not o['matched']]
            star1_decs = [o['dec'] for o in stars_by_exp[pix][exposure1] if not o['matched']]
            star1_indices = [o for o in range(len(stars_by_exp[pix][exposure1])) if not stars_by_exp[pix][exposure1][o]['matched']]
            for e2 in range(e1+1,ne):
                exposure2 = exposures[e2]
                
                star2_ras = [o['ra'] for o in stars_by_exp[pix][exposure2] if not o['matched']]
                star2_decs = [o['dec'] for o in stars_by_exp[pix][exposure2] if not o['matched']]
                star2_indices = [o for o in range(len(stars_by_exp[pix][exposure2])) if not stars_by_exp[pix][exposure2][o]['matched']]
                
                if len(star1_ras) == 0 or len(star2_ras) == 0:
                    continue
                
                inds1, inds2, dists = spherematch( star1_ras, star1_decs, star2_ras, star2_decs, tol=match_radius )
                
                if len(inds1) < 2:
                    continue
                
                for i in range(len(inds1)):
                    try:
                        global_list[star1_indices[inds1[i]]].append(stars_by_exp[pix][exposure2][star2_indices[inds2[i]]])
                        stars_by_exp[pix][exposure2][star2_indices[inds2[i]]]['matched'] = True
                    except:
                        global_list[star1_indices[inds1[i]]] = []
                        global_list[star1_indices[inds1[i]]].append(stars_by_exp[pix][exposure1][star1_indices[inds1[i]]])
                        global_list[star1_indices[inds1[i]]].append(stars_by_exp[pix][exposure2][star2_indices[inds2[i]]])
                        stars_by_exp[pix][exposure1][star1_indices[inds1[i]]]['matched'] = True
                        stars_by_exp[pix][exposure2][star2_indices[inds2[i]]]['matched'] = True
            
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
                
        outfile = open( 'gobjs_' + band + '_nside' + str(nside) + '_p' + str(pix), 'wb' )
        cPickle.dump(gos, outfile, cPickle.HIGHEST_PROTOCOL)
        outfile.close()
        print "Pixel %d wrote %d global objects" %(pix, len(gos))

        h5file.close()


def main():
    filt = 'i'
    nside = 32 # 2: 30 degrees per side, 4: 15 degrees per side, 8:  7.3 degrees per side
    
    print "Making global objects from DES, filter %s!" %filt
    if len(sys.argv) < 2:
        print "Usage: make_global_objs.py filename"
        print "where filename is like finalcutout.h5"
        exit(1)
    
    filename = sys.argv[1]
    print "Using file %s" %filename
    
    make_gos( filename, nside, filt )
    


if __name__ == '__main__':
    main()


