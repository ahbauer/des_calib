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
import gzip
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
    if star['magerr'] < 0.1:
        return True
    return False

def unmatched(star):
    if star['matched']:
        return False
    return True


def make_gos(nside):
    
    
    mag_index = 2
    
    max_nims = 1e6 # ick!
    max_nepochs = 200 # ick!
    max_objs_per_image = 500 # so that in super-dense areas like the LMC we don't spend ages, but instead use a subset of objects.
    
    npix = healpy.nside2npix(nside)
    pixelMap = np.arange(npix)
    
    # loop over pixels to populate
    catalog_filenames = []
    for p in pixelMap:
        catalog_filenames.append([])
    
    # go through all the image directories
    lsq_data_dir = '/Volumes/LSQ1'
    
    # loop over year directories
    year_dirs = os.listdir(lsq_data_dir)
    for year in year_dirs:
        if year in map(str,range(2000,2020)):
            year_dir = os.path.join(lsq_data_dir,year)
            date_dirs = os.listdir(year_dir)
            for date in date_dirs:
                if len(date) == 8 and date[0] == '2':
                    date_dir = os.path.join(year_dir,date)
                    image_dirs = os.listdir(date_dir)
                    for image in image_dirs:
                        if image[0:6] == 'images' and image[12] == 's':
                            data_dir = os.path.join(date_dir,image)
                            header_filename = os.path.join(data_dir,'header.txt')
                            if not os.path.isfile(header_filename):
                                print "Warning, no header.txt in {0}".format(data_dir)
                                continue
                            header = open(header_filename)
                            for line in header:
                                sline = line.split()
                                ra = float(sline[2])
                                dec = float(sline[3])
                                phi = ra*3.1415926/180.
                                theta = (90.-dec)*3.1415926/180.
                                ccd_pix = healpy.pixelfunc.ang2pix(nside, theta, phi)
                                filenames = os.listdir(data_dir)
                                for filename in filenames:
                                    if filename[0] == '2' and (filename[-4:] == '.cat' or filename[-7:] == '.cat.gz'):
                                        catalog_filenames[ccd_pix].append(os.path.join(data_dir,filename))
    
    n_cats = 0
    n_pix_wobjs = 0
    for cats in catalog_filenames:
        if len(cats) > 0:
            n_cats += len(cats)
            n_pix_wobjs += 1
    print "Found {0} catalogs total over {1} pixels".format(n_cats, n_pix_wobjs)
    
    for p,cats_by_pix in enumerate(catalog_filenames):
        if len(cats_by_pix) > 0:
            starlists_pix = []
            for c1, cat_filename1 in enumerate(cats_by_pix):
                catalog1 = None
                if cat_filename1[-4:] == '.cat':
                    catalog1 = open(cat_filename1)
                elif cat_filename1[-7:] == '.cat.gz':
                    catalog1 = gzip.open(cat_filename1, 'rb')
                starlist_img = []
                for line in catalog1:
                    sline = line.split()
                    star = dict()
                    star['ccd'] = int(sline[0])
                    (star['ra'], star['dec'], star['y_image'], star['x_image']) = map(float,sline[1:5])
                    star['mag'] = float(sline[5+mag_index])
                    star['magerr'] = float(sline[15+mag_index])
                    (star['background'], star['fwhm'], star['a'], star['b']) = map(float,sline[25:29])
                    star['matched'] = 0
                    star['count'] = 0
                    
                    if True in np.isnan(star.values()):
                        continue
                    
                    star['superpix'] = int(4*np.floor(star['y_image']/512.) + np.floor(star['x_image']/512.) + 32*star['ccd'])
                    
                    filename1 = os.path.basename(cat_filename1)
                    year = int(filename1[0:4])
                    month = int(filename1[4:6])
                    day = int(filename1[6:8])
                    hour = int(filename1[8:10])
                    minute = int(filename1[10:12])
                    second = int(filename1[12:14])
                    jd1 = day-32075+1461*(year+4800+(month-14)/12)/4+367*(month-2-(month-14)/12*12)  /12-3*((year+4900+(month-14)/12)/100)/4;
                    jd = jd1 + (hour/24.0) + minute/(24.0*60.0) + second/(24.0*3600.0) - 0.5;
                    star['mjd'] = jd - 2400000.5
                    
                    # print "{0} {1} {2} {3} {4} {5} {6} {7} {8}".format(star['ra'], star['dec'], star['ccd'], star['x_image'], star['y_image'], star['mag'], star['magerr'], star['superpix'], star['mjd'])
                    
                    if not good_quality(star):
                        continue
                    
                    # only keep stars in this pixel
                    phi = star['ra']*3.1415926/180.
                    theta = (90.-star['dec'])*3.1415926/180.
                    pix = healpy.pixelfunc.ang2pix(nside, theta, phi)
                    if pix != p:
                        continue
                    starlist_img.append(star)
                    
                starlists_pix.append(starlist_img)
                
            match_radius = 2.0/3600.
            gos = []
            n_imgs = len(starlists_pix)
            for i1 in range(n_imgs):
                global_list = dict()
                
                star1_ras = [o['ra'] for o in starlists_pix[i1] if not o['matched']]
                star1_decs = [o['dec'] for o in starlists_pix[i1] if not o['matched']]
                star1_indices = [o for o in range(len(starlists_pix[i1])) if not starlists_pix[i1][o]['matched']]
                
                for i2 in range(i1+1,n_imgs):
                    star2_ras = [o['ra'] for o in starlists_pix[i2] if not o['matched']]
                    star2_decs = [o['dec'] for o in starlists_pix[i2] if not o['matched']]
                    star2_indices = [o for o in range(len(starlists_pix[i2])) if not starlists_pix[i2][o]['matched']]
                    
                    if len(star1_ras) == 0 or len(star2_ras) == 0:
                        continue
                    
                    inds1, inds2, dists = spherematch( star1_ras, star1_decs, star2_ras, star2_decs, tol=match_radius )
                    
                    if len(inds1) < 2:
                        continue
                    
                    for i in range(len(inds1)):
                        try:
                            global_list[star1_indices[inds1[i]]].append(starlists_pix[i2][star2_indices[inds2[i]]])
                            starlists_pix[i2][star2_indices[inds2[i]]]['matched'] = True
                        except:
                            global_list[star1_indices[inds1[i]]] = []
                            global_list[star1_indices[inds1[i]]].append(starlists_pix[i1][star1_indices[inds1[i]]])
                            global_list[star1_indices[inds1[i]]].append(starlists_pix[i2][star2_indices[inds2[i]]])
                            starlists_pix[i1][star1_indices[inds1[i]]]['matched'] = True
                            starlists_pix[i2][star2_indices[inds2[i]]]['matched'] = True
                
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
            
            if len(gos) > 0:
                outfile = open( 'gobjs_nside' + str(nside) + '_p' + str(p), 'wb' )
                cPickle.dump(gos, outfile, cPickle.HIGHEST_PROTOCOL)
                outfile.close()
                print "Pixel %d wrote %d global objects" %(p, len(gos))


def main():
    nside = 32 # 2: 30 degrees per side, 4: 15 degrees per side, 8:  7.3 degrees per side
    
    print "Making global objects from LSQ!"
    if len(sys.argv) != 1:
        print "Usage: make_global_objs.py"
        exit(1)
    
    make_gos( nside )
    


if __name__ == '__main__':
    main()


