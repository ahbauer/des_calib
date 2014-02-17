import sys
import os
import math
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

# exposureid,imageid,object_id,x_image,y_image,ra,dec,mag_psf,magerr_psf,zeropoint,zeropointid,fwhm_arcsec,spread_model,flags,nite,run,propid,object,band,ccd,airmass,mjd_obs,exptime,photflag,skybrite,skysigma,image_ellipt,image_fwhm_arcsec,image_sat_level,imagetype

# for now.....
def good_quality(star):
    if star['magerr_psf'] < 0.05:
        return True
    return False

def unmatched(star):
    if star['matched']:
        return False
    return True


def ubercalibrate_by_filter(filename, band, makeplots=False, makestats=True):

    
    mag_type = 'psf'
    flux_name = "flux_" + "%s" %mag_type
    flux_err_name = "flux_err_" + "%s" %mag_type

    max_nims = 1e6 # ick!
    max_nepochs = 20 # ick!
    
    use_hdf5 = True
    
    n_ccds = 63
    n_xspx = 8
    n_yspx = 16
    spxsize = 256
    p_vector = None
    plot_mags = []
    plot_images = []
    plot_ras = []
    plot_decs = []
    plot_resids = []
    plot_resid_ns = []
    for ccd in range(n_ccds):
        plot_resids.append(np.zeros((n_yspx,n_xspx)))
        plot_resid_ns.append(np.zeros((n_yspx,n_xspx)))
    plot_count = 0

    print "Starting to read file"

    stars = []
    star_maps = []
    n_ra_divs = 360
    n_dec_divs = 90
    for rabin in range(n_ra_divs):
        star_maps.append([])
        for decbin in range(n_dec_divs):
            star_maps[rabin].append(index.Index()) #map of detections

    # hdf5 
    if use_hdf5:
        h5file = tables.openFile(filename, "r")
        table = None
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
        
        count = 0
        imgids = dict()
        for star in table.iterrows():
        
            if( good_quality(star) ):
                star2 = dict()
                star2['ra'] = star['ra']
                star2['dec'] = star['dec']
                star2['band'] = star['band']
                star2['mag_psf'] = star['mag_psf']
                star2['magerr_psf'] = star['magerr_psf']
                star2['x_image'] = star['x_image']
                star2['y_image'] = star['y_image']
                star2['image_id'] = star['imageid']
                star2['ccd'] = star['ccd']
                star2['matched'] = False # is it part of a global object yet?
                imgids[star2['image_id']] = True
                star2['count'] = count
                stars.append(star2)
                rabin = (int)(star['ra']/(360/n_ra_divs))
                decbin = (int)(star['dec']/(360/n_dec_divs))
                star_maps[rabin][decbin].insert( count, (star2['ra'],star2['dec'],star2['ra'],star2['dec']) )
                count+=1
        
        h5file.close()
        
    # ascii
    else:
        file = open(filename, 'r')
        header = headerline()
        count = 0
        imgids = dict()
        for line in file:
            # if l == 0:
            #     continue
            # entries = line.split(",")
            # star = dict()
            # star['ra'] = float(entries[5])
            # star['dec'] = float(entries[6])
            # star['band'] = entries[23]
            # star['mag_psf'] = float(entries[7])
            # star['magerr_psf'] = float(entries[8])
            # star['x_image'] = float(entries[3])
            # star['y_image'] = float(entries[4])
            # star['image_id'] = int(entries[1])
    
            entries = line.split(",")
            star = dict()
            try:
                star['ra'] = float(entries[header.ra])
                star['dec'] = float(entries[header.dec])
                star['band'] = entries[header.band]
                star['mag_psf'] = float(entries[header.mag_psf])
                star['magerr_psf'] = float(entries[header.magerr_psf])
                star['x_image'] = float(entries[header.x_image])
                star['y_image'] = float(entries[header.y_image])
                star['image_id'] = int(entries[header.imageid])
                star['ccd'] = int(entries[header.ccd])
            except TypeError:
                # This is the header line?
                index0 = 0
                for e, entry in enumerate(entries):
                    if( entries[0] == '#' ):
                        index0 = 1
                        continue
                    setattr(header, entry, e-index0)
                if entries[header.ra] is None:
                    print >> sys.stderr, "Error, catalog doesn't have ra"
                    exit(1)
                if entries[header.dec] is None:
                    print >> sys.stderr, "Error, catalog doesn't have dec"
                    exit(1)
                if entries[header.band] is None:
                    print >> sys.stderr, "Error, catalog doesn't have band"
                    exit(1)
                if entries[header.mag_psf] is None:
                    print >> sys.stderr, "Error, catalog doesn't have mag_psf"
                    exit(1)
                if entries[header.magerr_psf] is None:
                    print >> sys.stderr, "Error, catalog doesn't have magerr_psf"
                    exit(1)
                if entries[header.x_image] is None:
                    print >> sys.stderr, "Error, catalog doesn't have x_image"
                    exit(1)
                if entries[header.y_image] is None:
                    print >> sys.stderr, "Error, catalog doesn't have y_image"
                    exit(1)
                if entries[header.imageid] is None:
                    print >> sys.stderr, "Error, catalog doesn't have imageid"
                    exit(1)
                continue
            # if( star['band'] == 'i' and star['magerr_psf'] > 0.05 ):
            #     print "star: %f %f %s %f %f" %(star['ra'], star['dec'], star['band'], star['mag_psf'], star['magerr_psf'])
    
            if( star['band'] == band and good_quality(star) ):
                star['matched'] = False # is it part of a global object yet?
                imgids[star['image_id']] = True
                star['count'] = count
                stars.append(star)
                rabin = (int)(star['ra']/(360/n_ra_divs))
                decbin = (int)(star['dec']/(360/n_dec_divs))
                star_maps[rabin][decbin].insert( count, (star['ra'],star['dec'],star['ra'],star['dec']) )
                count+=1
    
        file.close()
    
    # end read in
    
    max_nimgs = len(imgids.keys())
    print "Read in %d = %d good quality stars in band %s from %d images" %(count, len(stars), band, max_nimgs)

    # if count < 100:
    #     print "Very few stars... quitting!"
    #     exit(1)

    # now we need to match the detections to get "global objects"
    # this matching is not careful in any way......

    # while we're at it, add stuff to the matrices so we don't have to loop over global objects.
    if p_vector is None:
        p_vector = np.zeros(max_nimgs)
    match_radius = 1.0/3600.
    image_id_count = -1
    image_id_dict = dict()
    image_ras = dict()
    image_decs = dict()
    image_ns = dict()
    ndets = 0
    sum_for_zps = np.zeros(max_nimgs)
    b_vector = []
    a_matrix = None
    c_vector = []
    count = 0
    sum_rms = 0.
    n_rms = 0
    matrix_size = 0
    max_matrix_size = (len(stars)/2.0)*max_nepochs
    a_matrix_xs = np.zeros(max_matrix_size)
    a_matrix_ys = np.zeros(max_matrix_size)
    a_matrix_vals = np.zeros(max_matrix_size)
    interval = int(len(stars)/10.)
    global_objects = []
    for s, star in enumerate(stars):
        if s%interval == 0:
            print "%d out of %d..." %(s, len(stars))
        if star['matched']:
            continue
        ra_match_radius = match_radius/math.cos(star['dec']*math.pi/180.)
        dec_match_radius = match_radius
        match_area = (star['ra']-ra_match_radius, star['dec']-dec_match_radius,
                        star['ra']+ra_match_radius, star['dec']+dec_match_radius)
        rabin = (int)(star['ra']/(360/n_ra_divs))
        decbin = (int)(star['dec']/(360/n_dec_divs))
        det_indices = list(star_maps[rabin][decbin].intersection(match_area))
        if len(det_indices) < 2:
            continue
        det_candidates = []
        for di in det_indices:
            det_candidates.append(stars[di])
    
        unmatched_list = filter(unmatched,det_candidates)
        if len(unmatched_list) < 2:
            continue
    
        # ok, now we have a "global object"
        count += 1
        ndet = len(unmatched_list)
        mags = np.zeros(ndet)
        mag_errors = np.zeros(ndet)
        image_ids = np.zeros(ndet, dtype=np.int)
        image_id_matrix = np.zeros([max_nimgs,ndet], dtype=np.int)
        d=0
        # print "for object %f %f" %(star['ra'], star['dec'])
        go = []
        for star2 in unmatched_list:
            # add up matrix info
            try:
                image_ids[d] = image_id_dict[star2['image_id']]
                image_id_matrix[image_ids[d],d] = 1
                image_ras[star2['image_id']] += star2['ra']
                image_decs[star2['image_id']] += star2['dec']
                image_ns[star2['image_id']] += 1
            except KeyError:
                # the current image id is not in the image_ids dictionary yet
                image_id_count += 1
                image_id_dict[star2['image_id']] = image_id_count
                image_ids[d] = image_id_dict[star2['image_id']]
                image_id_matrix[image_ids[d],d] = 1
                image_ras[star2['image_id']] = star2['ra']
                image_decs[star2['image_id']] = star2['dec']
                image_ns[star2['image_id']] = 1
            # print "match %f %f" %(star2['ra'], star2['dec'])
            # print "adding to mag %f p_vector[%d] = %f" %(star2['mag_psf'], image_id_dict[star2['image_id']], p_vector[image_id_dict[star2['image_id']]])
            mags[d] = star2['mag_psf'] + p_vector[image_id_dict[star2['image_id']]]
            mag_errors[d] = np.sqrt(star2['magerr_psf']*star2['magerr_psf'] + 0.0001)
            stars[star2['count']]['matched'] = True  # mark this star as taken
            # star2 = dict() # "erase" the star's info
            go.append(star2['count'])
            d += 1
        if( ndet != d ):
            print "hm, ndet = %d, d = %d" %(ndet, d)
            exit(1)
        
        global_objects.append(go)
        
        invsigma_array = 1.0/np.square(mag_errors)
        invsigma_matrix = np.tile(invsigma_array, (ndet,1))
        sum_invsigma2 = invsigma_array.sum()
        sum_m_i = (mags*invsigma_array).sum() / sum_invsigma2
        
        invsigma_matrix = np.tile(invsigma_array, (max_nimgs,1))
        sum_for_zps = (image_id_matrix*invsigma_matrix).sum(axis=1) / sum_invsigma2
        b_vector = np.append( b_vector, mags - sum_m_i )
        
        a_submatrix = np.tile(sum_for_zps, (ndet,1) )
        a_submatrix[range(ndet),image_ids[range(ndet)]] -= 1.0
        a_submatrix = coo_matrix(a_submatrix)
        
        indices = np.where(a_submatrix.data != 0.)[0]
        a_matrix_xs[matrix_size:(matrix_size+len(indices))] = a_submatrix.col[indices]
        a_matrix_ys[matrix_size:(matrix_size+len(indices))] = a_submatrix.row[indices]+ndets
        a_matrix_vals[matrix_size:(matrix_size+len(indices))] = a_submatrix.data[indices]
        matrix_size += len(indices)
        
        c_vector = np.append(c_vector, invsigma_array)
        
        if makeplots:
            plot_mags.append([])
            plot_images.append([])
            plot_ras.append(star['ra'])
            plot_decs.append(star['dec'])
            for m in range(len(mags)):
                plot_mags[plot_count].append(mags[m])
                plot_images[plot_count].append(image_ids[m])
            plot_count += 1
        
        # add up some stats
        sum_rms += np.std(mags-sum_m_i)
        n_rms += 1
        ndets += ndet

    
    for iid in image_ras.keys():
        image_ras[iid] /= image_ns[iid]
        image_decs[iid] /= image_ns[iid]

    print "Looped through %d global objects, %d measurements total" %( count, ndets )
    print "Mean RMS of the stars: %e" %(sum_rms/n_rms)
    
    
    a_matrix = coo_matrix((a_matrix_vals[0:matrix_size], (a_matrix_ys[0:matrix_size], a_matrix_xs[0:matrix_size])), shape=(ndets,max_nimgs))

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
    outfilename = 'ubercal_zps_' + band
    outfile = open( outfilename, 'w' )
    for imageid in sorted(image_id_dict.iterkeys()):
        outfile.write( "%d %e\n" %(imageid, p_vector[image_id_dict[imageid]]) );
    outfile.close()
    
    
    # now calculate how we did
    sum_rms = 0.
    n_rms = 0
    for go in global_objects:
        mags = np.zeros(len(go))
        mag_errors = np.zeros(len(go))
        for i,o in enumerate(go):
            star = stars[o]
            mags[i] = star['mag_psf'] + p_vector[image_id_dict[star['image_id']]]
            mag_errors[i] = np.sqrt(star['magerr_psf']*star['magerr_psf'] + 0.0001)

        invsigma_array = 1.0/np.square(mag_errors)
        sum_invsigma2 = invsigma_array.sum()
        sum_m_i = (mags*invsigma_array).sum() / sum_invsigma2
        sum_rms += np.std(mags-sum_m_i)
        n_rms += 1
        if makeplots:
            for i,o in enumerate(go):
                superpix_x = int(star['x_image']/spxsize)
                superpix_y = int(star['y_image']/spxsize)
                plot_resids[star['ccd']][superpix_y,superpix_x] += (mags[i] - sum_m_i)
                plot_resid_ns[star['ccd']][superpix_y,superpix_x] += 1
    print "Mean RMS after calibration: %e" %(sum_rms/n_rms)
    
    
    if makeplots:
        for ccd in range(len(plot_resids)):
            for x in range(n_xspx):
                for y in range(n_yspx):
                    if( plot_resid_ns[ccd][y,x] > 0 ):
                        plot_resids[ccd][y,x] /= plot_resid_ns[ccd][y,x]
        
        print "making plots"
        plot_file = 'ubercal_plots_' + band + '.pdf'
        pp = PdfPages(plot_file)
        
        stats0 = []
        stats1 = []
        stats2 = []
        stats_imgs = dict()
        ras = []
        decs = []
        stats0_out = []
        stats1_out = []
        for i in range(len(plot_mags)):
            mean_before = np.mean(plot_mags[i])
            mags_after1 = []
            for m in range(len(plot_mags[i])):
                mags_after1.append(plot_mags[i][m]+p_vector[plot_images[i][m]])
            mean_after1 = np.mean(mags_after1)
            for m in range(len(mags_after1)):
                before = plot_mags[i][m] - mean_before
                after = mags_after1[m] - mean_after1
                stats0.append(before)
                stats1.append(after)
                try:
                    stats_imgs[plot_images[i][m]].append(after)
                except KeyError:
                    stats_imgs[plot_images[i][m]] = []
                    stats_imgs[plot_images[i][m]].append(after)
                if( abs(before) > 0.2 ):
                    ras.append(plot_ras[i])
                    decs.append(plot_decs[i])
                    stats0_out.append(before)
                    stats1_out.append(after)
        s0_low = scoreatpercentile(stats0, 16)
        s0_high = scoreatpercentile(stats0, 84)
        s0_low2 = scoreatpercentile(stats0, 2.25)
        s0_high2 = scoreatpercentile(stats0, 97.75)
        s1_low = scoreatpercentile(stats1, 16)
        s1_high = scoreatpercentile(stats1, 84)
        s1_low2 = scoreatpercentile(stats1, 2.25)
        s1_high2 = scoreatpercentile(stats1, 97.75)
        
        # histogram of the zps
        title = "Ubercal Zero Points, filter %s" %band
        plt.clf()
        plt.hist(p_vector, 50)
        xlabel = "Magnitudes"
        ylabel = ""
        plt.xlabel(xlabel)
        plt.title(title)
        pp.savefig()
        # histogram of the mag differences from the mean before
        title = "Before: mag diff from mean, 68 (95) pc width %f (%f)" %(s0_high-s0_low, s0_high2-s0_low2)
        plt.clf()
        plt.yscale('log')
        plt.hist(stats0, 50)
        xlabel = "Magnitudes"
        ylabel = ""
        plt.xlabel(xlabel)
        plt.title(title)
        pp.savefig()
        # histogram of the mag differences from the mean after
        title = "After: mag diff from mean, 68 (95) pc width %f (%f)" %(s1_high-s1_low, s1_high2-s1_low2)
        plt.clf()
        plt.yscale('log')
        plt.hist(stats1, 50)
        xlabel = "Magnitudes"
        ylabel = ""
        plt.xlabel(xlabel)
        plt.title(title)
        pp.savefig()
        
        # 3D plots of image zp by position
        ras = []
        decs = []
        zps = []
        for img in image_ras.keys():
            ras.append(image_ras[img])
            decs.append(image_decs[img])
            zps.append(p_vector[image_id_dict[img]])
        title = "Image zero point by position"
        plt.clf()
        plt.yscale('linear')
        fig = plt.figure()
        ax3D = fig.add_subplot(111, projection='3d')
        ax3D.view_init(90, 0)
        ax3D.scatter(ras, decs, zps, c=zps)  
        xlabel = "ra"
        ylabel = "dec"
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        pp.savefig()
        
        # 3D plots of ccd residuals
        spx = []
        spy = []
        resids = []
        for x in range(n_xspx):
            for y in range(n_yspx):
                spx.append(x)
                spy.append(y)
                resids.append(plot_resids[ccd][y,x])
        for ccd in range(n_ccds):
            if ccd%8 == 0:
                if ccd>0:
                    pp.savefig()
                plt.clf()
                fig = plt.figure()
                loc = 1
            ax1 = fig.add_subplot(2,4,loc)
            # ax3D = fig.add_subplot(111, projection='3d')
            # ax3D.view_init(90, 0)
            # ax3D.scatter(spx, spy, resids, c=resids)
            ax1.imshow(plot_resids[ccd],interpolation='nearest')
            # xlabel = "x superpix"
            # ylabel = "y superpix"
            title = "ccd %d" %ccd
            # plt.xlabel(xlabel)
            # plt.ylabel(ylabel)
            plt.title(title)
            loc += 1
        pp.savefig()
        
        # # mag diffs vs image id
        # for img in stats_imgs.keys():
        #     title = "image %d after zps" %img
        #     plt.clf()
        #     plt.hist(stats_imgs[img], 50)
        #     xlabel = "Mag diff"
        #     ylabel = ""
        #     plt.xlabel(xlabel)
        #     plt.title(title)
        #     pp.savefig()
        
        # # 3D plots of mag diffs by position
        # title = "Before: mag diff from mean, by position"
        # plt.clf()
        # fig = plt.figure()
        # ax3D = fig.add_subplot(111, projection='3d')
        # ax3D.view_init(90, 0)
        # ax3D.scatter(ras, decs, stats0_out, c=stats0_out)  
        # xlabel = "ra"
        # ylabel = "dec"
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)
        # plt.title(title)
        # pp.savefig()
        # title = "After: mag diff from mean, by position"
        # plt.clf()
        # fig = plt.figure()
        # ax3D = fig.add_subplot(111, projection='3d')
        # ax3D.view_init(90, 0)
        # ax3D.scatter(ras, decs, stats1_out, c=stats0_out)  
        # xlabel = "ra"
        # ylabel = "dec"
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)
        # plt.title(title)
        # pp.savefig()
        
        pp.close()
    

def main():
    filt = 'i'
    print "Running ubercalibration for DES, filter %s!" %filt
    
    # filename = "/Users/bauer/surveys/DES/sva1/finalcut/finalcutout"
    filename = "/Users/bauer/surveys/DES/sva1/finalcut/finalcutout_subsetimg.h5"
    # filename = "/Users/bauer/surveys/DES/sva1/finalcut/test"
    ubercalibrate_by_filter( filename, filt, makeplots=True )



if __name__ == '__main__':
    main()


