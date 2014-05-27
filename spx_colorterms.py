import sys
import os
import yaml
import matplotlib.pyplot as plt
from nebencal_utils import global_object

class multicolor_info(object):
    def __init__(self):
        self.ras = []
        self.decs = []
        self.pixels = []
        self.mags = []
        self.colors = []
        self.spxs = []
    def add(self, r, d, p, m, c, s):
        self.ras.append(r)
        self.decs.append(d)
        self.pixels.append(p)
        self.mags.append(m)
        self.colors.append(c)
        self.spxs.append(s)

def main():
    
    if len(sys.argv) != 2:
        print "Usage: spx_colorterms.py config_filename"
        print "       To print out a default config file, run: spx_colorterms.py default"
        exit(1)
    elif sys.argv[1] == 'default':
        # config = open( os.path.dirname(os.path.realpath(__file__)) + '/configs/config_sample', 'r' )
        # print "\n### DEFAULT EXAMPLE CONFIG FILE ###\n"
        # for line in config:
        #     sys.stdout.write(line)
        print "Sorry, default config not implemented yet."
        exit(0)
        
    magerr_sys2 = 0.0004
        
    # read in config
    config_filename = sys.argv[1]
    print "Calculating and applying color temrs using config file %s!" %config_filename    
    config_file = open(config_filename)
    config = yaml.load(config_file)
    
    # what bands do we want to read in?
    primary_band = config['primary_band']
    secondary_band = config['secondary_band']
    
    # read in previous calibration info that we want to apply
    
    # read in the nominal ccd offsets (zp_phots), which will be our starting point
    zp_list = { primary_band: [], secondary_band: [] }
    for band in (primary_band, secondary_band):
        zp_phots = dict()
        zp_phot_file = open( config['general']['zp_phot_filename'][band], 'r' )
        zp_phots['id_string'] = 'ccd'
        zp_phots['operand'] = 1
        for line in zp_phot_file:
            entries = line.split()
            if entries[0][0] == '#':
                continue
            zp_phots[int(entries[0])] = -1.0*float(entries[4])
        zp_list[band].append(zp_phots)
        print "Read in ccd offsets (zp_phots)"
    
        # NOTE: assumes that the id_string is an integer!
        # read in the rest of the zps
        for calibration in config['calibrations'][band]:
            for i, zp_filename in enumerate(calibration['outfilenames']):
                zp_dict = dict()
                zp_dict['operand'] = calibration['operands'][i]
                zp_dict['id_string'] = calibration['id_strings'][i]
                zp_file = open( zp_filename, 'r' )
                for line in zp_file:
                    entries = line.split()
                    if entries[0][0] == '#':
                        continue
                    zp_dict[int(entries[0])] = float(entries[1])
                zp_list[band].append(zp_dict)
                print "Read in {0} zeropoints from {1} for id_string {2}, operand {3}".format(len(zp_dict.keys())-2, zp_filename, calibration['id_strings'][i], calibration['operands'][i])
    
    
    # read in global objects for each pixel seen in both bands
    globals_dir = config['general']['globals_dir']
    nside = config['general']['nside_file']
    mc_info = multicolor_info()
    
    # go through pixels in the primary band
    pix_dict = dict()
    dir_files = os.listdir(globals_dir)
    n_globals = dict()
    fileband = primary_band
    if primary_band == 'y':
        fileband = 'Y'
    for primary_fname in dir_files:
        if re.match('gobjs_' + str(fileband) + '_nside' + str(nside) + '_p', primary_fname) is not None:
            pixel_match = re.search('p(\d+)', primary_fname)
            if pixel_match is None:
                print "Problem parsing pixel number"
            pixel = int(pixel_match.group()[1:])
            
            fileband = secondary_band
            if fileband == 'y':
                fileband = 'Y'
            
            # look for this pixel in the second band
            secondary_fname = 'gobjs_' + str(fileband) + '_nside' + str(nside) + '_p' + str(pixel)
            if not os.path.exists( os.path.join(globals_dir,secondary_fname) ):
                continue
            
            global_objects = {}
            secondary_file = open(os.path.join(globals_dir,secondary_fname), 'rb')
            global_objects[secondary_band] = cPickle.load(secondary_file)
            secondary_file.close()
            primary_file = open(os.path.join(globals_dir,primary_fname), 'rb')
            global_objects[primary_band] = cPickle.load(primary_file)
            primary_file.close()
            
            # match the objects
            ras_1 = [o.ra-360. if o.ra>300. else o.ra for o in global_objects[primary_band]]
            decs_1 = [o.dec for o in global_objects[primary_band]]
            ras_2 = [o.ra-360. if o.ra>300. else o.ra for o in global_objects[secondary_band]]
            decs_2 = [o.dec for o in global_objects[secondary_band]]
            inds1, inds2, dists = spherematch( ras_1, decs_1, ras_2, decs_2, tol=1./3600. )
            print "Found {0} matches in pixel {1}".format(len(inds1), pixel)
            
            # for each detection, apply the zps and get mean calibrated mags
            mean_mags = {}
            for n in range(len(inds1)):
                gos = []
                gos.append(global_objects[primary_band][inds1[n]])
                gos.append(global_objects[secondary_band][inds2[n]])
                for b, band in enumerate(primary_band, secondary_band):
                    ndet = len(gos[b].objects)
                    mags = np.zeros(ndet)
                    mag_errors2 = np.zeros(ndet)
                    d=0
                    for d0 in range(ndet):
                        spx = int(4*np.floor(go.objects[d]['y_image']/512.) + np.floor(go.objects[d]['x_image']/512.) + 32*(go.objects[d]['ccd']))
                        mags[d] = go.objects[d]['mag_psf']
                        mag_errors2[d] = go.objects[d]['magerr_psf']*go.objects[d]['magerr_psf'] + magerr_sys2
                        badmag = False
                        for zps in zp_list:
                            operand = 1
                            if zps['operand'] not in [1,None,'None']:
                                operand = go.objects[d][zps['operand']]
                            id_string = go.objects[d][zps['id_string']]
                            if id_string in zps:
                                mags[d] += operand*zps[id_string]
                            else:
                                badmag = True
                        if badmag:
                            continue
                        d+=1
                    if d == 0:
                        continue
                    mags = mags[0:d]
                    mag_errs = mag_errs[0:d]
                    invsigma_array = 1.0/np.sqr(mag_errors)
                    sum_invsigma2 = invsigma_array.sum()
                    if spx not in mean_mags:
                        mean_mags[spx] = {}
                    mean_mags[spx][band] = (mags*invsigma_array).sum() / sum_invsigma2
                    # mc_info.add(go.ra, go.dec, go.pixel, mean_mag[primary_band], mean_mag[primary_band]-mean_mag[secondary_band], spx)
                    # how to add color from a superpixel?
                if spx_std in mean_mags and primary_band in mean_mags[spx]:
                    for spx in mean_mags:
                        if spx == spx_std:
                            continue
                        if primary_band in mean_mags[spx] and secondary_band in mean_mags[spx]:
                        mc_info.add(go.ra, go.dec, go.pixel, mean_mags[spx][primary_band]-mean_mags[spx_std][primary_band], mean_mags[spx][primary_band]-mean_mags[spx][secondary_band], spx)
            
            # fit the color term.  plot the result!
            # g-g_spx0 = A(color - color_0)
            
            # write out the matches to a new set of primary global objects

    
    

if __name__ == '__main__':
    main()
