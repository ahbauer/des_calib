""" 
nebencal_utils:

functions used in both nebencal and nebencal_plots

AHB 4/2014
"""

class global_object(object):
    def __init__(self):
        self.ra = None
        self.dec = None
        self.color = None
        self.objects = []
        
def read_precam( precam_stars, precam_map, filename, band ):
    precam_file = open( filename, 'r' )
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
        star['x_image'] = 1.
        star['y_image'] = 1.
        star['fp_r'] = 1.
        star['secz'] = 1.
        star['ccd'] = 0
        star['image_id'] = 1
        star['exposureid'] = 1
        star['superpix'] = -1
        star['airmass'] = 0
        star['color'] = 1.e-9
        star['cloud_nomad'] = 1.0
        star['band'] = band
        star['matched'] = False
        precam_stars.append(star)
        precam_map.insert( count, (star['ra'],star['dec'],star['ra'],star['dec']) )
        count += 1
    print "Read in %d PreCam standards" %count


def read_sdss( sdss_stars, sdss_map, filename, band ):
    sdssfile = open(filename, 'r')
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
        sdss_obj['x_image'] = 1.
        sdss_obj['y_image'] = 1.
        sdss_obj['fp_r'] = 1.
        sdss_obj['secz'] = 1.
        sdss_obj['ccd'] = 0
        sdss_obj['image_id'] = 1
        sdss_obj['exposureid'] = 1
        sdss_obj['superpix'] = -1
        sdss_obj['airmass'] = 0
        sdss_obj['color'] = 1.e-9
        sdss_obj['cloud_nomad'] = 1.0
        sdss_obj['matched'] = False
        if sdss_obj['mag_psf'] > 0.:
            sdss_stars.append(sdss_obj)
            sdss_map.insert( count, (sdss_obj['ra'],sdss_obj['dec'],sdss_obj['ra'],sdss_obj['dec']) )
            count += 1
    print "Read in %d SDSS standards" %count

def read_betoule( sdss_stars, sdss_map, filename, band ):
    sdssfile = open(filename, 'r')
    count=0
    for line in sdssfile:
        entries = line.split(" ")
        sdss_obj = dict()
        sdss_obj['ra'] = float(entries[0])
        sdss_obj['dec'] = float(entries[1])
        sdss_obj['mag_psf'] = float(entries[2])
        sdss_obj['magerr_psf'] = float(entries[3])
        
        sdss_obj['band'] = band
        sdss_obj['x_image'] = 1.
        sdss_obj['y_image'] = 1.
        sdss_obj['fp_r'] = 1.
        sdss_obj['secz'] = 1.
        sdss_obj['ccd'] = 0
        sdss_obj['image_id'] = 1
        sdss_obj['exposureid'] = 1
        sdss_obj['superpix'] = -1
        sdss_obj['airmass'] = 0
        sdss_obj['color'] = 1.e-9
        sdss_obj['cloud_nomad'] = 1.0
        sdss_obj['matched'] = False
        if sdss_obj['mag_psf'] > 0.:
            sdss_stars.append(sdss_obj)
            sdss_map.insert( count, (sdss_obj['ra'],sdss_obj['dec'],sdss_obj['ra'],sdss_obj['dec']) )
            count += 1
    print "Read in %d Betoule standards" %count
