import sys
import os
import numpy as np
import pyfits

def main():
    
    outfitsname = None
    output_filename = None
    primary_dirs = None
    secondary_dirs = None
    
    stripe82 = False
    spt = True
    
    if stripe82:
        outfitsname = 'zps_s82.fits'
        output_filename = '/Users/bauer/surveys/DES/y1p1/equatorial/output_file.txt'
        primary_dirs = ['/Users/bauer/surveys/DES/y1a1/challenge/g/wlabels/s82/wcolor',
        '/Users/bauer/surveys/DES/y1a1/challenge/r/wlabels/s82/wcolor',
        '/Users/bauer/surveys/DES/y1a1/challenge/i/wlabels/s82/wcolor',
    #    '/Users/bauer/surveys/DES/y1a1/challenge/z/wlabels/s82/wcolor',
        '/Users/bauer/surveys/DES/y1a1/challenge/y/wlabels/s82/wcolor']
        primary_dirs = []
        secondary_dirs = ['/Users/bauer/surveys/DES/y1a1/challenge/g/wlabels/s82/nocolor',
        '/Users/bauer/surveys/DES/y1a1/challenge/r/wlabels/s82/nocolor',
        '/Users/bauer/surveys/DES/y1a1/challenge/i/wlabels/s82/nocolor',
        '/Users/bauer/surveys/DES/y1a1/challenge/z/wlabels/s82/nocolor',
        '/Users/bauer/surveys/DES/y1a1/challenge/y/wlabels/s82/nocolor']
    
    elif spt:
        outfitsname = 'zps_spt.fits'
        output_filename = '/Users/bauer/surveys/DES/y1a1/spt_lt0/output_spt_lt0.txt'
        primary_dirs = ['/Users/bauer/surveys/DES/y1a1/challenge/g/wlabels/sptlt0/wcolor',
        '/Users/bauer/surveys/DES/y1a1/challenge/r/wlabels/sptlt0/wcolor',
        '/Users/bauer/surveys/DES/y1a1/challenge/i/wlabels/sptlt0/wcolor',
        '/Users/bauer/surveys/DES/y1a1/challenge/z/wlabels/sptlt0/wcolor']
        secondary_dirs = ['/Users/bauer/surveys/DES/y1a1/challenge/g/wlabels/sptlt0/nocolor',
        '/Users/bauer/surveys/DES/y1a1/challenge/r/wlabels/sptlt0/nocolor',
        '/Users/bauer/surveys/DES/y1a1/challenge/i/wlabels/sptlt0/nocolor',
        '/Users/bauer/surveys/DES/y1a1/challenge/z/wlabels/sptlt0/nocolor']
    
    print 'Making a fits table from primary directories {0}, secondary directories {1}'.format(primary_dirs, secondary_dirs)
    
    zps = {}
    # loop through secondary directories
    for d in secondary_dirs:
        print 'Reading secondary directory {0}'.format(d)
        # read in CCD zps (with labels)
        ccdzps = {}
        stdzps = {}
        expzps = {}
        for f in os.listdir(d):
            if f.find('nebencal_ccd_zps_') >= 0:
                print 'Found ccd zps'
                zpfile = open(os.path.join(d,f), 'r')
                for line in zpfile:
                    entries = line.split()
                    ccd = int(entries[0])
                    if ccd not in ccdzps:
                        ccdzps[ccd] = {}
                    ccdzps[ccd][int(entries[2])] = float(entries[1])
                zpfile.close()
            # read in std zps (with labels)
            elif f.find('nebencal_stds_zp_') >= 0:
                print 'Found std zps'
                zpfile = open(os.path.join(d,f), 'r')
                for line in zpfile:
                    entries = line.split()
                    stdzps[int(entries[0])] = float(entries[1])
                zpfile.close()
            # read in exposure zps (with labels)
            elif f.find('nebencal_exp_zps_') >= 0:
                print 'Found exposure zps'
                zpfile = open(os.path.join(d,f), 'r')
                for line in zpfile:
                    entries = line.split()
                    eid = int(entries[0])
                    if eid not in expzps:
                        expzps[eid] = {}
                    expzps[eid][int(entries[2])] = float(entries[1])
                zpfile.close()
        
        # add this info to the zp dictionary
        for exp in expzps.keys():
            for label in expzps[exp].keys():
                if label not in stdzps:
                    continue
                for ccd in ccdzps.keys():
                    if label not in ccdzps[ccd]:
                        continue
                    # add the info to the zp dict
                    if exp not in zps:
                        zps[exp] = {}
                    zps[exp][ccd] = ccdzps[ccd][label] + stdzps[label] + expzps[exp][label]
        
    # now do the same for the primary dirs, overwriting the secondary info.
    for d in primary_dirs:
        print 'Reading primary directory {0}'.format(d)
        # read in CCD zps (with labels)
        ccdzps = {}
        stdzps = {}
        expzps = {}
        for f in os.listdir(d):
            if f.find('nebencal_ccd_zps_') >= 0:
                print 'Found ccd zps'
                zpfile = open(os.path.join(d,f), 'r')
                for line in zpfile:
                    entries = line.split()
                    ccd = int(entries[0])
                    if ccd not in ccdzps:
                        ccdzps[ccd] = {}
                    ccdzps[ccd][int(entries[2])] = float(entries[1])
                zpfile.close()
            # read in std zps (with labels)
            elif f.find('nebencal_stds_zp_') >= 0:
                print 'Found std zps'
                zpfile = open(os.path.join(d,f), 'r')
                for line in zpfile:
                    entries = line.split()
                    stdzps[int(entries[0])] = float(entries[1])
                zpfile.close()
            # read in exposure zps (with labels)
            elif f.find('nebencal_exp_zps_') >= 0:
                print 'Found exposure zps'
                zpfile = open(os.path.join(d,f), 'r')
                for line in zpfile:
                    entries = line.split()
                    eid = int(entries[0])
                    if eid not in expzps:
                        expzps[eid] = {}
                    expzps[eid][int(entries[2])] = float(entries[1])
                zpfile.close()
        
        # add this info to the zp dictionary
        for exp in expzps.keys():
            for label in expzps[exp].keys():
                if label not in stdzps:
                    continue
                for ccd in ccdzps.keys():
                    if label not in ccdzps[ccd]:
                        continue
                    # add the info to the zp dict
                    zps[exp][ccd] = ccdzps[ccd][label] + stdzps[label] + expzps[exp][label]
    
    # now go through and add obs info from the output filename
    print 'Reading the huge output file'
    obsinfo = {}
    outputfile = open(output_filename, 'r')
    header = outputfile.next()
    header_entries = header.split(',')
    assert header_entries[0] == 'exposureid', '{0} != exposureid'.format(header_entries[0])
    assert header_entries[1] == 'imageid', '{0} != imageid'.format(header_entries[1])
    assert header_entries[19] == 'ccd', '{0} != ccd'.format(header_entries[19])
    assert header_entries[35] == 'exposurename', '{0} != exposurename'.format(header_entries[35])
    # assert header_entries[36] == 'expnum', '{0} != expnum'.format(header_entries[36])
    for line in outputfile:
        entries = line.split(',')
        eid = int(entries[0])
        imageid = int(entries[1])
        ccd = int(entries[19])
        exposurename = entries[35]
        if eid not in obsinfo:
            obsinfo[eid] = {}
        obsinfo[eid][ccd] = {'imageid':imageid, 'exposurename':exposurename}
    outputfile.close()
    
    # print obsinfo.keys()
    # now write out the zpdict to a fits table
    print 'Writing output'
    expcolumn = []
    ccdcolumn = []
    zpcolumn = []
    expnamecolumn = []
    imageidcolumn = []
    for exp in zps.keys():
        if exp not in obsinfo:
            print 'WARNING, can\'t find {0} in obsinfo!...'.format(exp)
            continue
        for ccd in zps[exp].keys():
            if ccd not in obsinfo[exp]:
                print 'WARNING, can\'t find {0} in obsinfo[{1}]'.format(ccd,exp)
                continue
            expcolumn.append(exp)
            ccdcolumn.append(ccd)
            zpcolumn.append(zps[exp][ccd])
            expnamecolumn.append(obsinfo[exp][ccd]['exposurename'])
            imageidcolumn.append(obsinfo[exp][ccd]['imageid'])
    expcolumn = np.array(expcolumn)
    ccdcolumn = np.array(ccdcolumn)
    zpcolumn = np.array(zpcolumn)
    expnamecolumn = np.array(expnamecolumn)
    imageidcolumn = np.array(imageidcolumn)
    col1 = pyfits.Column(name='EXPOSURE_iD', format='K', array=expcolumn)
    col2 = pyfits.Column(name='CCD', format='I', array=ccdcolumn)
    col3 = pyfits.Column(name='IMAGEID', format='K', array=imageidcolumn)
    col4 = pyfits.Column(name='EXPOSURENAME', format='32A', array=expnamecolumn)
    col5 = pyfits.Column(name='ZPS', format='E', array=zpcolumn)
    cols = pyfits.ColDefs([col1, col2, col3, col4, col5])
    tbhdu = pyfits.new_table(cols)
    tbhdu.writeto(outfitsname)

if __name__ == '__main__':
    main()