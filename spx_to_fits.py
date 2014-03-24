import os
import time
import copy
import shutil
import math
import datetime
import re

import sys

import numpy
import pyfits

if len(sys.argv) != 2:
    print "Usage: spx_to_fits.py zeropoint_filename"
    print "       requires a couple of auxiliary files given in the code"
    exit(1)

# read in the superpix zeropoints
n_ccds = 63
zpfile = open(sys.argv[1], 'r')
zps = numpy.zeros(32*n_ccds)
for line in zpfile:
    entries = line.split()
    zps[int(entries[0])] = float(entries[1])
zpfile.close()

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

input_path = "/Users/bauer/surveys/DES/starflats/december"
print "reading from directory %s for WCS info" %input_path
dir_list = os.listdir(input_path)
for file in dir_list:
    if re.match("DECam", file) is not None and re.search("fits$", file) is not None:
        break

# make the fits file that is the flat field
starflatname = "superpix.fits"
new_hdu = pyfits.PrimaryHDU()
new_hdu.header.update("NEXTEND", n_ccds)
pyfits.writeto(starflatname, data = numpy.array([]), header = new_hdu.header)

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

    for x in range(4):
        for y in range(8):
            spix = 4*y + x + 32*ccd_num
            zp = 10.0**( zps[spix]/2.5 )
            superpix = zp * numpy.ones((512,512))
            data_amp1[512*y:512*(y+1),512*x:512*(x+1)] = superpix
            data_amp2[512*y:512*(y+1),512*x:512*(x+1)] = superpix
    ccd_image.data[:,0:(width_int/2)-1] = data_amp1[:,0:(width_int/2)-1]
    ccd_image.data[:,(width_int/2):width_int] = data_amp2[:,(width_int/2):width_int]

    # Save image to the file
    # print "Finished writing ccd %d" %ccd_num
    pyfits.append(starflatname, data=ccd_image.data, header=ccd_image.header)
    print "Finished appending ccd %d" %ccd_num

print "Finished writing %s" %starflatname