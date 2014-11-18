import sys
from tables import *

class FinalCutObj(IsDescription):
    exposureid = Int64Col()
    imageid = Int64Col()
    object_id = Int64Col()
    zeropoint = Float32Col()
    flags = Int64Col()
    x_image = Float32Col()
    y_image = Float32Col()
    ra = FloatCol()
    dec = FloatCol()
    mag_psf = Float32Col()
    magerr_psf = Float32Col()
    spread_model = Float32Col()
    skybrite = Float32Col()
    skysigma = Float32Col()
    exptime = Float32Col()
    airmass = Float32Col()
    ha = StringCol(16)
    zd = Float32Col()
    telera = Float32Col()
    teledec = Float32Col()
    mjd = Float32Col()
    band = StringCol(4)
    ccd = Int32Col()
    image_fwhm_arcsec = Float32Col()
    fwhm_arcsec = Float32Col()
    image_ellipt = Float32Col()
    gskyphot = Int32Col()
    lskyphot = Int32Col()
    gskyhot = Float32Col()
    lskyhot = Float32Col()
    cloud_nomad = Float32Col()
        
def assign_obj(object, entries, header):
    int_entries = ['exposureid','imageid','object_id','flags','ccd']
    float_entries = ['x_image','y_image','ra','dec','mag_psf','magerr_psf','zeropoint','fwhm_arcsec','spread_model','airmass','zd',
                     'telera','teledec','mjd','exptime','skybrite','skysigma','image_fwhm_arcsec']
    string_entries = ['band','ha']
    
    for entry in int_entries:
        object[entry] = int(entries[header[entry]])
    for entry in float_entries:
        object[entry] = float(entries[header[entry]])
    for entry in string_entries:
        object[entry] = entries[header[entry]]
    
    if entries[header['cloud_nomad'] == '':
        print 'Filling in for an empty cloud_nomad'
        object['cloud_nomad'] = 0.5 # give missing values a bad value
    else:
        object['cloud_nomad'] = float(entries[header['cloud_nomad']])
    if entries[header['gskyphot']] == 'T':
        object['gskyphot'] = 1
    else:
        object['gskyphot'] = 0
    if entries[header['lskyphot']] == 'T':
        object['lskyphot'] = 1
    else:
        object['lskyphot'] = 0
    object['gskyhot'] = float(entries[header['gskyhot']])
    if entries[header['lskyhot']]:
        object['lskyhot'] = float(entries[header['lskyhot']])
    else:
        object['lskyhot'] = 0.0

def parse_header(header_line):
    header = {}
    index0 = 0
    entries = header_line.split(',')
    for e, entry in enumerate(entries):
        if( entries[0] == '#' ):
            index0 = 1
            continue
        header[entry] = e-index0
    return header


infilename = sys.argv[1]


h5filename = "finalcutout.h5"
h5file = openFile(h5filename, mode = "w", title = "Y1A1 EQU")
group = h5file.createGroup("/", 'data', 'Data')
table_u = h5file.createTable(group, 'table_u', FinalCutObj, "u band")
table_g = h5file.createTable(group, 'table_g', FinalCutObj, "g band")
table_r = h5file.createTable(group, 'table_r', FinalCutObj, "r band")
table_i = h5file.createTable(group, 'table_i', FinalCutObj, "i band")
table_z = h5file.createTable(group, 'table_z', FinalCutObj, "z band")
table_y = h5file.createTable(group, 'table_y', FinalCutObj, "y band")

object_u = table_u.row
object_g = table_g.row
object_r = table_r.row
object_i = table_i.row
object_z = table_z.row
object_y = table_y.row

file = open(infilename, 'r')
header_line = file.next()
header = parse_header( header_line )
for line in file:
    entries = line.split(",")
    # header!
    if entries[0] == "exposureid":
        continue

    if entries[18] == 'u':
        assign_obj( object_u, entries, header )
        object_u.append()
    elif entries[18] == 'g':
        assign_obj( object_g, entries, header )
        object_g.append()
    elif entries[18] == 'r':
        assign_obj( object_r, entries, header )
        object_r.append()
    elif entries[18] == 'i':
        assign_obj( object_i, entries, header )
        # print "{0} {1} {2}".format(object_i['ra'], object_i['dec'], object_i['gskyhot'])
        object_i.append()
    elif entries[18] == 'z':
        assign_obj( object_z, entries, header )
        object_z.append()
    elif entries[18] == 'Y':
        assign_obj( object_y, entries, header )
        object_y.append()

table_u.flush()
table_g.flush()
table_r.flush()
table_i.flush()
table_z.flush()
table_y.flush()

print "Tables added with lengths %d %d %d %d %d %d" %(table_u.nrows, table_g.nrows, table_r.nrows, table_i.nrows, table_z.nrows, table_y.nrows)

h5file.close()
file.close()


  #  3 # Define a user record to characterize some kind of particles
  #  4 class Particle(IsDescription):
  #  5     name      = StringCol(16)   # 16-character String
  #  6     idnumber  = Int64Col()      # Signed 64-bit integer
  #  7     ADCcount  = UInt16Col()     # Unsigned short integer
  #  8     TDCcount  = UInt8Col()      # unsigned byte
  #  9     grid_i    = Int32Col()      # integer
  # 10     grid_j    = Int32Col()      # integer
  # 11     pressure  = Float32Col()    # float  (single-precision)
  # 12     energy    = FloatCol()      # double (double-precision)
  # 13 
  # 14 filename = "test.h5"
  # 15 # Open a file in "w"rite mode
  # 16 h5file = open_file(filename, mode = "w", title = "Test file")
  # 17 # Create a new group under "/" (root)
  # 18 group = h5file.create_group("/", 'detector', 'Detector information')
  # 19 # Create one table on it
  # 20 table = h5file.create_table(group, 'readout', Particle, "Readout example")
  # 21 # Fill the table with 10 particles
  # 22 particle = table.row
  # 23 for i in xrange(10):
  # 24     particle['name']  = 'Particle: %6d' % (i)
  # 25     particle['TDCcount'] = i % 256
  # 26     particle['ADCcount'] = (i * 256) % (1 << 16)
  # 27     particle['grid_i'] = i
  # 28     particle['grid_j'] = 10 - i
  # 29     particle['pressure'] = float(i*i)
  # 30     particle['energy'] = float(particle['pressure'] ** 4)
  # 31     particle['idnumber'] = i * (2 ** 34)
  # 32     # Insert a new particle record
  # 33     particle.append()
  # 34 # Close (and flush) the file
  # 35 h5file.close()
