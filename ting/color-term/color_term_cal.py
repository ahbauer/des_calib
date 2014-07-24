# Here are the functions computing the color term. 
# Inputs are airmass, mjd, g-r color of the star, radial position of the star at the focal plane.
# Output is one correction per star, or one class mag_all per exposure.
# One example is given at the end .
# Email to Ting Li @ sazabi@neo.tamu.edu if you have any questions

import pyfits
import pylab as pyl
import numpy as np
import os

# find the nearest value for pwv and airmass
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

# get the pwv from the GPS
def get_pwv(mjd):
    gps_data = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),'pwv/SuominetResults_revised.dat'), dtype='str')
    mjd_list = gps_data[:,2].astype('float')
    pwv_list = gps_data[:,4].astype('float')
    pwv = np.interp(mjd, mjd_list, pwv_list)
    if pwv < 0 : 
        pwv = 3
        print "Warning!! Water vapor wasn't measure properly; a fiduical water vapor is used"
    return pwv

# define the structure for the transformation class
class mag_all():
    def __init__(self,n):
        self.gr0 = np.zeros(n)
        self.mag0 = np.zeros(n)   # fiducial magnitude
        self.mag1 = np.zeros(n)   # magnitude at position 1, 2, 3, 4
        self.mag2 = np.zeros(n)
        self.mag3 = np.zeros(n)
        self.mag4 = np.zeros(n)
        self.gr_ref = 0        # the color and magnitude for the reference star
        self.mag0_ref = 0
        self.mag1_ref = 0
        self.mag2_ref = 0
        self.mag3_ref = 0
        self.mag4_ref = 0

# calculate the synthetic magnitude
def mag_cal(spectrum,bandpass,filter_num=1,spectrum_num=1,atmo=1,z=0):
    c= 3E18 # unit Ang/s

    if isinstance(spectrum, type('string')): 
        spectrum = np.loadtxt(spectrum)
    if isinstance(bandpass, type('string')): 
        bandpass = np.loadtxt(bandpass)
    if isinstance(atmo, type('string')): 
        atmo = np.loadtxt(atmo)
    lamb = np.arange(3000,11000.1,1)
    if isinstance(atmo, np.ndarray):
        trans = np.interp(lamb, atmo[:,0]*10, atmo[:,1])*np.interp(lamb, bandpass[:,0]*10, bandpass[:,filter_num],left=0,right=0)

    else:
        trans = np.interp(lamb, bandpass[:,0]*10, bandpass[:,filter_num],left=0,right=0)
    flambda=np.interp(lamb, spectrum[:,0]*(1+z), spectrum[:,spectrum_num],left=0,right=0)
    num_integrand=flambda*trans*lamb
    numer=np.trapz(num_integrand,lamb)
    syn_mag= - 2.5 * np.log10(numer)
    return syn_mag   

# input are:  band on of 'g','r','i','z','y', pwv(mm), airmass, return a class with transformation at different radii
# run this every exposure (or every CCD is the airmass difference is large from one side to another side)
def color_term_correction(band, pwv=3, airmass=1.3):  #default is pwv=3mm, airmass=1.3  
    '''
    # get the fiducial system response + atmospheric throughput
    # https://cdcvs.fnal.gov/redmine/projects/descalibration/wiki
    referene=np.loadtxt('fiducial/system_atm_tput.csv', delimiter = ',')
    reference[:,0]=reference[:,0]*10
    if band == 'g': col = 2
    if band == 'r': col = 3
    if band == 'i': col = 4
    if band == 'z': col = 5
    if band == 'y': col = 6
    # end up not using this one because this throughput considered the out-of-band light-leaking.
    # However, I set all out-of-band throughput to be zero for the throughtput at different radii.
    '''
    # fiducial atmospheric throughtput pwv=3mm, airmass=1.3
    atmo0=os.path.join(os.path.dirname(os.path.realpath(__file__)),'database/uvspec_afglus_pressure780_airmass1.3_asea1_avul1_pw3.0_tau0.00.out')
    
    mag = mag_all(131)  # 131 pickle spectra
    
    # get the closest atmoshperic transmisssion model based on the airmass and water vapor
    pwv_list = np.arange(0.0, 15.0, 0.5)
    if pwv < 15: pwv_nearest = find_nearest(pwv_list,pwv)
    if pwv >= 15: 
        pwv_nearest = 14.5
        print 'Warning!! PWV is ' + str(pwv) + ' but we used pwv=14.5mm for the correction'
        
    airmass_list = np.arange(1.0, 2.0, 0.01)
    if airmass < 2: airmass_nearest = find_nearest(airmass_list,airmass)
    if airmass >= 2: 
        airmass_nearest = 1.99
        print 'Warning!! Airmass is ' + str(airmass) + ' but we used airmass=1.99 for the correction'

    atmo=os.path.join(os.path.dirname(os.path.realpath(__file__)),'database/uvspec_afglus_pressure780_airmass' + str(airmass_nearest) + '_asea1_avul1_pw' + str(pwv_nearest) + '_tau0.00.out')

        
    # Now calculate the synthetic magnitude, mag.mag0 is synthetic magnitude with fiducial throughput, 
    # mag.mag1-4 is synthetic magnitude at 4 different position with a specific atmospheric condition
    SpecDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Pickles/dat_uvk/')
    SpecList = 'Pickle.lst'
    SpecName = SpecDIR + SpecList
    k = 0
    for ii in file(SpecName):
        spectrum = SpecDIR + ii
        hdr = pyfits.getheader(spectrum)
        tab = pyfits.getdata(spectrum)
        objectLam = tab.field('WAVELENGTH')
        objectFlux = tab.field('FLUX')
        spectrum = np.column_stack((objectLam, objectFlux))
        if band == 'r':
            end_file = '_005.dat'
        else: 
            end_file = '_003.dat'
        mag.gr0[k] = mag_cal(spectrum, os.path.join(os.path.dirname(os.path.realpath(__file__)),'filter_curves/g_003.dat'), 1, 1, atmo0) - mag_cal(spectrum,os.path.join(os.path.dirname(os.path.realpath(__file__)),'filter_curves/r_005.dat'), 1, 1, atmo0)
        mag.mag0[k] = mag_cal(spectrum, os.path.join(os.path.dirname(os.path.realpath(__file__)),'filter_curves/'+band+end_file), 1, 1, atmo0)
        mag.mag1[k] = mag_cal(spectrum, os.path.join(os.path.dirname(os.path.realpath(__file__)),'filter_curves/'+band+end_file), 2, 1, atmo)
        mag.mag2[k] = mag_cal(spectrum, os.path.join(os.path.dirname(os.path.realpath(__file__)),'filter_curves/'+band+end_file), 3, 1, atmo)
        mag.mag3[k] = mag_cal(spectrum, os.path.join(os.path.dirname(os.path.realpath(__file__)),'filter_curves/'+band+end_file), 4, 1, atmo)
        mag.mag4[k] = mag_cal(spectrum, os.path.join(os.path.dirname(os.path.realpath(__file__)),'filter_curves/'+band+end_file), 5, 1, atmo)
        if k==25:   # define G2V is the reference star
            mag.gr_ref=mag.gr0[k]
            mag.mag0_ref=mag.mag0[k]
            mag.mag1_ref=mag.mag1[k]
            mag.mag2_ref=mag.mag2[k]
            mag.mag3_ref=mag.mag3[k]
            mag.mag4_ref=mag.mag4[k]
        k+=1
    # pyl.figure(figsize=[8,6])
    # pyl.plot(mag.gr0,mag.mag1-mag.mag0-mag.mag1_ref+mag.mag0_ref,'o',label='r~5%Rmax') 
    # pyl.plot(mag.gr0,mag.mag2-mag.mag0-mag.mag2_ref+mag.mag0_ref,'o',label='r~15%Rmax') 
    # pyl.plot(mag.gr0,mag.mag3-mag.mag0-mag.mag3_ref+mag.mag0_ref,'o',label='r~45%Rmax') 
    # pyl.plot(mag.gr0,mag.mag4-mag.mag0-mag.mag4_ref+mag.mag0_ref,'o',label='r~80%Rmax') 
    # # mag1-4 is corrected by the reference star
    # pyl.plot(mag.gr_ref,0,'o',markersize=5,markerfacecolor='w',markeredgewidth=3,markeredgecolor='y')
    # pyl.xlabel('g-r')
    # pyl.ylabel('$\Delta$'+band)
    # pyl.legend(loc=2)
    return mag 
    # mag is a class with the transformation at 4 different position from the fiducial throughput to the actual throughput 

# input the class of the transfomation, the g-r color of the star, the position on the focal plane
# return the correction to that specific star.
# run this on every star
def fit_color_position(mag,gr=0.5,radius=0.45): # defaual is g-r=0.5, radius=0.45
    order=3     # making 3rd-order polynomial fitting for the color
    x=np.linspace(-0.5,2,100)
    if gr> 2: 
        print 'Warning!! the g-r color of the object is redder than 2. Correction might be not right.'
    if gr<-0.5: 
        print 'Warning!! the g-r color of the object is bluer than -0.5. Correction might be not right.'
    p1=np.poly1d(np.polyfit(mag.gr0,mag.mag1-mag.mag0-mag.mag1_ref+mag.mag0_ref,order))
    p2=np.poly1d(np.polyfit(mag.gr0,mag.mag2-mag.mag0-mag.mag2_ref+mag.mag0_ref,order))
    p3=np.poly1d(np.polyfit(mag.gr0,mag.mag3-mag.mag0-mag.mag3_ref+mag.mag0_ref,order))
    p4=np.poly1d(np.polyfit(mag.gr0,mag.mag4-mag.mag0-mag.mag4_ref+mag.mag0_ref,order))
    # pyl.plot(x,p1(x),'k')
    # pyl.plot(x,p2(x),'b')
    # pyl.plot(x,p3(x),'g')
    # pyl.plot(x,p4(x),'r')
    pos=np.array([0.05,0.15,0.45,0.80])
    corr_pos=np.array([p1(gr),p2(gr),p3(gr),p4(gr)])
    # here a linear interpolation for the correction between 4 position.
    if radius < 0.05: 
        corr = p1(gr) - (p2(gr) - p1(gr)) / (0.15 - 0.05) * (0.05-radius)
        # print 'Warning!! the position of the star on the focal plane is less than 5% of the Rmax. Extropolation for the correction'
    if radius < 0.80 and radius > 0.05: 
        corr=np.interp(radius, pos, corr_pos)
    if radius > 0.80: 
        corr = p4(gr) + (p4(gr) - p3(gr)) / (0.80 - 0.45) * (radius - 0.80)
        # print 'Warning!! the position of the star on the focal plane is more than 80% of the Rmax. Extropolation for the correction'
    # pyl.plot(np.ones(4)*gr,corr_pos)
    # pyl.plot(gr,corr,'y*',markersize=15) # the yellow star in the figure is the final correction to that star
    return corr # corr is the correction for that specific star at that specific position


# # an example of how to use the functions above
# mjd=56515.7
# airmass=1.5
# gr=1.7
# r_pos=0.9
# band='z'
# pwv=get_pwv(56515.7)   # get the pwv 
# mag_class=color_term_correction(band,pwv,airmass) # color_term_correction(band grizy, pwv, airmass)
# #mag_class is a class with all the transfomations for stars with any colors and at four position over the focal plane
# correction=fit_color_position(mag_class,gr,r_pos)  # fit_color_position(mag_class,g-r color, r/Rmax)
# #correction is one number for a specific star at a specific position
# pyl.tight_layout()
# pyl.show()
