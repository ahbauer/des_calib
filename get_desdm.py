#!/usr/bin/env python

import time,os,argparse
from collections import deque
import subprocess as sub
import glob
import pyfits
import shutil
import cx_Oracle

parser = argparse.ArgumentParser(description='Run single file')
parser.add_argument('--exposure',type=int,help='exposure number',default=-1)
parser.add_argument('--file',help='exposures from a file',
                    default="")
parser.add_argument('--start',type=int,default=-1,
                    help='starting exposure number')
parser.add_argument('--end',type=int,help='end exposure number')
parser.add_argument('--run',default='',help='run')
parser.add_argument('--firstcut',default=False,type=bool,
                    help='only use firstcut')
parser.add_argument('--finalcut',default=False,type=bool,
                    help='only use finalcut')
parser.add_argument('--outdir',default='',help='run')
parser.add_argument('--remove_old',type=bool,default=False,help='run')

args = parser.parse_args()
exp_list=[]

if args.exposure>0:
    exp_list.append(args.exposure)
elif args.file!="":
    file=open(args.file)
    for line in file:
        exp=int(line)
        exp_list.append(exp)
else:
     exp_list=range(args.start,args.end+1)


print exp_list
for exp in exp_list:

   
     
    connection = cx_Oracle.connect('ahbauer/ahb70chips@leovip148.ncsa.uiuc.edu/desoper')
    cursor = connection.cursor()
    run=args.run

    if run=="":
        query="select band from exposure where expnum=%d" % exp        
        cursor.execute(query)
        band=cursor.fetchall();
        #print band[0][0]
        #raw_input()

        if band[0][0] == 'r':
            print "Downloading exposure %d"%exp
        else:
            print band[0][0], ' skipping'
            continue
        if not args.firstcut and not args.finalcut:
            query="select distinct(image.run) from exposure,image where image.imagetype='red' and image.exposureid=exposure.id and expnum=%d order by run desc" % exp
        elif args.firstcut :
            query="select distinct(image.run) from exposure,image,run where image.imagetype='red' and image.exposureid=exposure.id and image.run=run.run and run.pipeline='firstcut' and expnum=%d order by image.run desc " % exp
        elif args.finalcut :
            query="select distinct(image.run) from exposure,image,run where image.imagetype='red' and image.exposureid=exposure.id and image.run=run.run and run.pipeline='finalcut' and expnum=%d order by image.run desc " % exp
            
        
        cursor.execute(query)
        results=cursor.fetchall();
        print results
        if len(results)==0:
            print 'Could not find exposure %d'%exp
            continue
        if len(results)>1:
            print 'Found multitple runs using latest'
                
        run=results[0][0]

    outdir='%s/DECam_%d' % (args.outdir,exp)

    if args.remove_old:
        if os.path.exists(outdir):
            os.system('rm '+outdir+'/*')


        
    #loc=('wget --no-check-certificate -r -l7 -nH --cut-dirs=7 https://desar2.cosmology.illinois.edu/DESFiles/desardata/OPS/red/%s/red/DECam_%08d/' % (run,exp))
    loc = ('wget -r -np -nH --cut-dirs=7 --no-check-certificate https://desar2.cosmology.illinois.edu/DESFiles/desardata/OPS/red/%s/red/DECam_%08d/' % (run,exp))
    #print loc
    #print outdir
    if not os.path.exists(outdir): os.makedirs (outdir)
    cmd = 'wget -P %s %s ' % (outdir,loc)
    #print cmd
    os.system(cmd)

