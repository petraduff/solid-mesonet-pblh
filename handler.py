import pandas as pd
from datetime import timedelta,datetime
import os
import argparse
import traceback
import getPBLh as alg

parser = argparse.ArgumentParser(
        description='Make PBLh files from beta and w .csv files stored on the local device. Saves to a new folder called "processed" in the target folder.',
        epilog='Written by Petra Duff (petra.duff@utoronto.ca) April 2026.'
        )
parser.add_argument('lidarLoc',help='Lidar site location code.')
parser.add_argument('inPath',help='Path to beta and w .csv files.')
parser.add_argument('startDate',nargs='?',default='20240219',help='Desired start date in YYYYMMDD format.')
parser.add_argument('endDate',nargs='?',default='20240220',help='Desired end date in YYYYMMDD format.')
parser.add_argument('-s','--suffix',default='',
        help='Suffix for file saving (default none).')
parser.add_argument('--cloud_threshold',type=float,default='-4.7',
        help='Log beta threshold for cloud height estimation (default -4.7).')
parser.add_argument('--wind_threshold',type=float,default='1.5',
        help='Wind variance threshold for threshold method (default 1.5).')
parser.add_argument('--maxHgt',type=int,default='2000',
        help='Max height for consideration (default 2000).')
parser.add_argument('--minHgt',type=int,default='100',
        help='Min height for consideration (default 100).')
args = parser.parse_args()

outPath = os.path.join(args.inPath,'processed',args.lidarLoc,args.suffix,'')
if not os.path.exists(outPath):
    os.mkdir(outPath)

start_date = datetime.strptime(args.startDate,"%Y%m%d")
end_date = datetime.strptime(args.endDate,"%Y%m%d")
    
cur_date = start_date

while cur_date <= end_date:

        betaFile = os.path.join(args.inPath,args.lidarLoc+'_beta_'+cur_date.strftime('%Y%m%d')+'.csv')
        windFile = os.path.join(args.inPath,args.lidarLoc+'_w_'+cur_date.strftime('%Y%m%d')+'.csv')
        intensityFile = os.path.join(args.inPath,args.lidarLoc+'_intensity_'+cur_date.strftime('%Y%m%d')+'.csv')

        print("Reading input files for date ",cur_date.strftime('%Y/%m/%d')) 
        try:   
                dfIntensity = pd.read_csv(intensityFile,index_col='time',parse_dates=True)
        except:
                print("Could not read intensity file for date ",cur_date.strftime("%Y/%m/%d"), " due to error:")
                print(traceback.format_exc())
                print("Continuing to next day...")
                continue

        try:   
                dfBeta = pd.read_csv(betaFile,index_col='time',parse_dates=True)
        except:
                print("Could not read beta file for date ",cur_date.strftime("%Y/%m/%d"), " due to error:")
                print(traceback.format_exc())
                print("Continuing to next day...")
                continue

        try:
                dfWind = pd.read_csv(windFile,index_col='time',parse_dates=True)
        except:
                print("Could not read wind file for date ",cur_date.strftime("%Y/%m/%d"), " due to error:")
                print(traceback.format_exc())
                print("Continuing to next day...")
                continue

        print("Calculating PBLh...")
        beta_pbl_var = alg.pblHeight(dfBeta, dfIntensity, method="beta",stat="var", 
                                maxHgt = args.maxHgt, minHgt=args.minHgt, lidar_loc=args.lidarLoc)
        beta_cbh = alg.cloudHeight(dfBeta, dfIntensity, stat="mean",cloud_threshold=args.cloud_threshold,maxHgt=args.maxHgt,minHgt=args.minHgt)

        pbl_wind = alg.pblHeight(dfWind, dfIntensity, method="threshold",stat="var",  threshold = args.wind_threshold, 
                                maxHgt = args.maxHgt, minHgt=args.minHgt, lidar_loc=args.lidarLoc)
        pbl_wind_grad = alg.pblHeight(dfWind, dfIntensity, method="gradient",stat="var", threshold = args.wind_threshold, 
                                maxHgt = args.maxHgt, minHgt=args.minHgt, lidar_loc=args.lidarLoc)

        print("Writing to file...")                              
        toWrite = pd.concat([beta_cbh,pbl_wind,pbl_wind_grad,beta_pbl_var],axis=1) #Note: beta_cbh is UTC and the rest ET, pd.concat() converts everything to UTC in this case
        toWrite.index = toWrite.index.tz_localize(None)
        titleFile = args.lidarLoc+'_PBLh_'+cur_date.strftime("%Y%m%d")+'.txt'
        fullTitle=outPath+titleFile
        if os.path.exists(fullTitle): 
                os.remove(fullTitle)
        toWrite.to_csv(fullTitle, float_format= '%s',
                        header=['Cloud base height (m)', 
                        'PBL height doppler variance threshold method (m)', 
                        'PBL height doppler variance gradient method (m)', 
                        'PBLh backscatter method from variance (m)'],
                        index_label='Timestamp (UTC)')
        print("Done!")
        cur_date += timedelta(days=1)
        
    