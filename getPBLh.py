import pandas as pd
import numpy as np

from zoneinfo import ZoneInfo
from datetime import datetime,timedelta
from astral import sun, Observer
import sys
import copy

import statsmodels.tsa.api as smt
from scipy.signal import savgol_filter,find_peaks


def drop_outliers(ts):
    Q1 = ts.quantile(0.25)
    Q3 = ts.quantile(0.75)
    IQR = Q3 - Q1 
    outliers = (ts < Q1 - 1.5 * IQR) | (ts > Q3 + 1.5 * IQR)
    return ts.mask(outliers)

def qualitycontrol(df,dfIntensity,var_type='beta',outpath='placeholder'):
    # Perform QAQC procedures in an attempt to ensure noisy measurements beyond the sensitivity range that may bias the estimate are filtered out 

    # Filter out low SNR measurements according to Mariani et al. (2018)
    snrMask = dfIntensity < 1.004
    df = df.mask(snrMask)
    
    if var_type == 'wind':
        
        for t,ts in df.iterrows():
            # Identify any consecutive jumps indicating the limits of the sensitivity range and filter out everything above
            largeJumpSeries = ((ts - ts.shift(1)).abs() > 2.) & ((ts - ts.shift(3)).abs() > 2.) & ((ts - ts.shift(5)).abs() > 2.)
            whereJumps = ts.where(largeJumpSeries)

            if whereJumps.first_valid_index() == None:
                continue
            else:
                ts.loc[whereJumps.first_valid_index():] = np.nan

            df.loc[t] = ts

        # Drop outliers  
        df = drop_outliers(df)

        #Filter out any points with sharp changes
        largeJumps = ((df - df.shift(1)).abs() > 2.) & (df.shift(1).abs() > 3.)
        df = df.mask(largeJumps)

        #Filter out unphysical values
        largeValMask = df > 5.
        df = df.mask(largeValMask)
    else:
        #Filter out negative (unphysical) aerosol backscatter coefficients
        negMask = df < 0.
        df = df.mask(negMask)

    return df

def instrument_variance(ts,var_type='wind'):
    # Calculates instrument variance to be subtracted from total variance according to Lernschow et al. (2000)

    l = 4 #Use lag of 4 as per Lernschow et al. (2000)
    try:
        ts_acov = smt.acovf(ts,fft=False,nlag=l-1) # Find first few lags of autocovariance
    except ValueError:
        return np.nan
    M_0 = ts_acov[0] #Get autocovariance at lag 0
    try:
        M_fit = np.polyfit(range(l),ts_acov,1) # Get linear fit of the first few lags
    except np.linalg.LinAlgError:
        return np.nan
    M_lim0 = M_fit[1] # Get limit as fit goes to 0
    var_inst = M_0-M_lim0 # Get error variance from M11(0)-lim>0(M11)

    return var_inst

def prepData(df_in, dfIntensity_in, stat="var", var_type="beta", avetime=None,maxHgt = 2500, minHgt=100):
    # Prepare raw dataset for input into PBLh estimation algorithms
    # Applies QAQC, calculates variance where appropriate, reformats to be readable by functions
    df = pd.DataFrame(columns = df_in.columns, index=df_in.index, data = copy.deepcopy(df_in.values)) #necessary to prevent the original dataframe from being modified 
    dfIntensity = pd.DataFrame(columns = dfIntensity_in.columns, index=dfIntensity_in.index, data = copy.deepcopy(dfIntensity_in.values)) #necessary to prevent the original dataframe from being modified 
    
    df = qualitycontrol(df,dfIntensity,var_type=var_type)

    # Confining the dataset to data recorded within the height boundaries
    df.columns = df.columns.astype('int')
    df = df.drop(df.loc[:,(df.columns<minHgt) | (df.columns>maxHgt)].columns,axis=1)

    # Resampling the dataset to the averaging time specified (default 1h)
    if stat == 'var':
        df_cop = pd.DataFrame(columns = df_in.columns, index=df_in.index, data = copy.deepcopy(df_in.values)) #necessary to prevent the original dataframe from being modified 

        final_t = []
        final_var = []
        df_iter = df_cop.resample('5min') # Pandas' rolling window function has no custom step size for datetime yet, so we improvise to save time/avoid doing three times the computations
        for t,grp in df_iter:
            # Get rolling 30-min window
            grp = df.loc[(df.index>t-timedelta(minutes=15))&(df.index<t+timedelta(minutes=15))] 

            # Move on if we're at the edges of the dataset/there isn't enough data to compute variance
            if len(grp.index)<4: 
                continue

            # Filter outliers
            grp = drop_outliers(grp)

            # Calculate physical variance by subtracting instrument variance according to Lernschow et al. (2000)
            var_inst = grp.apply(instrument_variance,var_type=var_type)
            var_tot = grp.var()
            var_w = var_tot - var_inst

            final_t.append(t)
            final_var.append(var_w.to_frame().T)

        df = pd.concat(final_var)
        df.index = pd.Series(final_t).rename('time')

        # Filter negative values (physical/non-noisy variance can only be positive)
        negMask = df < 0 
        df = df.mask(negMask) 

    elif stat == 'mean':
        df = df.resample(avetime).mean()
    elif stat == 'std':
        df = df.resample(avetime).std()
    else:
        print("Incorrect input for stat, exiting.")
        sys.exit()

    df.columns.name = 'alt'
    df = df.stack().to_frame(name=var_type)

    return df


def getDaytime(df,lidar_loc):
    # Identify whether data points are during the day or night using sunrise/sunset times for each location 

    lidar_locs = {'UTSC':[43.7835, -79.1880, 128.0+24.0],'WEST':[43.0753, -81.3373, 280.0],'CYYZ':[43.6660, -79.6062, 163.0],\
              'MOEH':[43.7109, -79.5422, 134.0+10.0],'DOWN':[43.7814, -79.4685, 201.0],'KING':[43.9636, -79.5742, 360.0],\
                'CARE':[44.2328, -79.7812, 250.0],'CYOW':[45.3273, -75.6637, 117.0]}
    
    observer = Observer(latitude=lidar_locs[lidar_loc][0],longitude=lidar_locs[lidar_loc][1],elevation=lidar_locs[lidar_loc][2])
    df = df.tz_convert(ZoneInfo('America/Toronto'),level='time')

    sunrise = []
    sunset = []
    for i,xdate in enumerate(df.index.get_level_values('time')):
        try:
            sunrise.append(sun.sunrise(observer=observer,date=xdate.date(),tzinfo=ZoneInfo('America/Toronto'))+timedelta(hours=1))#.time())
        except ValueError:
            sunrise.append(sunrise[i-1])
            
        try:
            sunset.append(sun.sunset(observer=observer,date=xdate.date(),tzinfo=ZoneInfo('America/Toronto'))-timedelta(hours=1))#.time())
        except ValueError:
            sunset.append(sunset[i-1])
    df['sunrise'] = sunrise
    df['sunset'] = sunset
    df['daytime'] = (df.index.get_level_values('time').time >= df['sunrise'].dt.time) & (df.index.get_level_values('time').time < df['sunset'].dt.time)
    return df

def cloudHeight(df_in, dfIntensity, avetime="5min",stat="mean",cloud_threshold=-4.7,maxHgt=2000,minHgt=100):
    # Calculate cloud base height from aerosol backscatter coefficient
    df = prepData(df_in, dfIntensity, stat, 'beta', avetime,maxHgt=maxHgt,minHgt=minHgt)

    df = np.log10(df)

    df_grp = df.groupby(level='time')
    CBh_list = []
    for grptime,grp in df_grp:
        grp_threshold = grp.where(grp['beta'] > cloud_threshold, other=np.nan)['beta']
            
        # Get the height of the first value above the threshold scanning from the top down
        cbh = grp_threshold.last_valid_index()
        if cbh==None: #Take it to be nan if we don't find any values above the threshold
            cbh = (grptime,np.nan)

        CBh_list.append(cbh)
    
    return pd.DataFrame(CBh_list,columns=['time','cbh']).set_index('time')

def pblHeight(df_in, dfIntensity,method="beta",stat="var", avetime="5min", threshold = 0.04, 
               maxHgt = 2500, minHgt=100, lidar_loc='CYYZ'):
    
    if method == "beta":
        df = prepData(df_in, dfIntensity, stat, 'beta', maxHgt=maxHgt,minHgt=minHgt)
        df = getDaytime(df,lidar_loc)
        PBLh = pblhFromVariance(df,'beta',threshold=threshold)
    elif method == "threshold":
        df = prepData(df_in, dfIntensity, stat, 'wind', maxHgt=maxHgt,minHgt=minHgt)
        df = getDaytime(df,lidar_loc)
        PBLh = pblhFromVariance(df,'wind',threshold=threshold)
    elif method == "gradient":
        df = prepData(df_in, dfIntensity, stat, 'wind', maxHgt=maxHgt,minHgt=minHgt)
        df = getDaytime(df,lidar_loc)
        PBLh = pblhFromGrad(df,threshold=threshold)
    else:
        raise ValueError("Invalid input for variable 'method'. Valid inputs are: 'beta', 'threshold', and 'gradient'.")
    PBLh = PBLh.loc[PBLh.index.dropna()]
    return PBLh

def pblhFromVariance(df,var_type, threshold = 0.04):

    df_grp = df.groupby(level='time')
    PBLh_list = []
    for grptime,grp in df_grp:
        
        if var_type == 'wind':
            # Assume wind variance values above 5 are noise indicating loss of sensitivity and filter everything above
            largest = 5
            # Get only the values larger than largest and set rest to nan
            grp_largest = grp.where(grp['wind'] > largest,other=np.nan)['wind']
            # Find the first height at which values are larger than largest and set everything after to nan
            if grp_largest.first_valid_index() == None: #if none are larger than largest
                pass
            else:
                grp.loc[grp_largest.first_valid_index():,'wind'] = np.nan

            grp_threshold = grp.where(grp['wind'] < threshold, other=np.nan)['wind']
            # Set PBLh to be the height of the first value below the threshold scanning from the surface up
            pblht = grp_threshold.first_valid_index()

            if pblht==None or grp['wind'].dropna().iloc[0] < threshold: 
                # Take PBLh to be nan if we don't find any values above the threshold or if it's not turbulent at the surface
                pblht = (grptime,np.nan)

        elif var_type == 'beta':
            try:
                # Set PBLh as height of maximum variance in beta
                pblht = grp['beta'].idxmax()
            except ValueError:
                pblht = (grptime,np.nan)

        else:
            raise ValueError('Incorrect value for var_type. "wind" and "beta" are the only valid types.')

        PBLh_list.append(pblht)

    return pd.DataFrame(PBLh_list,columns=['time','pblh_'+var_type]).set_index('time')

def pblhFromGrad(df, threshold = 0.04):

    df_grp = df.groupby(level=0)
    PBLh = []
    for grptime,grp in df_grp:
        # Assume wind variance values above 5 are noise indicating loss of sensitivity and filter everything above
        largest = 5
        # Get only the values larger than largest and set rest to nan
        grp_largest = grp.where(grp['wind'] > largest,other=np.nan)['wind']
        # Find the first height at which values are larger than largest and set everything after to nan
        if grp_largest.first_valid_index() == None: #if none are larger than largest
            pass
        else:
            grp.loc[grp_largest.first_valid_index():,'wind'] = np.nan
        
        grp = grp.dropna()

        try:

            if len(grp)<=60: # Move on if there aren't enough datapoints to fit a smoothing window to
                PBLh.append((grptime,np.nan))
                continue

            elif grp['wind'].iloc[0]<threshold: # Move on if conditions aren't turbulent
                PBLh.append((grptime,np.nan))
                continue

            else:
                # Apply a smoothing Savitsky-Golay filter to wind speed variance before getting gradient
                grp.loc[:,'w_grd_smooth'] = np.gradient(savgol_filter(grp.loc[:,'wind'],60,polyorder=3), grp.index.get_level_values('alt'))
                
                # Find prominent minima with find_peaks
                peaks,peaks_properties = find_peaks(-grp['w_grd_smooth'],width=10,prominence=np.abs(0.1*grp['w_grd_smooth'].min()))

                pblht = (grptime,grp.index.get_level_values('alt')[peaks].min())

                if len(peaks) == 0: # Move on if no peaks are found 
                    PBLh.append((grptime,np.nan))
                    continue
        except np.linalg.LinAlgError: #Move on if the filter fit didn't converge
            PBLh.append((grptime,np.nan))
            continue

        PBLh.append(pblht)
        
    PBLh = pd.DataFrame(PBLh,columns=['time','pblh_grad']).set_index('time')

    return PBLh
