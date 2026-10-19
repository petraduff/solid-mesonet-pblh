## PBLh estimation for the SOLID Mesonet

### Usage
The code here estimates planetary boundary layer height (PBLh) given input .csv files of vertical wind speed and aerosol backscatter coefficient. These files are read in as Pandas dataframes for each day, with altitude on the vertical axis (index) and time on the horizontal axis (columns). Example input files are included for a case study of February 19th - 20th, 2024, at the Pearson Airport (CYYZ) Southern Ontario LIDar (SOLID) Mesonet site. 

Estimates are run using the code `handler.py`, as follows:

```
lidarLoc inPath [startDate] [endDate]
```

for the 4-letter site code `lidarLoc` (e.g. CYYZ for Pearson Airport) and the path pointing to the input data `inPath`. The arguments `startDate` and `endDate` are optional but default to the case study dates. Pass `python handler.py -h` for more options. Output files are saved in a new folder labeled "processed" in the input directory.

### Output

PBLh is estimated via three estimation algorithms, two using vertical wind speeds and one aerosol backscatter coefficients. They are saved in the final files as:

*PBL height doppler variance threshold method (m)*
Calculates the vertical wind speed variance in time and sets the PBLh at each 5-minute time step to be the point at which the rolling 30-minute variance drops below a set threshold (default: 0.04 m $^2$/s $^2$) when scanning up from the surface.

*PBL height doppler variance gradient method (m)*
Similarly finds vertical wind speed variance in time, but then sets the PBLh to be the minimum gradient of the variance. Minima are found using the `scipy.find_peaks()` method. 

*PBLh backscatter method from variance (m)*
Finds the variance in time of the aerosol backscatter coefficient and sets the PBLh to be the maximal variance in height (i.e. the entrainment zone). 

More details on the PBLh estimation performed can be found in Duff, et al. (2026; DOI pending). 

### Site information 


| Site | Lat (◦) | Lon (◦) | Altitude (m a.s.l.) | Altitude (m AGL) | Deployment date |Region type |
|------|----------|-----------------|-----------------|-----------------|--------|------------|
|CYOW | 45.3273 | -75.6637 | 112 | 0 |  September 7, 2024 | Airport hangars |
|UTSC | 43.7835 | -79.1880 | 152 | 21 | October 20, 2023 | Suburban |
| MECP (MOEH) | 43.7109 | -79.5422 | 141 | 10 | November 6, 2023 | Urban, highways |
| CYYZ | 43.6660 | -79.6062 | 163 | 1 | November 15 2022 | Airport runways |
|WEST | 43.0753 | -81.3373 | 280 | 2 | October 17, 2023 | Farmland |


### Other
Note: when running this code, you may encounter the following error:
```
Intel oneMKL ERROR: Parameter 6 was incorrect on entry to DGELSD.
```
which doesn't cause the program to fail, but pops up with the Savitsky-Golay filter step in the wind variance gradient method and can be annoying. It occurs due to incompatabilities between Python packages, and I've been able to stop it by running the following command in my Anaconda environment:
```
conda install nomkl numpy scipy
```
which installs dependencies that don't rely on Intel oneMKL.

