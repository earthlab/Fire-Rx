# NASA Rx-Fire Ready-to-Use Data
This repository contains codes to collect, process, and create ready-to-use Western US layers from NASA assets. These layers include ECOSTRESS Water Use Efficiency, SRTM Elevation, Slope and Aspect, GEDI aboveground biomass, and Forest Age from Pan et al. (2012). 

This repository is a product of the project "NASA Rx-Fire: Prescribed Fire to Support Ecosystem Services" (Award Number 80NSSC23K0397).

## ECOSTRESS WUE data
ECOSTRESS data is acquired by sending requests to the ECOSTRESS USGS servers at
https://e4ftl01.cr.usgs.gov/ECOSTRESS/. The script enables both temporal and spatial filtering. Temporal filtering is
achieved by specifying a year, a range of months, and the desired daily hours. Each file’s acquisition date is 
extracted from its name, which contains a UTC timestamp; this timestamp is then converted to local time based on the 
time zone at the center of the raster.  Spatial filtering is implemented by defining a bounding box  
[min_lon, min_lat, max_lon, max_lat]. Before downloading a data file, the script retrieves its corresponding metadata
file, which includes a bounding polygon delineating the raster’s spatial extent. If this polygon intersects the
user-specified bounding box, the file is selected for download.

To produce a georeferenced Water Use Efficiency (WUE) file, the ECO4WUE.001 data is combined with its corresponding
ECO1BGEO.001 file using a modified version of NASA’s ECOSTRESS_swath2grid.py script (available in our repository).
The matching of WUE and GEO files is performed by parsing the orbit, scene ID, and precise timestamp
(year, month, day, hour, minute, second) from their file names—excluding the build ID and version, which can differ
between corresponding files.

The process of creating a WUE composite file is split into 3 phases: 
1) Download the WUE and GEO files based on temporal/spatial filter and rasterize the WUE files
2) Create a time series of the WUE rasters and then split it into chunks
3) Calculate the median of a pixel in the time series chunk and then stitch all the chunks back together

This can be done with one Python executable: bin/create_wue_composite.py

Example usage
First, set FIRE_RX_USERNAME and FIRE_RX_PASSWORD ENV vars to your NASA Earthdata username and password. Then from the
command line run, for example:
```bash
python bin/create_wue_composite.py --years 2020 2021 --month_start 5 --month_end 7 --hour_start 10 --hour_end 14 --bbox -119.0 39.0 -117.0 41.0
```

This downloads all WUE and GEO files in 2020 and 2021, from May to July, during the 10th to 14th hours of the day, 
which overlap the specified bounding box. The median of the time series is then calculated and output to a raster.

Progress is saved when finding overlapping files in case the process needs to be restarted.

## SRTM elevation data
The SRTM data can be downloaded and filtered by a bounding box. You can use the
bin/create_elevation_slope_aspect_composites.py to create an elevation, slope, and aspect file
```bash
python bin/create_elevation_slope_aspect_composites.py --bbox -105 40 -104 41
```
## GEDI aboveground biomass density data
We used Global Ecosystem Dynamics Investigation (GEDI) footprint-based aboveground biomass density data. Aboveground biomass density estimates have been derived by applying a plant functional-specific model to GEDI-based relative height metrics, and the units of the data are at megagrams per hectare (Mg/ha). Please refer to the full Python notebook on how to get the footprint-based biomass data within a user-defined polygon. The GEDI biomass python code here was adapted from the notebook provided by (https://github.com/ornldaac/gedi_tutorials/blob/main/access_gedi_l4a_hyrax.ipynb).

## Forest age data

This project used forest age data from NACP Forest Age Maps at 1-km Resolution for Canada (2004) and the U.S.A. (2006).(https://daac.ornl.gov/NACP/guides/NA_Tree_Age.html). This data set contains forest age at 1-km resolution for Canada and the United States (U.S.A.). This data set has been derived by compiling forest inventory data, historical fire data, optical satellite data, and images from NASA’s Landsat Ecosystem Disturbance Adaptive Processing System (LEDAPS) project. 
Please refer the reference below for the full description of the forest age data set.
Pan, Y., Chen, J. M., Birdsey, R., McCullough, K., He, L., and Deng, F.: Age structure and disturbance legacy of North American forests, Biogeosciences Discuss., 7, 979-1020, doi:10.5194/bgd-7-979-2010, 2010.  [ © Author(s) 2011. CC Attribution 3.0 License.] http://daac.ornl.gov/daacdata/nacp/NA_TreeAge/comp/Pan_et_al_bg-8-715-2011.pdf



