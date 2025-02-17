# Fire-Rx
NASA Rx-Fire: Prescribed Fire to Support Ecosystem Services (22-SMDSS22-0104)

# GEDI aboveground biomass density data

# ECOSTRESS WUE data
ECOSTRESS data is acquired by sending requests to the ECOSTRESS USGS servers at
https://e4ftl01.cr.usgs.gov/ECOSTRESS/. The script enables both temporal and spatial filtering. Temporal filtering is
achieved by specifying a year, a range of months, and the desired hours in a day. Each file’s acquisition date is 
extracted from its name, which contains a UTC timestamp; this timestamp is then converted to local time based on the 
time zone at the center of the raster.  Spatial filtering is implemented by defining a bounding box in the form 
[min_lon, min_lat, max_lon, max_lat]. Before downloading a data file, the script retrieves its corresponding metadata
file, which includes a bounding polygon delineating the raster’s spatial extent. If this polygon intersects the
user-specified bounding box, the file is selected for download.

To produce a georeferenced Water Use Efficiency (WUE) file, the ECO4WUE.001 data is combined with its corresponding
ECO1BGEO.001 file using a modified version of NASA’s ECOSTRESS_swath2grid.py script (available in our repository).
The matching of WUE and GEO files is performed by parsing the orbit, scene ID, and precise timestamp
(year, month, day, hour, minute, second) from their file names—excluding the build ID and version, which can differ
between corresponding files.

The process of creating a WUE composite file is split into 3 phases: 
1) Download the WUE and GEO files based on temporal / spatial filter and rasterize the WUE files
2) Create a time series of the WUE rasters and then split it chunks
3) Calculate the median of pixel in the time series chunk and then stitch all the chunks back together

This can be done with one python executable: bin/create_wue_composite.py

Example usage
First set FIRE_RX_USERNAME and FIRE_RX_PASSWORD ENV vars to your NASA Earthdata username and password. Then from the
command line run, for example:
```bash
python bin/create_wue_composite.py --years 2020 2021 --month_start 5 --month_end 7 --hour_start 10 --hour_end 14 --bbox -119.0 39.0 -117.0 41.0
```

This downloads all WUE and GEO files in 2020 and 2021, from May - July during the 10th - 14th hours of the day and
which overlap the specified bounding box. The median of the time series is then calculated and finally output to a raster.

Progress is saved when finding overlapping files in case the process needs to be restarted.

# SRTM elevation data

## slope and aspect data

# Forest age data

# National Land cover data

Future Plans:
    Objectives and plans for future: Put more focus on publication (user portal) and 
    working on large data sets / problems (fire-rx, bioextremes) 
    
    Delivering 3 publications:
    Fire-Rx (Jerry) August
    FIREDPY NRT September
    BioExtremes October
    
    Then more time on ESIIL to deliver the portal -> 1st author paper
    
    Potentially participate in proposal writing later this year

Presentations:
    Add BioExtremes presentation from May

Software Development:
    User portal (Add collaborators)
    OASIS (Add collaborators)

Publications:
    Abstracts (BioExtremes)
