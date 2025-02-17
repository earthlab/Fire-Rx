"""
Classes wrapping http requests to LP DAAC servers for SRTMGL1 v003 elevation data and NASADEM_SC v001 slope and
curvature data.
"""

import getpass
import math
import os
import re
import shutil
import urllib
from argparse import Namespace
from http.cookiejar import CookieJar
from typing import Tuple, List
import zipfile

import numpy as np
import rasterio
from netCDF4 import Dataset
from osgeo import gdal
from osgeo import osr, ogr
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import richdem as rd


def _base_download_task(task_args: Namespace) -> None:
    """
    Downloads data from the LP DAAC servers. Authentication is established using NASA EarthData username and password.
    Args:
        task_args (Namespace): Contains attributes required for requesting data from LP DAAC
            task_args.link (str): URL to datafile on servers
            task_args.username (str): NASA EarthData username
            task_args.password (str): NASA EarthData password
            task_args.dest (str): Path to where the data will be written (a .zip file)
    """
    try:
        link = task_args.link
        print(link)
        pm = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        pm.add_password(None, "https://urs.earthdata.nasa.gov", task_args.username, task_args.password)
        cookie_jar = CookieJar()
        opener = urllib.request.build_opener(
            urllib.request.HTTPBasicAuthHandler(pm),
            urllib.request.HTTPCookieProcessor(cookie_jar)
        )
        urllib.request.install_opener(opener)
        myrequest = urllib.request.Request(link)
        response = urllib.request.urlopen(myrequest)
        response.begin()

        with open(task_args.dest, 'wb') as fd:
            while True:
                chunk = response.read()
                if chunk:
                    fd.write(chunk)
                else:
                    break

        # If the destination file is a ZIP file, extract it and then delete the ZIP file.
        if task_args.dest.lower().endswith('.zip'):
            extract_dir = os.path.dirname(task_args.dest)
            with zipfile.ZipFile(task_args.dest, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            os.remove(task_args.dest)
            print(f"Extracted and removed {task_args.dest}")

    except Exception as e:
        print(str(e))



# Functions for elevation parallel data processing


def _elevation_download_task(task_args: Namespace) -> None:
    """
    Downloads SRTMGL1 v003 slope data from LP DAAC servers. Files are downloaded as netCDF files and are thus
    converted to tif files so file structures are uniform throughout the api
    Args:
        task_args (Namespace): Contains attributes required for requesting data from EarthData servers
            task_args.link (str): URL to datafile on servers
            task_args.out_dir (str): Path to directory where files will be written
            task_args.username (str): NASA EarthData username
            task_args.password (str): NASA EarthData password
            task_args.top_left_coord (list): List giving coordinate of top left of image [lon, lat] so tif file can be
            geo-referenced
    """
    dest = os.path.join(task_args.out_dir, os.path.basename(task_args.link))
    task_args.dest = dest
    _base_download_task(task_args)

    # Sometimes the files will not exist, as in the case of a swath over water
    if not os.path.exists(task_args.dest):
        return


class BaseAPI:
    """
    Defines all the attributes and methods common to the child APIs. NASA EarthData credentials are queried for and
    written to secrets.yaml file if it is not found
    """

    BASE_URL = None

    def __init__(self, username, password):
        """
        Initializes the NASA EarthData credentials
        """
        self._username, self._password = username, password

    @staticmethod
    def get_utm_epsg(lat, lon) -> str:
        utm_zone = int((lon + 180) // 6) + 1
        epsg_code = 32600 if lat >= 0 else 32700
        epsg_code += utm_zone
        return 'EPSG:' + str(epsg_code)

    @staticmethod
    def _create_raster(output_path: str, columns: int, rows: int, n_band: int = 1,
                       gdal_data_type: int = gdal.GDT_Float32, driver: str = r'GTiff'):
        """
        Credit:
        https://gis.stackexchange.com/questions/290776/how-to-create-a-tiff-file-using-gdal-from-a-numpy-array-and-
        specifying-nodata-va

        Creates a blank raster for data to be written to
        Args:
            output_path (str): Path where the output tif file will be written to
            columns (int): Number of columns in raster
            rows (int): Number of rows in raster
            n_band (int): Number of bands in raster
            gdal_data_type (int): Data type for data written to raster
            driver (str): Driver for conversion
        """
        # create driver
        driver = gdal.GetDriverByName(driver)

        output_raster = driver.Create(output_path, columns, rows, n_band, eType=gdal_data_type)
        return output_raster

    @staticmethod
    def _numpy_array_to_raster(output_path: str, numpy_array: np.array, geo_transform,
                               projection, n_bands: int = 1, no_data: int = np.nan,
                               gdal_data_type: int = gdal.GDT_Float32):
        """
        Returns a gdal raster data source
        Args:
            output_path (str): Full path to the raster to be written to disk
            numpy_array (np.array): Numpy array containing data to write to raster
            geo_transform (gdal GeoTransform): tuple of six values that represent the top left corner coordinates, the
            pixel size in x and y directions, and the rotation of the image
            n_bands (int): The band to write to in the output raster
            no_data (int): Value in numpy array that should be treated as no data
            gdal_data_type (int): Gdal data type of raster (see gdal documentation for list of values)
        """
        rows, columns = numpy_array.shape[0], numpy_array.shape[1]

        # create output raster
        output_raster = BaseAPI._create_raster(output_path, int(columns), int(rows), n_bands, gdal_data_type)

        output_raster.SetProjection(projection)
        output_raster.SetGeoTransform(geo_transform)
        for i in range(n_bands):
            output_band = output_raster.GetRasterBand(i + 1)
            output_band.SetNoDataValue(no_data)
            output_band.WriteArray(numpy_array[:, :, i] if numpy_array.ndim == 3 else numpy_array)
            output_band.FlushCache()
            output_band.ComputeStatistics(False)

        if not os.path.exists(output_path):
            raise Exception('Failed to create raster: %s' % output_path)

        return output_path


class Elevation(BaseAPI):
    """
    SRTMGL1 v003 elevation data specific methods for requesting data from LP DAAC servers. Files are downloaded in
    parallel as netCDF files and then converted to tif format. Tif files are then made into a single tif mosaic covering
    the entire requested region.
    Data homepage: https://lpdaac.usgs.gov/products/srtmgl1v003/
    Ex.
        elevation = Elevation()
        elevation.download_district('data/elevation/ethiopia/yem.tif', 'Ethiopia', 'Yem')
        OR
        elevation.download_bbox('data/elevation/ethiopia/yem.tif', [37.391, 7.559, 37.616, 8.012])
    """
    BASE_URL = 'https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/'

    def __init__(self, username, password):
        super().__init__(username, password)

    @staticmethod
    def _get_auth_credentials() -> Tuple[str, str]:
        """
        Ask the user for their urs.earthdata.nasa.gov username and login if secrets.yaml file is not found in project
        root. After the first query, the secrets.yaml will be created for any subsequent initializations
        Returns:
            username (str): urs.earthdata.nasa.gov username
            password (str): urs.earthdata.nasa.gov password
        """
        username = input('Username:')
        password = getpass.getpass('Password:', stream=None)

        return username, password

    @staticmethod
    def _create_substrings(min_deg: int, max_deg: int, min_ord: str, max_ord: str, padding: int) -> List[str]:
        """
        The file conventions for the data files give information about the location of the data in each file. For
        example, an elevation file named like N00E003.SRTMGL1_NC.nc will be a 1x1 deg North pointing area with its
        bottom left pixel situated at N0 E003. Thus, given the min and max value and ordinal for either lat or lon
        this function will create all ranges of substrings in the range.
        Args:
            min_deg (int): The minimum value of the range in degrees
            max_deg (int): The maximum value of the range in degrees
            min_ord (str): The ordinal direction (n,e,s,w) of the minimum degree value
            max_ord (str): The ordinal direction (n,e,s,w) of the maximum degree value
            padding (int): The amount of zero padding to add to substring values. For lat padding is 2, 3 for lon
        Returns:
            substrings (List): List of substrings that are within range of [min_deg, max_deg]
        """
        substrings = []
        format_str = '{0:03d}' if padding == 3 else '{0:02d}'
        if min_ord == max_ord:
            abs_min = min(min_deg, max_deg)
            abs_max = max(min_deg, max_deg)
            deg_range = np.arange(abs_min, abs_max + 1, 1)
            for deg in deg_range:
                substrings.append(min_ord + format_str.format(deg))
        else:
            # Only other combo would be min_lon_ord is w and max_lon_ord is e
            neg_range = np.arange(1, min_deg + 1, 1)
            pos_range = np.arange(0, max_deg + 1, 1)
            for deg in neg_range:
                substrings.append(min_ord + format_str.format(deg))
            for deg in pos_range:
                substrings.append(max_ord + format_str.format(deg))

        return substrings

    @staticmethod
    def _mosaic_tif_files(input_dir: str, output_file: str) -> None:
        """
        Creates a mosaic from a group of tif files in the specified input_dir
        Args:
            input_dir (str): The path to the input dir containing the tif files to be mosaiced
            output_file (str): Path to where the output mosaic file will be written
        """
        # Specify the input directory containing the TIFF files
        tiff_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if
                      f.endswith('.hgt')]  # Open all the TIFF files using rasterio
        src = None
        src_files_to_mosaic = []
        for file in tiff_files:
            src = rasterio.open(file)
            src_files_to_mosaic.append(src)  # Merge the TIFF files using rasterio.merge
        mosaic, out_trans = merge(src_files_to_mosaic)  # Specify the output file path and name

        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans
                         })  # Write the merged TIFF file to disk using rasterio
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(mosaic)

    def download_bbox(self, bbox: List[float], out_dir:str,  buffer: float = 0) -> None:
        """
        Downloads data given a region and district. Bounding box will be read in from the region_info.yaml file in the
        root data directory. If the region or district is not found an error will be raised.
        Args:
            out_file (str): Path to where the output mosaic tif file of requested data will be written to
            bbox (list): Bounding box defining area of data request in [min_lon, min_lat, max_lon, max_lat]
            buffer (float): Buffer to add to the bounding box in meters
        """
        self._download_bbox(bbox, out_dir, buffer)

        # 2) Create a composite of all tiffs in the temp_dir
        self._mosaic_tif_files(out_dir, output_file='elevation.tif')

    def lat_lon_to_meters(self, input_tiff: str):
        input_tiff_file = gdal.Open(input_tiff, gdal.GA_Update)

        src_crs = osr.SpatialReference()
        src_crs.ImportFromEPSG(4326)  # Lat / lon

        geo_transform = input_tiff_file.GetGeoTransform()
        dst_epsg = self.get_utm_epsg(geo_transform[3], geo_transform[0])
        dst_crs = osr.SpatialReference()
        dst_crs.ImportFromEPSG(dst_epsg)

        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(geo_transform[3], geo_transform[0])
        transform = osr.CoordinateTransformation(src_crs, dst_crs)
        point.Transform(transform)
        new_geo_transform = [point.GetX(), 30.87, 0, point.GetY(), 0, -30.87]
        input_tiff_file.SetGeoTransform(new_geo_transform)
        input_tiff_file.SetProjection(dst_crs.ExportToWkt())
        input_tiff_file = None

    def _download_bbox(self, bbox: List[float], out_dir: str, buffer: float = 0) -> str:
        """
        Downloads data given a region and district. Bounding box will be read in from the region_info.yaml file in the
        root data directory. If the region or district is not found an error will be raised.
        Args:
            bbox (list): Bounding box defining area of data request in [min_lon, min_lat, max_lon, max_lat]
            buffer (float): Buffer to add to the bounding box in meters
        """

        # Convert the buffer from meters to degrees lat/long at the equator
        buffer /= 111000

        # Adjust the bounding box to include the buffer (subtract from min lat/long values, add to max lat/long values)
        bbox[0] -= buffer
        bbox[1] -= buffer
        bbox[2] += buffer
        bbox[3] += buffer

        # 1) Download all overlapping files and convert to tiff in parallel
        file_names = self._resolve_filenames(bbox)
        task_args = []
        for file_name in file_names:
            task_args.append(
                Namespace(
                    link=(os.path.join(self.BASE_URL, file_name)),
                    out_dir=out_dir,
                    top_left_coord=self._get_top_left_coordinate_from_filename(file_name.replace('.zip', '')),
                    username=self._username,
                    password=self._password
                )
            )
        for task in task_args:
            _elevation_download_task(task)



    @staticmethod
    def _get_top_left_coordinate_from_filename(infile: str) -> Tuple[float, float]:
        """
        Parses the specified infile name for coordinate info. File names indicate the bottom left pixel's coordinate,
        so need to add 1 deg (files are 1x1 deg and North-up) to lat to get top left corner.
        Args:
            infile (str): Name of path of file to get top left coordinate for
        Returns:
            top_left_coord (Tuple): Top left corner as (lon, lat)
        """
        infile = os.path.basename(infile)

        match = re.search(r'^([NS])(\d{2})([EW])(\d{3})\.SRTMGL1\.hgt$', infile, re.IGNORECASE)
        if match:
            n_or_s = match.group(0)[0]
            e_or_w = match.group(0)[3]
            n_value = match.group(1)
            e_value = match.group(2)

        lat = float(n_value) * (1 if n_or_s.lower() == 'n' else -1) + 1
        lon = float(e_value) * (1 if e_or_w.lower() == 'e' else -1)

        return lon, lat

    def _resolve_filenames(self, bbox: List[float]) -> List[str]:
        """
        Files are stored with the following naming convention, for example: N00E003.SRTMGL1_NC.nc where N00 is 0 deg
        latitude and E003 is 3 deg longitude, referring to the lower left coordinate of the data file. So create all
        the possible file name combinations within the range of the bounding box
        Args:
            bbox (list): Bounding box coordinates in [min_lon, min_lat, max_lon, max_lat]
        Returns:
            file_names (list): List of SRTMGL1_NC formatted file names within the bbox range
        """
        min_lon = bbox[0]
        min_lat = bbox[1]
        max_lon = bbox[2]
        max_lat = bbox[3]

        # First find the longitude range
        if min_lon < 0:
            min_lon_ord = 'w'
            round_min_lon = math.ceil(abs(min_lon))
        else:
            min_lon_ord = 'e'
            round_min_lon = math.floor(min_lon)

        if max_lon < 0:
            round_max_lon = math.ceil(abs(max_lon))
            max_lon_ord = 'w' if round_max_lon != 0 else 'e'
        else:
            max_lon_ord = 'e'
            round_max_lon = math.floor(max_lon)

        # Next latitude range
        if min_lat < 0:
            min_lat_ord = 's'
            round_min_lat = math.ceil(abs(min_lat))
        else:
            min_lat_ord = 'n'
            round_min_lat = math.floor(min_lat)

        if max_lat < 0:
            round_max_lat = math.ceil(abs(max_lat))
            max_lat_ord = 's' if round_max_lat != 0 else 'n'
        else:
            max_lat_ord = 'n'
            round_max_lat = math.floor(max_lat)

        lon_substrings = self._create_substrings(round_min_lon, round_max_lon, min_lon_ord, max_lon_ord, 3)
        lat_substrings = self._create_substrings(round_min_lat, round_max_lat, min_lat_ord, max_lat_ord, 2)

        file_names = []
        for lon in lon_substrings:
            for lat in lat_substrings:
                file_names.append(f"{lat.upper()}{lon.upper()}.SRTMGL1.hgt.zip")

        return file_names

    def reproject_to_meters(self, input_tif, output_tif):
        """
        Reproject a raster file to a specified CRS (in meters).

        :param input_tif: Path to the input TIFF file (in lat/lon).
        :param output_tif: Path to the output TIFF file (in meters).
        """
        with rasterio.open(input_tif) as src:
            # Instead of using the filename, get the top left coordinate from the geotransform.
            # This returns a tuple (x, y) for the coordinate of pixel (0,0).
            lon, lat = src.transform * (0, 0)
            print(lon, lat)

            utm_epsg = self.get_utm_epsg(lat, lon)
            transform, width, height = calculate_default_transform(
                src.crs, utm_epsg, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'driver': 'GTiff',
                'crs': utm_epsg,
                'transform': transform,
                'width': width,
                'height': height
            })
            print(kwargs)

            with rasterio.open(output_tif, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=utm_epsg,
                        resampling=Resampling.nearest)

    def reproject_to_latlon(self, input_tif, output_tif, target_crs):
        """
        Reproject a raster file to latitude/longitude (EPSG:4326).

        :param input_tif: Path to the input TIFF file (in meters).
        :param output_tif: Path to the output TIFF file (in lat/lon).
        :param source_crs: The source CRS.
        :param target_crs: The target CRS (lat/lon).
        """
        with rasterio.open(input_tif) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(output_tif, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest)

    def rd_derive_slope_from_elevation(self, elevation_file, slope_outfile):
        """
        A Slope calculation (degrees)
        C Horn, B.K.P., 1981. Hill shading and the reflectance map. Proceedings of the IEEE 69, 14–47. doi:10.1109/PROC.1981.11918
        :param elevation_file:
        :param slope_outfile:
        :param attrib:
        :return:
        """
        _ = self.reproject_to_meters(elevation_file, 'temp1.tif')
        f = rd.LoadGDAL('temp1.tif')
        slope = rd.TerrainAttribute(f, attrib='slope_riserun')

        meters_dataset = gdal.Open('temp1.tif')
        geo_transform = meters_dataset.GetGeoTransform()
        projection = meters_dataset.GetProjection()

        with rasterio.open(elevation_file) as f:
            target_crs = f.crs

        self._numpy_array_to_raster('temp2.tif', slope, geo_transform, projection, no_data=-9999)
        self.reproject_to_latlon('temp2.tif', slope_outfile, target_crs)
        os.remove('temp1.tif')
        os.remove('temp2.tif')

    def rd_derive_aspect_from_elevation(self, elevation_file, aspect_outfile):
        """
        A Aspect attribute calculation
        C Horn, B.K.P., 1981. Hill shading and the reflectance map. Proceedings of the IEEE 69, 14–47. doi:10.1109/PROC.1981.11918
        :param elevation_file:
        :param aspect_outfile:
        :param attrib:
        :return:
        """
        _ = self.reproject_to_meters(elevation_file, 'temp1.tif')
        f = rd.LoadGDAL('temp1.tif')
        slope = rd.TerrainAttribute(f, attrib='aspect')

        meters_dataset = gdal.Open('temp1.tif')
        geo_transform = meters_dataset.GetGeoTransform()
        projection = meters_dataset.GetProjection()

        with rasterio.open(elevation_file) as f:
            target_crs = f.crs

        self._numpy_array_to_raster('temp2.tif', slope, geo_transform, projection, no_data=-9999)
        self.reproject_to_latlon('temp2.tif', aspect_outfile, target_crs)
        os.remove('temp1.tif')
        os.remove('temp2.tif')

    def merge_tif_files(self, input_directory, output_tif):
        """
        Merge multiple TIFF files into a single TIFF file.
res = 0
        stack = []
        for i, n in enumerate(arr):
            while stack and stack[-1][0] > n:
                v, j = stack.pop()
                res += (v * (i-j)) + n
            stack.append((n, i))

        while len(stack) > 1:
            v, i = stack.pop()
            res += v * (len(arr) - i)

        return res
        :param input_directory: Directory containing the input TIFF files.
        :param output_tif: Path to the output merged TIFF file.
        """
        # List to store opened datasets
        src_files_to_mosaic = []

        # Iterate over all files in the directory
        for file in os.listdir(input_directory):
            src_path = os.path.join(input_directory, file)
            src = rasterio.open(src_path)
            src_files_to_mosaic.append(src)

        # Merge function
        mosaic, out_trans = rasterio.merge.merge(src_files_to_mosaic)

        # Copy metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })

        # Write the mosaic raster to disk
        with rasterio.open(output_tif, "w", **out_meta) as dest:
            dest.write(mosaic)

        # Close all source files
        for src in src_files_to_mosaic:
            src.close()
