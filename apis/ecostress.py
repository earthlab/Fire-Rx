import collections
import getpass
import json
import math
import os
import re
import shutil
import sys
import urllib
import warnings
from argparse import Namespace
from http.cookiejar import CookieJar
from multiprocessing import Pool
from multiprocessing import set_start_method
from typing import Tuple, List, Dict
from datetime import datetime, timedelta, time
import calendar
import subprocess as sp
import h5py
from apis.ecostress_conv.ECOSTRESS_swath2grid import main
from glob import glob
import xml.etree.ElementTree as ET
import multiprocessing as mp
import time

import bs4
import certifi
import requests
import urllib3.util
from osgeo import gdal
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon
import rasterio
from rasterio.merge import merge
import pytz
from timezonefinder import TimezoneFinder
import pandas as pd

# -109.138184,36.923548,-101.942139,41.071069
#l.download_composite(2021, 6, 8, 13, 17, 'test.tif', [-124.980469, 28.767659, -103.359375, 49.382373], esi=False)
# l.download_composite(2021, 6, 8, 13, 17, 'test.tif', [-125.0, 39.0, -117.0, 44.0])

class BaseAPI:
    """
    Defines all the attributes and methods common to the child APIs.
    """
    PROJ_DIR = os.path.dirname(os.path.dirname(__file__))
    _BASE_WUE_URL = 'https://e4ftl01.cr.usgs.gov/ECOSTRESS/ECO4WUE.001/'
    _BASE_GEO_URL = 'https://e4ftl01.cr.usgs.gov/ECOSTRESS/ECO1BGEO.001/'
    _BASE_CLOUD_URL = 'https://e4ftl01.cr.usgs.gov/ECOSTRESS/ECO2CLD.001/'
    _BASE_ESI_URL = 'https://e4ftl01.cr.usgs.gov/ECOSTRESS/ECO4ESIPTJPL.001/'

    _XML_DIR = os.path.join(PROJ_DIR, 'xml_files')

    def __init__(self, username: str = None, password: str = None, lazy: bool = False):
        """`
        Initializes the common attributes required for each data type's API
        """
        self._username = os.environ.get('FIRE_RX_USER', username)
        self._password = os.environ.get('FIRE_RX_PASS', password)
        self._core_count = os.cpu_count()
        if not lazy:
            self._configure()
        self._file_re = None
        self._tif_re = None

        os.makedirs(self._XML_DIR, exist_ok=True)

    @staticmethod
    def retrieve_links(url: str, suffix: str) -> List[str]:
        """
        Creates a list of all the links found on a webpage
        Args:
            url (str): The URL of the webpage for which you would like a list of links

        Returns:
            (list): All the links on the input URL's webpage
        """
        request = requests.get(url)
        soup = bs4.BeautifulSoup(request.text, 'html.parser')
        return [link.get('href') for link in soup.find_all('a') if
                isinstance(link.get('href'), str) and link.get('href').endswith(suffix)]

    @staticmethod
    def _cred_query() -> Tuple[str, str]:
        """
        Ask the user for their urs.earthdata.nasa.gov username and login
        Returns:
            username (str): urs.earthdata.nasa.gov username
            password (str): urs.earthdata.nasa.gov password
        """
        print('Please input your earthdata.nasa.gov username and password. If you do not have one, you can register'
              ' here: https://urs.earthdata.nasa.gov/users/new')
        username = input('Username:')
        password = getpass.getpass('Password:', stream=None)

        return username, password

    def _configure(self) -> None:
        """
        Queries the user for credentials and configures SSL certificates
        """
        if self._username is None or self._password is None:
            username, password = self._cred_query()

            self._username = username
            self._password = password

        # This is a macOS thing... need to find path to SSL certificates and set the following environment variables
        ssl_cert_path = certifi.where()
        if 'SSL_CERT_FILE' not in os.environ or os.environ['SSL_CERT_FILE'] != ssl_cert_path:
            os.environ['SSL_CERT_FILE'] = ssl_cert_path

        if 'REQUESTS_CA_BUNDLE' not in os.environ or os.environ['REQUESTS_CA_BUNDLE'] != ssl_cert_path:
            os.environ['REQUESTS_CA_BUNDLE'] = ssl_cert_path

    def _download(self, query: Tuple[str, str], retry: int = 0) -> None:
        """
        Downloads data from the NASA earthdata servers. Authentication is established using the username and password
        found in the local ~/.netrc file.
        Args:
            query (tuple): Contains the remote location and the local path destination, respectively
        """
        link = query[0]
        dest = query[1]
        if os.path.exists(dest):
            print(f'Skipping {dest}')
            return

        pm = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        pm.add_password(None, "https://urs.earthdata.nasa.gov", self._username, self._password)
        cookie_jar = CookieJar()
        opener = urllib.request.build_opener(
            urllib.request.HTTPBasicAuthHandler(pm),
            urllib.request.HTTPCookieProcessor(cookie_jar)
        )
        urllib.request.install_opener(opener)
        myrequest = urllib.request.Request(link)
        response = urllib.request.urlopen(myrequest)
        response.begin()
        with open(dest, 'wb') as fd:
            while True:
                chunk = response.read()
                if chunk:
                    fd.write(chunk)
                else:
                    break

        if not self._verify_hdf_file(dest):
            os.remove(dest)
            if retry < 1:
                self._download(query, retry=1)

    def download_time_series(self, queries: List[Tuple[str, str]], outdir: str):
        """
        Attempts to create download requests for each query, if that fails then makes each request in series.
        Args:
            queries (list): List of tuples containing the remote and local locations for each request
        Returns:
            outdir (str): Path to the output file directory
        """
        # From earthlab firedpy package
        if len(queries) > 0:
            print("Retrieving data... skipping over any cached files")

            with Pool(int(self._core_count / 2)) as pool:
                for _ in tqdm(pool.imap_unordered(self._download, queries), total=len(queries)):
                    pass

        print(f'Wrote {len(queries)} files to {outdir}')

    @staticmethod
    def _verify_hdf_file(file_path: str) -> bool:
        try:
            h5py.File(file_path)
            return True
        except OSError:
            return False

    def _parse_bbox_from_xml(self, xml_url: str) -> Polygon:
        filename = os.path.basename(xml_url)
        file_path = os.path.join(self._XML_DIR, filename)

        if not os.path.exists(file_path):
            # Send a GET request with HTTP Basic Authentication
            pm = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            pm.add_password(None, "https://urs.earthdata.nasa.gov", self._username, self._password)
            cookie_jar = CookieJar()
            opener = urllib.request.build_opener(
                urllib.request.HTTPBasicAuthHandler(pm),
                urllib.request.HTTPCookieProcessor(cookie_jar)
            )
            urllib.request.install_opener(opener)
            myrequest = urllib.request.Request(xml_url)
            with urllib.request.urlopen(myrequest) as response:
                xml_content = response.read()

                # Write XML content to file
                with open(file_path, 'wb') as f:
                    f.write(xml_content)

        # Parse the XML content
        root = ET.fromstring(open(file_path, 'rb').read())
        bounding_rect = root.find('.//BoundingRectangle')
        west = float(bounding_rect.find('WestBoundingCoordinate').text)
        north = float(bounding_rect.find('NorthBoundingCoordinate').text)
        east = float(bounding_rect.find('EastBoundingCoordinate').text)
        south = float(bounding_rect.find('SouthBoundingCoordinate').text)
        return Polygon([(west, north), (east, north), (east, south), (west, south)])

    def _spatio_temporal_overlap(self, target_bbox: List[float], hour_start: int, hour_end: int,
                                 file_time_utc: datetime, xml_url: str):
        """
        The file dates included in the file name are in UTC, and must be converted to their respective time zones before
        seeing if they overlap the hour of day range input by the user. The corresponding XML file for each h5 file is
        thus downloaded to determine the time zone and if the file overlaps the target bbox.
        :param target_bbox [float, float, float, float]: Bounding box in min_lon, min_lat, max_lon, max_lat for
                downloading files within
        :param hour_start: Time of day in local time that file must start after to be downloaded
        :param hour_end: Time of day in local time that file must start before to be downloaded
        :param xml_url: URL for the h5 file's corresponding XML file which contains geolocation metadata
        :return: bool
        """
        # Now apply spatial filter by downloading the xml files and checking if they overlap the bounding box
        min_lon, min_lat, max_lon, max_lat = target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3]
        target_bbox = Polygon([(min_lon, max_lat), (max_lon, max_lat), (max_lon, min_lat), (min_lon, min_lat)])
        try:
            file_bbox = self._parse_bbox_from_xml(xml_url)

            centroid = file_bbox.centroid
            center_lon, center_lat = centroid.x, centroid.y

            tf = TimezoneFinder()
            timezone_str = tf.timezone_at(lng=center_lon, lat=center_lat)
            if timezone_str is None:
                raise ValueError("Could not determine the time zone from the given coordinates.")

            local_timezone = pytz.timezone(timezone_str)
            file_time_local = file_time_utc.astimezone(local_timezone)

            return hour_start <= file_time_local.hour <= hour_end and file_bbox.intersects(target_bbox)
        except:
            return False

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


class L4(BaseAPI):

    def __init__(self, username: str = None, password: str = None, lazy: bool = False):
        super().__init__(username=username, password=password, lazy=lazy)
        common_regex = r'\_(?P<orbit>\d{5})\_(?P<scene_id>\d{3})\_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})T(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})\_(?P<build_id>\d{4})\_(?P<version>\d{2})\.h5$'
        self._wue_file_re = r'ECOSTRESS\_L4\_WUE' + common_regex
        self._bgeo_file_re = r'ECOSTRESS\_L1B\_GEO' + common_regex
        self._cloud_file_re = r'ECOSTRESS\_L2\_CLOUD' + common_regex
        self._esi_file_re = r'ECOSTRESS\_L4\_ESI\_PT-JPL' + common_regex
        self._cloud_file_tif_re = r'ECOSTRESS\_L2\_CLOUD' + common_regex.replace('.h5', '_CloudMask_GEO.tif')
        self._wue_tif_re = r'ECOSTRESS\_L4\_WUE' + common_regex.replace('.h5', '_WUEavg_GEO.tif')
        self._esi_tif_re = r'ECOSTRESS\_L4\_ESI\_PT-JPL' + common_regex.replace('.h5', 'ESI\_PT-JPLavg_GEO.tif')
        self._db_re = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2}) (?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})\_(?P<min_lon>\-?\d+\.\d+)\_(?P<max_lon>\-?\d+\.\d+)\_(?P<min_lat>\-?\d+\.\d+)\_(?P<max_lat>\-?\d+\.\d+)\_(?P<lon_res>\-?\d+\.\d+)\_(?P<lat_res>\-?\d+\.\d+)\.db$'
        self._res = 0.0006298419
        self._projection = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'

    @staticmethod
    def _get_last_day_of_month(year, month):
        # monthrange returns a tuple (weekday of first day of the month, number of days in month)
        _, num_days = calendar.monthrange(year, month)
        return num_days

    @staticmethod
    def _generate_file_key(file_group_dict: Dict[str, str]) -> tuple:
        return (file_group_dict['orbit'], file_group_dict['scene_id'], file_group_dict['year'],
                file_group_dict['month'], file_group_dict['day'], file_group_dict['hour'], file_group_dict['minute'],
                file_group_dict['second'])

    def _worker(self, args):
        day_url, day_file, bbox, hour_start, hour_end = args
        match = re.match(self._wue_file_re, day_file)
        if match:
            groups = match.groupdict()
            file_time_utc = datetime(int(groups['year']), int(groups['month']), int(groups['day']),
                                     int(groups['hour']), int(groups['minute']), int(groups['second']),
                                     tzinfo=pytz.UTC)
            url = urllib.parse.urljoin(day_url, day_file)

            return os.path.basename(day_url[:-1]), url, self._spatio_temporal_overlap(bbox, hour_start, hour_end,
                                                                                      file_time_utc, url + '.xml')

        return None

    def _find_matching_urls(self, base_url: str, date: str, links: List[str], file_re: str) -> Dict[tuple, str]:
        file_dict = {}
        for link in links:
            match = re.match(file_re, link)
            if match:
                file_group_dict = match.groupdict()
                key = self._generate_file_key(file_group_dict)
                file_dict[key] = urllib.parse.urljoin(base_url + '/' + date + '/', link)

        return file_dict

    def gather_file_links(self, years: List[int], month_start: int, month_end: int, hour_start: int,
                          hour_end: int, bbox: List[int], url_progress_file_path: str) -> List[Tuple[str, str, str]]:
        day_urls = []
        for year in years:
            start_date = datetime(year, month_start, 1)

            # Add one extra day to end date to account for UTC conversion
            end_date = datetime(year, month_end, self._get_last_day_of_month(year, month_end)) + timedelta(days=1)

            while start_date <= end_date:
                day_urls.append(urllib.parse.urljoin(self._BASE_WUE_URL, start_date.strftime('%Y.%m.%d') + '/'))
                start_date += timedelta(days=1)

        progress_df = pd.read_csv(url_progress_file_path)

        # Prepare arguments for multiprocessing
        args = []
        for day_url in day_urls:
            day_files = self.retrieve_links(day_url, '.h5')
            for day_file in day_files:
                if urllib.parse.urljoin(day_url, day_file) not in progress_df['wue_url'].values:
                    args.append((day_url, day_file, bbox, hour_start, hour_end))

        print('Finding overlapping files')
        save_interval = 500
        i = 0
        with Pool(mp.cpu_count() - 1) as pool:
            # Using tqdm to create a progress bar
            for result in tqdm(pool.imap_unordered(self._worker, args), total=len(args),
                               desc='Finding overlapping files'):
                if result:
                    day, url, accepted = result
                    new_row = {
                        'wue_url': url,
                        'esi_url': None,
                        'geo_url': None,
                        'cloud_url': None,
                        'day': day,
                        'accepted': accepted
                    }
                    # geo_lookup[day].append(url)
                    progress_df.loc[len(progress_df)] = new_row
                    if i > 0 and i % save_interval == 0:
                        progress_df.to_csv(url_progress_file_path, index=False)
                    i += 1

        progress_df.to_csv(url_progress_file_path, index=False)
        progress_df = pd.read_csv(url_progress_file_path)
        geo_lookup = collections.defaultdict(list)
        for _, row in progress_df.iterrows():
            if ((not isinstance(row['geo_url'], str) or not isinstance(row['cloud_url'], str) or not
                 isinstance(row['esi_url'], str)) and row['accepted']):
                geo_lookup[row['day']].append(row['wue_url'])

        # Now find the GEO and CLOUD urls. The versions are not always the same (this doesn't matter for the swath2grid function)
        # so you cannot infer the GEO url from the WUE url. What should match is the orbit, scene_id, and date.
        i = 0
        for date, wue_file_links in geo_lookup.items():
            geo_date_url = urllib.parse.urljoin(self._BASE_GEO_URL, date)
            cloud_date_url = urllib.parse.urljoin(self._BASE_CLOUD_URL, date)
            esi_date_url = urllib.parse.urljoin(self._BASE_ESI_URL, date)
            geo_links = self.retrieve_links(geo_date_url, '.h5')
            cloud_links = self.retrieve_links(cloud_date_url, '.h5')
            esi_links = self.retrieve_links(esi_date_url, '.h5')

            geo_file_dict = self._find_matching_urls(self._BASE_GEO_URL, date, geo_links, self._bgeo_file_re)
            cloud_file_dict = self._find_matching_urls(self._BASE_CLOUD_URL, date, cloud_links, self._cloud_file_re)
            esi_file_dict = self._find_matching_urls(self._BASE_ESI_URL, date, esi_links, self._esi_file_re)

            for wue_file_link in wue_file_links:
                group_dict = re.match(self._wue_file_re, os.path.basename(wue_file_link)).groupdict()
                key = self._generate_file_key(group_dict)
                if key in geo_file_dict and key in cloud_file_dict and key in esi_file_dict:
                    progress_df.loc[progress_df['wue_url'] == wue_file_link, 'geo_url'] = geo_file_dict[key]
                    progress_df.loc[progress_df['wue_url'] == wue_file_link, 'cloud_url'] = cloud_file_dict[key]
                    progress_df.loc[progress_df['wue_url'] == wue_file_link, 'esi_url'] = esi_file_dict[key]

                    if i > 0 and i % save_interval == 0:
                        progress_df.to_csv(url_progress_file_path, index=False)
                    i += 1

        progress_df.to_csv(url_progress_file_path, index=False)

        accepted_df = progress_df[progress_df['accepted'] == True]

        return [tuple(row) for row in accepted_df[['wue_url', 'esi_url', 'cloud_url', 'geo_url']].values]

    @staticmethod
    def _create_cloud_mask(cloud_data: np.array) -> np.array:
        mask = np.zeros_like(cloud_data)
        for row_idx, col_idx in np.ndindex(cloud_data.shape):
            v = cloud_data[row_idx, col_idx]
            bits = [bool(int(c)) for c in bin(v)[2:].zfill(8)]
            bits.reverse()
            mask[row_idx, col_idx] = 1 if (
                    bits[0] and
                    (bits[1] or bits[2]) and
                    not bits[6] and
                    not bits[7]
            ) else 0

        print(any(mask.flatten()), np.count_nonzero(mask.flatten()), np.count_nonzero(mask.flatten()) / mask.size)

        return mask

    def process_region(self, args):
        overlapping_files, matching_cloud_files, mosaic_array, region_bounds, outfile = args
        region_min_lon, region_max_lon, region_min_lat, region_max_lat = region_bounds

        index_to_median = collections.defaultdict(list)
        for f_i, file in enumerate(overlapping_files):
            g = gdal.Open(file)
            gt = g.GetGeoTransform()
            data = g.ReadAsArray()

            cloud_file = gdal.Open(matching_cloud_files[file])
            cloud_mask = self._create_cloud_mask(cloud_file.ReadAsArray())

            t1 = time.time()
            for i, row in enumerate(data):
                row_lat = gt[3] + (gt[5] * i)
                if not region_min_lat <= row_lat <= region_max_lat:
                    continue
                for j, column in enumerate(row):
                    row_lon = gt[0] + (gt[1] * j)
                    region_indices = (int((region_max_lat - row_lat) / self._res),
                                      int((row_lon - region_min_lon) / self._res))
                    val = data[i, j]
                    index_to_median[region_indices].append(val if val >= 0 and not cloud_mask[i, j] else np.nan)
                if i % 1000 == 0:
                    print(f'{i} / {data.shape[0]} File {f_i + 1} / {len(overlapping_files)} {time.time() - t1}')

        for k, v in index_to_median.items():
            mosaic_array[k] = np.nanmedian(v)

        mosaic_array = mosaic_array.astype(np.float32)
        self._numpy_array_to_raster(
            outfile, mosaic_array, [region_min_lon, self._res, 0, region_max_lat, 0, -self._res],
            self._projection)

    @staticmethod
    def create_mosaic(in_dir: str, out_file: str):
        # List all TIFF files in the directory
        all_files = glob(os.path.join(in_dir, "*.tif"))

        # List to hold opened raster datasets
        src_files_to_mosaic = []

        # Open and append each raster to the list
        for fp in all_files:
            src = rasterio.open(fp)
            src_files_to_mosaic.append(src)

        # Merge function returns a single mosaic array and the transformation info
        mosaic, out_trans = merge(src_files_to_mosaic)

        # Copy the metadata
        out_meta = src.meta.copy()

        # Update the metadata to reflect the number of layers in the mosaic
        out_meta.update({"driver": "GTiff",
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans
                         })

        # Write the mosaic raster to disk
        with rasterio.open(out_file, "w", **out_meta) as dest:
            dest.write(mosaic)

    def _create_composite(self, file_dir: str, year: int, bbox: List[int], out_dir: str, output_type: str,
                          n_regions: int = 10, processes: int = 6):
        if output_type == 'WUE':
            output_re = self._wue_tif_re
            output_prefix = 'ECOSTRESS_L4_WUE_'
        elif output_type == 'ESI':
            output_re = self._esi_tif_re
            output_prefix = 'ECOSTRESS_L4_ESI_PT-JPL_'

        # First get all the files, filtering on the hour month and bounding box
        min_lon, min_lat, max_lon, max_lat = bbox[0], bbox[1], bbox[2], bbox[3]

        print('Finding cloud files')
        cloud_file_dict = {}
        for file in os.listdir(file_dir):
            match = re.match(self._cloud_file_tif_re, file)
            if match:
                key = self._generate_file_key(match.groupdict())
                cloud_file_dict[key] = os.path.join(file_dir, file)

        file_regions = {}
        for file in os.listdir(file_dir):
            if re.match(output_re, file):
                file_path = os.path.join(file_dir, file)
                g = gdal.Open(file_path)
                gt = g.GetGeoTransform()
                dim = g.ReadAsArray().shape
                bounds = gt[0], gt[0] + (gt[1] * dim[1]), gt[3] + (gt[5] * dim[0]), gt[3]
                file_regions[file_path] = bounds

        print('Matching files')
        matching_files = {}
        matching_cloud_files = {}
        for file, bounds in file_regions.items():
            group_dict = re.match(output_re, os.path.basename(file)).groupdict()
            key = self._generate_file_key(group_dict)
            if (
                    int(group_dict['year']) == year and
                    bounds[0] < max_lon and
                    bounds[1] > min_lon and
                    bounds[2] < max_lat and
                    bounds[3] > min_lat
            ) and key in cloud_file_dict:
                matching_files[file] = bounds
                matching_cloud_files[file] = cloud_file_dict[key]

        # Create an empty array with 70m x 70m resolution
        min_lon = min([c[0] for c in matching_files.values()])
        max_lon = max([c[1] for c in matching_files.values()])
        min_lat = min([c[2] for c in matching_files.values()])
        max_lat = max([c[3] for c in matching_files.values()])

        n_rows = int((max_lat - min_lat) / self._res)
        n_cols = int((max_lon - min_lon) / self._res)

        # Define the number of regions
        num_regions = n_regions

        # Define the size of each region
        region_height = n_rows // num_regions

        # Prepare arguments for each region
        print('Preparing regions')
        args = []
        for i in range(0, n_rows, region_height):
            h = min(region_height, n_rows - i)
            r_min_lat = min_lat + (i * self._res)
            r_max_lat = min(max_lat, r_min_lat + (h * self._res))
            r_outfile = os.path.join(out_dir, f'{output_prefix}{min_lon}_{max_lon}_{r_min_lat}_{r_max_lat}_{i}.tif')

            if os.path.exists(r_outfile):
                continue

            mosaic_array = np.full((h, n_cols), np.nan)

            overlapping_files = []
            for file, bounds in matching_files.items():
                if bounds[0] < max_lon and bounds[1] > min_lon and bounds[2] < r_max_lat and bounds[3] > r_min_lat:
                    overlapping_files.append(file)
            args.append((
                overlapping_files, matching_cloud_files, mosaic_array, (min_lon, max_lon, r_min_lat, r_max_lat),
                r_outfile))

        print('Processing regions')
        with mp.Pool(processes=processes) as pool:
            results = pool.map(self.process_region, args)

    @staticmethod
    def _h5_in_completed_files(h5_name: str, completed_files: List[str]):
        m = {
            '_WUE_': '_WUEavg_GEO.tif',
            '_ESI_PT-JPL_': '_ESIavg_GEO.tif',
            '_CLOUD_': '_CloudMask_GEO.tif'
        }
        for k, v in m.items():
            if k in h5_name:
                return h5_name.replace('.h5', v) in completed_files
        return False

    @staticmethod
    def _tif_file_exists(dest: str) -> bool:
        return (os.path.exists(os.path.join(os.path.dirname(os.path.dirname(dest)), 'geo_tiffs',
                                            os.path.basename(dest).strip('.h5') + '_WUEavg_GEO.tif')) or
                glob(os.path.join(os.path.dirname(os.path.dirname(dest)), 'geo_tiffs',
                                  os.path.basename(dest).strip('.h5').replace('L1B_GEO', 'L4_WUE')[:43] +
                                  '*' + '_WUEavg_GEO.tif')))

    def download_composite(self, years: List[int], month_start: int, month_end: int, hour_start: int, hour_end: int,
                           bbox: List[int], esi: bool = True, batch_size: int = 50):
        set_start_method('fork')

        out_dir = os.path.join(self.PROJ_DIR, 'data',
                               f"{'_'.join([str(y) for y in years])}_{month_start}_{month_end}_{hour_start}_{hour_end}")
        os.makedirs(out_dir, exist_ok=True)

        batch_out_dir = os.path.join(out_dir, 'batch')
        os.makedirs(batch_out_dir, exist_ok=True)

        geo_tiff_dir = os.path.join(out_dir, 'geo_tiffs')
        os.makedirs(geo_tiff_dir, exist_ok=True)

        url_progress_file_path = os.path.join(out_dir, 'url_progress.csv')
        if not os.path.exists(url_progress_file_path):
            pd.DataFrame({
                'wue_url': [],
                'geo_url': [],
                'esi_url': [],
                'cloud_url': [],
                'day': [],
                'accepted': []
            }).to_csv(url_progress_file_path, index=False)

        completed_files_path = os.path.join(out_dir, 'completed_files.txt')
        if not os.path.exists(completed_files_path):
            with open(completed_files_path, 'w+') as f:
                f.writelines([file + '\n' for file in os.listdir(geo_tiff_dir)])

        with open(completed_files_path, 'r') as f:
            completed_files = set([c.replace('\n', '') for c in f.readlines()])

        # Download the files if they don't exist
        urls = self.gather_file_links(years, month_start, month_end, hour_start, hour_end, bbox, url_progress_file_path)

        # The geo files are large enough that it makes sense to delete them periodically by processing the swaths in
        # batches
        url_batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]

        for url_batch in url_batches:
            os.makedirs(batch_out_dir, exist_ok=True)
            wue_requests, esi_requests, cloud_requests, geo_requests = [], [], [], []
            for url_set in url_batch:
                f = False
                if isinstance(url_set[0], str) and not self._h5_in_completed_files(os.path.basename(url_set[0]), completed_files):
                    f = True
                    wue_requests.append((url_set[0], os.path.join(batch_out_dir, os.path.basename(url_set[0]))))
                if esi and isinstance(url_set[1], str) and not self._h5_in_completed_files(os.path.basename(url_set[1]), completed_files):
                    f = True
                    esi_requests.append((url_set[1], os.path.join(batch_out_dir, os.path.basename(url_set[1]))))
                if isinstance(url_set[2], str) and not self._h5_in_completed_files(os.path.basename(url_set[2]), completed_files):
                    f = True
                    cloud_requests.append((url_set[2], os.path.join(batch_out_dir, os.path.basename(url_set[2]))))
                if isinstance(url_set[3], str) and f:
                    geo_requests.append((url_set[3], os.path.join(batch_out_dir, os.path.basename(url_set[3]))))

            if wue_requests:
                print(f'Downloading WUE requests {wue_requests}')
                self.download_time_series(wue_requests, batch_out_dir)

            if geo_requests:
                print(f'Downloading GEO requests {geo_requests}')
                self.download_time_series(geo_requests, batch_out_dir)

            if cloud_requests:
                print(f'Downloading CLOUD requests {cloud_requests}')
                self.download_time_series(cloud_requests, batch_out_dir)

            if esi_requests:
                print(f'Downloading ESI requests {esi_requests}')
                self.download_time_series(esi_requests, batch_out_dir)

            # Convert them into TIFs
            if geo_requests or wue_requests or cloud_requests or esi_requests:
                main(Namespace(proj='GEO', dir=batch_out_dir, out_dir=geo_tiff_dir, sds=None, utmzone=None, bt=None))
                with open(completed_files_path, 'r') as f:
                    completed_files = f.readlines()
                with open(completed_files_path, 'w') as f:
                    f.writelines(list(set(completed_files + [file + '\n' for file in os.listdir(geo_tiff_dir)])))
                shutil.rmtree(batch_out_dir)
