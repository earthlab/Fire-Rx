import collections
import getpass
import json
import math
import os
import re
import shutil
import sys
import urllib
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
from sqlalchemy.orm import Session
from sqlalchemy import extract, and_, create_engine
from database.tables import Pixel, Base
import time

import bs4
import certifi
import requests
import urllib3.util
from osgeo import gdal
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon


def process_file(file, max_lat, min_lon, res):
    # Function to process a single file
    medians = collections.defaultdict(list)
    g = gdal.Open(file)
    a = g.ReadAsArray()
    geo_transform = g.GetGeoTransform()

    progress = tqdm(total=a.size, desc=os.path.basename(file))

    for j, row in enumerate(a):
        lat_index = int((max_lat - (geo_transform[3] + (j * geo_transform[5])) / res))
        for i in range(len(row)):
            lon_index = int(((geo_transform[0] + (i * geo_transform[1])) - min_lon) / res)
            medians[(lat_index, lon_index)].append(a[j, i])
            progress.update(1)
    return medians


def combine_medians(results):
    # Function to combine the results from all processes
    combined_medians = collections.defaultdict(list)
    for medians in results:
        for key in tqdm(medians):
            combined_medians[key].append(medians[key])
    return combined_medians


class BaseAPI:
    """
    Defines all the attributes and methods common to the child APIs.
    """
    PROJ_DIR = os.path.dirname(os.path.dirname(__file__))
    _BASE_WUE_URL = 'https://e4ftl01.cr.usgs.gov/ECOSTRESS/ECO4WUE.001/'
    _BASE_GEO_URL = 'https://e4ftl01.cr.usgs.gov/ECOSTRESS/ECO1BGEO.001/'

    def __init__(self, username: str = None, password: str = None, lazy: bool = False):
        """
        Initializes the common attributes required for each data type's API
        """
        self._username = os.environ.get('FIRE_RX_USER', username)
        self._password = os.environ.get('FIRE_RX_PASS', password)
        self._core_count = os.cpu_count()
        if not lazy:
            self._configure()
        self._file_re = None
        self._tif_re = None

    @staticmethod
    def retrieve_links(url: str) -> List[str]:
        """
        Creates a list of all the links found on a webpage
        Args:
            url (str): The URL of the webpage for which you would like a list of links

        Returns:
            (list): All the links on the input URL's webpage
        """
        request = requests.get(url)
        soup = bs4.BeautifulSoup(request.text, 'html.parser')
        return [link.get('href') for link in soup.find_all('a')]

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
        response = urllib.request.urlopen(myrequest)

        # Check if the request was successful
        if response.status == 200:
            # Parse the XML content
            root = ET.fromstring(response.read())
            bounding_rect = root.find('.//BoundingRectangle')
            west = float(bounding_rect.find('WestBoundingCoordinate').text)
            north = float(bounding_rect.find('NorthBoundingCoordinate').text)
            east = float(bounding_rect.find('EastBoundingCoordinate').text)
            south = float(bounding_rect.find('SouthBoundingCoordinate').text)

            return Polygon([(west, north), (east, north), (east, south), (west, south)])

        else:
            print("Failed to retrieve XML: HTTP Status Code", response.status_code)

    def _overlaps_bbox(self, target_bbox: List[int], xml_url: str):
        # Now apply spatial filter by downloading the xml files and checking if they overlap the bounding box
        min_lon, min_lat, max_lon, max_lat = target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3]
        target_bbox = Polygon([(min_lon, max_lat), (max_lon, max_lat), (max_lon, min_lat), (min_lon, min_lat)])
        file_bbox = self._parse_bbox_from_xml(xml_url)
        return file_bbox.intersects(target_bbox)

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


class L4WUE(BaseAPI):

    def __init__(self, username: str = None, password: str = None, lazy: bool = False):
        super().__init__(username=username, password=password, lazy=lazy)
        common_regex = r'\_(?P<orbit>\d{5})\_(?P<scene_id>\d{3})\_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})T(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})\_(?P<build_id>\d{4})\_(?P<version>\d{2})\.h5$'
        self._wue_file_re = r'ECOSTRESS\_L4\_WUE' + common_regex
        self._bgeo_file_re = r'ECOSTRESS\_L1B\_GEO' + common_regex
        self._wue_tif_re = r'ECOSTRESS\_L4\_WUE' + common_regex.replace('.h5', '_WUEavg_GEO.tif')
        self._db_re = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2}) (?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})\_(?P<min_lon>\-?\d+\.\d+)\_(?P<max_lon>\-?\d+\.\d+)\_(?P<min_lat>\-?\d+\.\d+)\_(?P<max_lat>\-?\d+\.\d+)\_(?P<lon_res>\-?\d+\.\d+)\_(?P<lat_res>\-?\d+\.\d+)\.db$'
        self._res = 0.0006298419

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
            if hour_start <= int(groups['hour']) < hour_end:
                url = urllib.parse.urljoin(day_url, day_file)
                if self._overlaps_bbox(bbox, url + '.xml'):
                    return os.path.basename(day_url[:-1]), url
        return None

    def gather_file_links(self, year: int, month_start: int, month_end: int, hour_start: int,
                          hour_end: int, bbox: List[int]) -> List[Tuple[str, str]]:
        start_date = datetime(year, month_start, 1)
        end_date = datetime(year, month_end, self._get_last_day_of_month(year, month_end))

        day_urls = []
        while start_date <= end_date:
            day_urls.append(urllib.parse.urljoin(self._BASE_WUE_URL, start_date.strftime('%Y.%m.%d') + '/'))
            start_date += timedelta(days=1)

        # Prepare arguments for multiprocessing
        args = []
        for day_url in day_urls:
            day_files = self.retrieve_links(day_url)
            for day_file in day_files:
                args.append((day_url, day_file, bbox, hour_start, hour_end))

        geo_lookup = collections.defaultdict(list)

        with Pool(mp.cpu_count() - 1) as pool:
            # Using tqdm to create a progress bar
            for result in tqdm(pool.imap_unordered(self._worker, args), total=len(args),
                               desc='Finding overlapping files'):
                if result:
                    day, url = result
                    geo_lookup[day].append(url)

        # Now find the GEO urls. The versions are not always the same (this doesn't matter for the swath2grid function)
        # so you cannot infer the GEO url from the WUE url. What should match is the orbit, scene_id, and date.
        paired_urls = []
        for date, wue_file_links in geo_lookup.items():
            date_url = urllib.parse.urljoin(self._BASE_GEO_URL, date)
            geo_links = self.retrieve_links(date_url)
            geo_file_dict = {}
            for geo_link in geo_links:
                match = re.match(self._bgeo_file_re, geo_link)
                if match:
                    geo_file_group_dict = match.groupdict()
                    key = self._generate_file_key(geo_file_group_dict)
                    geo_file_dict[key] = urllib.parse.urljoin(date_url + '/', geo_link)

            for wue_file_link in wue_file_links:
                group_dict = re.match(self._wue_file_re, os.path.basename(wue_file_link)).groupdict()
                key = self._generate_file_key(group_dict)
                if key in geo_file_dict:
                    paired_urls.append((wue_file_link, geo_file_dict[key]))

        return paired_urls

    @staticmethod
    def process_region(args):
        i_start, i_end, j_start, j_end, min_lat, min_lon, res, matching_db = args
        region = np.empty((i_end - i_start, j_end - j_start), dtype=np.float32)

        prog = tqdm(total=(i_end - i_start) * (j_end - j_start), desc=str(i_start))

        matching_db = [create_engine(m) for m in matching_db]

        for i in range(i_start, i_end):
            row_min_lat = min_lat + (i * res)
            row_max_lat = row_min_lat + res
            for j in range(j_start, j_end):
                col_min_lon = min_lon + (j * res)
                col_max_lon = col_min_lon + res

                vals = []
                for engine in matching_db:
                    with Session(bind=engine) as session:
                        pixels = session.query(Pixel).filter(
                            and_(
                                Pixel.latitude < row_max_lat,
                                Pixel.latitude >= row_min_lat,
                                Pixel.longitude < col_max_lon,
                                Pixel.longitude >= col_min_lon
                            )
                        ).all()
                        vals += [p.value for p in pixels]
                region[i - i_start, j - j_start] = np.nanmedian(vals)
                prog.update(1)

        return i_start, j_start, region

    def _create_composite(self, in_dir: str, year: int, month_start: int, month_end: int, hour_start: int,
                          hour_end: int, bbox: List[int], out_file: str):

        # First get all the files, filtering on the hour month and bounding box
        min_lon, min_lat, max_lon, max_lat = bbox[0], bbox[1], bbox[2], bbox[3]

        matching_db = {}
        for file in os.listdir(in_dir):
            match = re.match(self._db_re, file)
            if match is not None:
                group_dict = match.groupdict()
                f_min_lon = float(group_dict['min_lon'])
                f_max_lon = float(group_dict['max_lon'])
                f_min_lat = float(group_dict['min_lat'])
                f_max_lat = float(group_dict['max_lat'])

                if (
                        hour_start <= int(group_dict['hour']) < hour_end and
                        month_start <= int(group_dict['month']) <= month_end and
                        int(group_dict['year']) == year and
                        f_min_lon < max_lon and
                        f_max_lon > min_lon and
                        f_min_lat < max_lat and
                        f_max_lat > min_lat
                ):
                    matching_db[f'sqlite:///{os.path.join(in_dir, file)}'] = (f_min_lon, f_max_lon, f_min_lat, f_max_lat)

        # Create an empty array with 70m x 70m resolution
        min_lon = min([c[0] for c in matching_db.values()])
        max_lon = max([c[1] for c in matching_db.values()])
        min_lat = min([c[2] for c in matching_db.values()])
        max_lat = max([c[3] for c in matching_db.values()])

        print(min_lon, max_lon, min_lat, max_lat)

        n_rows = int(math.ceil(max_lat - min_lat) / self._res)
        n_cols = int(math.ceil(max_lon - min_lon) / self._res)

        print(n_rows, n_cols)

        # pool = mp.Pool(mp.cpu_count() - 1)
        # combined_medians = combine_medians(pool.starmap(process_file, [(file, max_lat, min_lon, self._res) for file in
        #                                                matching_geo_tiffs]))
        # pool.close()
        # pool.join()

        mosaic_array = np.empty((n_rows, n_cols), dtype=np.float32)
        progress = tqdm(total=mosaic_array.size, desc='Mosaicing')
        # Define the number of regions (this could be the number of available CPU cores)
        num_regions = 7

        # Define the size of each region
        region_height = n_rows // num_regions
        region_width = n_cols

        # Prepare arguments for each region
        regions = []
        for i in range(0, n_rows, region_height):
            args = (i, min(i + region_height, n_rows), 0, n_cols, min_lat, min_lon, self._res, matching_db)
            regions.append(args)

        # Process each region in parallel
        with Pool() as pool:
            results = pool.map(self.process_region, regions)

        # Combine the results
        for i_start, j_start, region in results:
            mosaic_array[i_start:i_start + region.shape[0], j_start:j_start + region.shape[1]] = region

        # Assuming progress is a tqdm object
        progress.update(mosaic_array.size)

        self._numpy_array_to_raster(out_file, mosaic_array,
                                    [min_lon, self._res, 0, max_lat, 0, -self._res],
                                    [min_lon, self._res, 0, max_lat, 0, -self._res])

    @staticmethod
    def _tif_file_exists(dest: str) -> bool:
        return (os.path.exists(os.path.join(os.path.dirname(os.path.dirname(dest)), 'geo_tiffs',
                                            os.path.basename(dest).strip('.h5') + '_WUEavg_GEO.tif')) or
                glob(os.path.join(os.path.dirname(os.path.dirname(dest)), 'geo_tiffs',
                                  os.path.basename(dest).strip('.h5').replace('L1B_GEO', 'L4_WUE')[:43] +
                                  '*' + '_WUEavg_GEO.tif')))

    @staticmethod
    def _prepare_pixel_data(array, gt):
        pixel_data = []
        for i, row in enumerate(array):
            mid_lat = gt[3] + (i * gt[5]) + (gt[5] / 2)
            for j, col in enumerate(row):
                pixel_info = {
                    'value': array[i, j],
                    'latitude': mid_lat,
                    'longitude': gt[0] + (j * gt[1]) + (gt[1] / 2)
                }
                pixel_data.append(pixel_info)
        return pixel_data

    @staticmethod
    def _insert_pixels(pixel_data):
        pixel_data, session, batch = pixel_data

        try:
            pixels = []
            for data in tqdm(pixel_data, total=len(pixel_data), desc=f'Processing batch {batch}'):
                new_pixel = Pixel(
                    value=data['value'],
                    latitude=data['latitude'],
                    longitude=data['longitude'],
                    _file_id=data['file_id']  # Assuming 'file' is referenced by 'file_id'
                )
                pixels.append(new_pixel)
            session.bulk_save_objects(pixels)
            session.commit()
        except Exception as e:
            print(f"Error: {e}")
            session.rollback()
        finally:
            session.close()

    def _add_file_to_db(self, file_data):
        file, geo_tiff_dir, db_dir = file_data
        print('running')

        match = re.match(self._wue_tif_re, file)
        if match:
            params = match.groupdict()

            g = gdal.Open(os.path.join(geo_tiff_dir, file))
            gt = g.GetGeoTransform()
            array = g.ReadAsArray()

            timestamp = datetime(year=int(params['year']), month=int(params['month']), day=int(params['day']),
                                 hour=int(params['hour']), minute=int(params['minute']),
                                 second=int(params['second']))
            min_lon = str(gt[0])
            max_lon = str(gt[0] + (gt[1] * array.shape[0]))
            max_lat = str(gt[3])
            min_lat = str(gt[3] + (gt[5] * array.shape[1]))
            lon_res = str(gt[1])
            lat_res = str(gt[5])

            db_file_path = os.path.join(db_dir, str(timestamp) + '_' + min_lon + '_' + max_lon + '_' + min_lat + '_'
                                        + max_lat + '_' + lon_res + '_' + lat_res + '.db')

            engine = create_engine(f'sqlite:///{db_file_path}')
            Base.metadata.create_all(engine)

            print('Adding new file')

            with Session(bind=engine) as session:
                pixels = []
                for i, row in enumerate(tqdm(array, total=array.shape[0], desc='Adding pixels')):
                    mid_lat = gt[3] + (i * gt[5]) + (gt[5] / 2)
                    for j, col in enumerate(row):
                        # p = {
                        #     'value': array[i, j],
                        #     'latitude': mid_lat,
                        #     'longitude': gt[0] + (j * gt[1]) + (gt[1] / 2)
                        # }

                        p = Pixel(
                            value=array[i, j],
                            latitude=mid_lat,
                            longitude=gt[0] + (j * gt[1]) + (gt[1] / 2)
                        )
                        pixels.append(p)

                        if len(pixels) >= 500000:
                            session.bulk_save_objects(pixels)
                            #session.commit()
                            pixels.clear()

                print('Added pixels')
                session.bulk_save_objects(pixels)
                session.commit()

    def add_files_to_db(self, geo_tiff_dir, db_dir: str, n_tasks: int = mp.cpu_count() - 1):
        args_list = [(file, geo_tiff_dir, db_dir) for file in os.listdir(geo_tiff_dir)]
        with mp.Pool(processes=n_tasks) as pool:
            # Use pool.imap_unordered to process the files in parallel
            # Iterate over the imap_unordered iterator to start the tasks and wait for them to finish
            for _ in pool.imap_unordered(self._add_file_to_db, args_list):
                pass
        # for arg in args_list:
        #     self._add_file_to_db(arg)

    def download_composite(self, year: int, month_start: int, month_end: int, hour_start: int, hour_end: int,
                           out_file: str, bbox: List[int], batch_size: int = 50):
        set_start_method('fork')

        out_dir = os.path.join(self.PROJ_DIR, 'apis', f'{year}_{month_start}_{month_end}_{hour_start}_{hour_end}')
        os.makedirs(out_dir, exist_ok=True)

        batch_out_dir = os.path.join(out_dir, 'batch')
        os.makedirs(batch_out_dir, exist_ok=True)

        geo_tiff_dir = os.path.join(out_dir, 'geo_tiffs')
        os.makedirs(geo_tiff_dir, exist_ok=True)

        # Download the files if they don't exist
        urls = self.gather_file_links(year, month_start, month_end, hour_start, hour_end, bbox)

        print(f'{len(urls)} URLs total')

        # The geo files are large enough that it makes sense to delete them periodically by processing the swaths in
        # batches
        url_batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]

        for url_batch in url_batches:
            os.makedirs(batch_out_dir, exist_ok=True)

            wue_requests = [(url_pair[0], os.path.join(batch_out_dir, os.path.basename(url_pair[0]))) for
                            url_pair in url_batch if
                            not self._tif_file_exists(os.path.join(batch_out_dir, os.path.basename(url_pair[0])))]

            geo_requests = [(url_pair[1], os.path.join(batch_out_dir, os.path.basename(url_pair[1]))) for
                            url_pair in url_batch if
                            not self._tif_file_exists(os.path.join(batch_out_dir, os.path.basename(url_pair[1])))]

            if wue_requests:
                self.download_time_series(wue_requests, batch_out_dir)

            if geo_requests:
                self.download_time_series(geo_requests, batch_out_dir)

            # Convert them into TIFs
            if os.listdir(batch_out_dir):
                main(Namespace(proj='GEO', dir=batch_out_dir, out_dir=geo_tiff_dir, sds=None, utmzone=None, bt=None))
                shutil.rmtree(batch_out_dir)

        self.add_files_to_db(geo_tiff_dir)
