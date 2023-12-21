import collections
import getpass
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
from datetime import datetime, timedelta
import calendar
import subprocess as sp
from rasterio.merge import merge
import rasterio as rio
import h5py
from apis.ecostress_conv.ECOSTRESS_swath2grid import main
from glob import glob
import xml.etree.ElementTree as ET
import multiprocessing as mp

import bs4
import certifi
import requests
import urllib3.util
from osgeo import gdal
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon


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
        response = requests.get(xml_url, auth=(self._username, self._password))

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the XML content
            root = ET.fromstring(response.content)
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


class L4WUE(BaseAPI):

    def __init__(self, username: str = None, password: str = None, lazy: bool = False):
        super().__init__(username=username, password=password, lazy=lazy)
        common_regex = r'\_(?P<orbit>\d{5})\_(?P<scene_id>\d{3})\_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})T(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})\_(?P<build_id>\d{4})\_(?P<version>\d{2})\.h5$'
        self._wue_file_re = r'ECOSTRESS\_L4\_WUE' + common_regex
        self._bgeo_file_re = r'ECOSTRESS\_L1B\_GEO' + common_regex

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

        # urls = []
        # geo_lookup = collections.defaultdict(list)
        # for day_url in day_urls:
        #     day_files = self.retrieve_links(day_url)
        #     for day_file in day_files:
        #         match = re.match(self._wue_file_re, day_file)
        #         if match:
        #             groups = match.groupdict()
        #             if hour_start <= int(groups['hour']) < hour_end:
        #                 url = urllib.parse.urljoin(day_url, day_file)
        #                 if self._overlaps_bbox(bbox, url + '.xml'):
        #                     geo_lookup[os.path.basename(day_url[:-1])].append(url)
        #                     urls.append(url)
        #                 else:
        #                     print('Doesnt overlap')

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
    def _create_composite(in_dir: str, out_file: str):
        # Get a list of all the geotif images in the specified directory
        image_list = glob(os.path.join(in_dir, '*.tif'))

        if not image_list:
            raise FileNotFoundError(f'No tif files found in {in_dir}')

        # Read the data from each image and store it in a list
        image_data = []
        x_size = None
        y_size = None
        # min_lat, max_lat, max_lon, min_lon = None, None, None, None
        for image in image_list:
            raster = rio.open(image)
            image_data.append(raster)
            # dataset = gdal.Open(image)
            # gt = dataset.GetGeoTransform()
            # # min_lat = gt[0] if min_lat is None or min_lat > gt[0] else min_lat
            # # min_lon = gt[3] if min_lon is None or min_lon > gt[3] else min_lon
            # # max_lat = gt[0] if max_lat is None or max_lat < gt[0] else max_lat
            # # max_lon = gt[3] if max_lon is None or max_lon < gt[3] else max_lon
            # x_size = dataset.RasterXSize if x_size is None else x_size
            # y_size = dataset.RasterYSize if y_size is None else y_size
            # image_data.append(dataset.ReadAsArray())
        # print(min_lat, max_lat, max_lon, min_lon)
        # median_image = np.median(image_data, axis=0)

        mosaic, output = merge(image_data, method='max')

        output_meta = raster.meta.copy()
        output_meta.update(
            {"driver": "GTiff",
             "height": mosaic.shape[1],
             "width": mosaic.shape[2],
             "transform": output,
             }
        )

        with rio.open(out_file, 'w', **output_meta) as m:
            m.write(mosaic)

        # # Write the median composite image to a new geotif file
        # driver = gdal.GetDriverByName('GTiff')
        # output_dataset = driver.Create(out_file, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Float32)
        # output_dataset.SetGeoTransform(dataset.GetGeoTransform())
        # output_dataset.SetProjection(dataset.GetProjection())
        # output_dataset.GetRasterBand(1).WriteArray(median_image)
        # output_dataset.FlushCache()

    @staticmethod
    def _tif_file_exists(dest: str) -> bool:
        return (os.path.exists(os.path.join(os.path.dirname(os.path.dirname(dest)), 'geo_tiffs',
                                            os.path.basename(dest).strip('.h5') + '_WUEavg_GEO.tif')) or
                glob(os.path.join(os.path.dirname(os.path.dirname(dest)), 'geo_tiffs',
                                  os.path.basename(dest).strip('.h5').replace('L1B_GEO', 'L4_WUE')[:43] +
                                  '*' + '_WUEavg_GEO.tif')))

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

        # Create the mosaic
        self._create_composite(out_dir, out_file)
