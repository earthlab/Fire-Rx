import getpass
import os
import re
import sys
import urllib
from http.cookiejar import CookieJar
from multiprocessing import Pool
from typing import Tuple, List
from datetime import datetime, timedelta
import calendar
import subprocess as sp

import bs4
import certifi
import requests
import urllib3.util
from osgeo import gdal
import numpy as np
import glob
from tqdm import tqdm


class BaseAPI:
    """
    Defines all the attributes and methods common to the child APIs.
    """
    PROJ_DIR = os.path.dirname(os.path.dirname(__file__))

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

    def _download(self, query: Tuple[str, str]) -> None:
        """
        Downloads data from the NASA earthdata servers. Authentication is established using the username and password
        found in the local ~/.netrc file.
        Args:
            query (tuple): Contains the remote location and the local path destination, respectively
        """
        link = query[0]
        dest = query[1]

        if os.path.exists(dest):
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
            try:
                with Pool(int(self._core_count / 2)) as pool:
                    for _ in tqdm(pool.imap_unordered(self._download, queries), total=len(queries)):
                        pass

            except Exception as pe:
                try:
                    _ = [self._download(q) for q in tqdm(queries, position=0, file=sys.stdout)]
                except Exception as e:
                    template = "Download failed: error type {0}:\n{1!r}"
                    message = template.format(type(e).__name__, e.args)
                    print(message)

        print(f'Wrote {len(queries)} files to {outdir}')


class L4WUE(BaseAPI):
    _BASE_WUE_URL = 'https://e4ftl01.cr.usgs.gov/ECOSTRESS/ECO4WUE.001/'
    _BASE_GEO_URL = 'https://e4ftl01.cr.usgs.gov/ECOSTRESS/ECO1BGEO.001/'

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

    def download_time_series(self, year: int, month_start: int, month_end: int, hour_start: int, hour_end: int,
                             out_dir: str) -> None:
        start_date = datetime(year, month_start, 1)
        end_date = datetime(year, month_end, self._get_last_day_of_month(year, month_end))

        day_urls = []
        while start_date <= end_date:
            day_urls.append(urllib.parse.urljoin(self._BASE_WUE_URL, start_date.strftime('%Y.%m.%d') + '/'))
            start_date += timedelta(days=1)

        urls = []
        for day_url in day_urls:
            print(day_url)
            day_files = self.retrieve_links(day_url)
            print(day_files)
            for day_file in day_files:
                match = re.match(self._wue_file_re, day_file)
                if match:
                    print(day_file)
                    groups = match.groupdict()
                    if hour_start <= int(groups['hour']) <= hour_end:
                        urls.append(urllib.parse.urljoin(day_url, day_file))
                        urls.append(urllib.parse.urljoin(day_url.replace(self._BASE_WUE_URL, self._BASE_GEO_URL),
                                                         day_file.replace('L4_WUE', 'L1B_GEO')))

        super().download_time_series([(url, os.path.join(out_dir, os.path.basename(url))) for url in urls], out_dir)

    @staticmethod
    def _create_composite(in_dir: str, out_file: str):
        # Get a list of all the geotif images in the specified directory
        image_list = glob.glob(os.path.join(in_dir, '*.tif'))

        if not image_list:
            raise FileNotFoundError(f'No tif files found in {in_dir}')

        # Read the data from each image and store it in a list
        image_data = []
        for image in image_list:
            dataset = gdal.Open(image)
            image_data.append(dataset.ReadAsArray())

        # Compute the median composite image using numpy
        median_image = np.median(image_data, axis=0)

        # Write the median composite image to a new geotif file
        driver = gdal.GetDriverByName('GTiff')
        output_dataset = driver.Create(out_file, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Float32)
        output_dataset.SetGeoTransform(dataset.GetGeoTransform())
        output_dataset.SetProjection(dataset.GetProjection())
        output_dataset.GetRasterBand(1).WriteArray(median_image)
        output_dataset.FlushCache()

    def download_composite(self, year: int, month_start: int, month_end: int, hour_start: int, hour_end: int,
                           out_dir: str, out_file: str):

        # Download the files if they don't exist
        self.download_time_series(year, month_start, month_end, hour_start, hour_end, out_dir)

        # Convert them into TIFs
        sp.call([sys.executable, os.path.join(self.PROJ_DIR, 'apis', 'ecostress_conv', 'ECOSTRESS_swath2grid.py'),
                 '--proj', 'GEO', '--dir', out_dir])

        # Create the mosaic
        self._create_composite(out_dir, out_file)
