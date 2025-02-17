import os
import sys
import argparse

from apis.srtm import Elevation

PROJ_DIR = os.path.dirname(os.path.dirname(__file__))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbox', required=True, nargs='+', type=float,
                        help='Bounding box in [min_lon, min_lat, max_lon, max_lat] for which to find intersecting'
                             ' SRTM files.')
    parser.add_argument('--out_dir', type=str, required=False)

    args = parser.parse_args()

    if args.out_dir is None:
        out_dir = os.path.join(PROJ_DIR, 'data', 'srtm', f"srtm_{'_'.join([str(int(b)) for b in args.bbox])}")
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    username = os.environ.get('FIRE_RX_USERNAME')
    password = os.environ.get('FIRE_RX_PASSWORD')
    if not username:
        print('Please set FIRE_RX_USERNAME ENV variable to your NASA Earthdata username')
        sys.exit(2)
    if not password:
        print('Please set FIRE_RX_PASSWORD ENV variable to your NASA Earthdata password')
        sys.exit(2)

    e = Elevation(username, password)
    print(f'Downloading to {out_dir}')
    e.download_bbox(args.bbox, out_dir)
    print(f"Elevation file written to {os.path.join(out_dir, 'elevation.tif')}")
    print(f"Writing slope file to {os.path.join(out_dir, 'slope.tif')}")
    e.rd_derive_slope_from_elevation(os.path.join(out_dir, 'elevation.tif'), os.path.join(out_dir, 'slope.tif'))
    print('Done')
    print(f"Writing aspect file to {os.path.join(out_dir, 'aspect.tif')}")
    e.rd_derive_aspect_from_elevation(os.path.join(out_dir, 'elevation.tif'), os.path.join(out_dir, 'aspect.tif'))
    print('Done')
