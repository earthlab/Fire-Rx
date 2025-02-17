import sys

from apis.ecostress import WUE
import argparse
import os


PROJ_DIR = os.path.dirname(os.path.dirname(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', required=True, nargs='+', type=int,
                        help='The years for which to download WUE data for the resulting composite')
    parser.add_argument('--month_start', required=True, type=int,
                        help='The beginning of month range for which to download WUE data for the resulting composite')
    parser.add_argument('--month_end', required=True, type=int,
                        help='The end of month range for which to download WUE data for the resulting composite')
    parser.add_argument('--hour_start', required=True, type=int,
                        help='The beginning of hour of day range for which to download WUE data for the resulting composite')
    parser.add_argument('--hour_end', required=True, type=int,
                        help='The end of hour of day range for which to download WUE data for the resulting composite')
    parser.add_argument('--bbox', required=True, nargs='+', type=float,
                        help='Bounding box in [min_lon, min_lat, max_lon, max_lat] for which to find intersecting'
                             ' WUE files. Download WUE for the resulting composite.')
    parser.add_argument('--download_batch_size', required=False, default=50, type=int)

    parser.add_argument('--n_regions', type=int, default=10)

    parser.add_argument('--processes', type=int, default=6)

    parser.add_argument('--out_dir', required=False)

    args = parser.parse_args()

    # Try to get the credentials from environment variables
    username = os.environ.get('FIRE_RX_USERNAME')
    password = os.environ.get('FIRE_RX_PASSWORD')

    # If they're not set, prompt the user securely
    if not username:
        print('Please set FIRE_RX_USERNAME ENV variable to your NASA Earthdata username')
        sys.exit(2)
    if not password:
        print('Please set FIRE_RX_PASSWORD ENV variable to your NASA Earthdata password')
        sys.exit(2)

    # Now create the WUE instance with the secure credentials
    wue = WUE(username, password)

    if args.out_dir is None:
        out_dir = os.path.join(PROJ_DIR, 'data',
                            f"{'_'.join([str(int(b)) for b in args.bbox])}_{'_'.join([str(y) for y in args.years])}_{args.month_start}_{args.month_end}_{args.hour_start}_{args.hour_end}")
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f'Downloading files to {out_dir}')
    wue.download_and_rasterize_ecostress(out_dir, args.years, args.month_start, args.month_end, args.hour_start,
                                                args.hour_end, args.bbox, args.download_batch_size)
    for year in args.years:
        output_dir = os.path.join(out_dir, str(year))
        wue.create_composite(os.path.join(out_dir, 'geo_tiffs'), year, args.bbox, output_dir, args.n_regions, args.processes)
        out_file = f"ECOSTRESS_L4_WUE_{'_'.join([str(int(b)) for b in args.bbox])}_{year}_{args.month_start}_{args.month_end}_{args.hour_start}_{args.hour_end}.tif"
        print(f'Writing mosaic to {os.path.join(output_dir, out_file)}')
        wue.create_mosaic(output_dir, os.path.join(output_dir, out_file))