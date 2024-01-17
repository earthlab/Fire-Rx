from apis.ecostress import L4WUE
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir')
    parser.add_argument('--n_regions')
    parser.add_argument('--processes')
    parser.add_argument('--outdir')

    args = parser.parse_args()

    l = L4WUE('', '')
    l._create_composite(args.in_dir, 2019, 1, 12, 0, 24,
                        [-360, -360, 360, 360], args.outdir, int(args.n_regions), int(args.processes))
