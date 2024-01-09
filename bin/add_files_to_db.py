import argparse
from apis.ecostress import L4WUE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir')
    parser.add_argument('--db_dir')
    parser.add_argument('--n_tasks')

    args = parser.parse_args()

    l = L4WUE('', '')
    l.add_files_to_db(args.in_dir, args.db_dir, int(args.n_tasks))
