import os
import sys

def generate_all(mtx_data_dir):
    '''
    Generates a set of labeled data pairs (x,y), where each x is a csv containing
    1601 sampled rows of a matrix A and columns containing k statistics about
    those rows, and each corresponding y is the optimal MueLu parameters for A
    '''
    print('Encoding all mtx files...')

    _fname = None
    for i in range(10):
	    for root, dirs, files in os.walk(mtx_data_dir):

	        files = sorted(list(files))
	        files = [f for f in files if f.endswith('.mtx')]
	        print(f'Found {len(files)} matrix market files!')

	        for f in files:
	            if f.endswith('.mtx'):
	                _fname = f
	                fname = os.path.join(root, _fname)
	                cmd = f'mpiexec --np 1 ./mtx_stats {fname}'
	                os.system(cmd)

	    print('Done.')

def main():

    # Get program arguments
    args = sys.argv
    args_opt = [a for a in args if a.startswith('--')]
    
    # Default matrix market data directory
    mtx_data_dir = 'data/matrices'
    
    # Switch to passed dir if provided
    if '--mtxdir' in args_opt:
        mtx_data_dir = args[args.index('--mtxdir') + 1]

    generate_all(mtx_data_dir)

if __name__ == '__main__':
    main()

