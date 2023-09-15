import os
import sys
rootdir = "/Users/conradkrueger/eICU_benchmark/output"
elim = sys.argv[1]
verbose = len(sys.argv) == 3
if elim is None:
    sys.exit()
for subdir, _, _ in os.walk(rootdir):
    try:
        if verbose: print("DELETED",os.path.join(subdir, elim))
        os.remove(os.path.join(subdir, elim))
    except Exception as e:
        print(e)
            
print("DONE")