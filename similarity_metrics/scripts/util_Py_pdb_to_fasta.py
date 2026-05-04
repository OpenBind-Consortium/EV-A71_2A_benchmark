import os
import sys
from Bio import SeqIO

PDBFile = sys.argv[1]
outfile = sys.argv[2]
out = []
with open(PDBFile, 'r') as pdb_file:
    for record in SeqIO.parse(pdb_file, 'pdb-atom'):
        #print(record)
        #print('>' + record.id)
        print(f'>{os.path.basename(PDBFile)[:-4]}.' + record.annotations["chain"])
        out.append(f'>{os.path.basename(PDBFile)[:-4]}.' + record.annotations["chain"])
        out.append(str(record.seq))

with open(outfile, 'w') as fo:
    fo.write('\n'.join(out))

