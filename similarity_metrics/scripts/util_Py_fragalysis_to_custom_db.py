import argparse
import shutil
import os
from pymol import cmd, stored
from Bio import SeqIO

parser = argparse.ArgumentParser()

parser.add_argument('--fragalysis_dir', '-fd', help='Path to directory of aligned_files/ fragalaysis results')
parser.add_argument('--outdir', '-od', help='Path to an output directory where reformatted results will be stored')

args = parser.parse_args()

ALPHA='ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def main():
    os.makedirs(args.outdir, exist_ok=True)

    structout = f'{args.outdir}/structures'
    os.makedirs(structout, exist_ok=True)
    case_l = os.listdir(args.fragalysis_dir)
    
    annotations_out = ['system_id,release_date'] #Give a dummy release date for now
    for case in case_l:
        case_dir = f'{args.fragalysis_dir}/{case}/'
        
        print(case_dir)
        rec_pdb = f'{case_dir}/{case}_apo-desolv.pdb'
        system_lig = f'{case_dir}/{case}_ligand.sdf'

        #for record in SeqIO.parse(rec_pdb, "pdb-atom"):
        #    print(f"Chain {record.id}: {record.seq}")
        
        cmd.reinitialize()
        cmd.load(rec_pdb, 'rec')
        prot_ch_l = cmd.get_chains('polymer.protein')
        print(prot_ch_l)
        cmd.load(system_lig, 'lig',)
        for a in ALPHA:
            if a not in prot_ch_l:
                cmd.alter('lig', f'chain="{a}"')
                print(f'Assign lig chain {a}')
                break

        lig_ch_l = cmd.get_chains('lig')
        print(lig_ch_l)
        
        #7pcs__1__1.B__1.G_1.H
        outname = f'{case}__1__{"_".join(prot_ch_l)}__{"_".join(lig_ch_l)}'

        annotations_out.append(f'{outname},2026-04-07')
        print(outname)

        # Save outputs:
        case_outdir = f'{structout}/{outname}'
        os.makedirs(case_outdir, exist_ok=True)
        os.makedirs(f'{case_outdir}/ligand_files', exist_ok=True)

        cmd.save(f'{case_outdir}/system.cif')
        cmd.save(f'{case_outdir}/receptor.cif', 'rec')
        for lc in lig_ch_l:
            cmd.save(f'{case_outdir}/ligand_files/{lc}.sdf', f'lig and chain {lc}')

    with open(f'{args.outdir}/annotations.csv', 'w') as fo:
        fo.write('\n'.join(annotations_out))


if __name__=='__main__':
    main()
