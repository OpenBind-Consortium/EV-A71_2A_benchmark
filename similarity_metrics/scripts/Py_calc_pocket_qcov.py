import os
import json
import argparse
import pandas as pd
import datetime
from Bio import AlignIO
from pymol import cmd, stored

TRAIN_CUTOFF = datetime.datetime(2021, 9, 30)

parser = argparse.ArgumentParser()

parser.add_argument('--d3i_tsv_foldseek', '-i_foldseek', help='d3i alignment file in tsv format')
parser.add_argument('--d3i_tsv_mmseqs', '-i_mmseqs', help='mmseqs d3i alignment file in tsv format')
parser.add_argument('--query_pdb', '-pdb', help='Path to the .pdb file used to run the foldseek search')
parser.add_argument('--query_lig', '-sdf', help='Path to an .sdf with the ligand bount to the query pdb')
parser.add_argument('--outfile', '-o', help='Name of the output .tsv file')
parser.add_argument('--custom', '-c', help='Provide a path to a custom database (i.e. not PLINDER). The data should be formatted like PLINDER. i.e, it contains with a structures/ parent directory, with subdirectories named after each system in the set. Each subdirectory should contain: receptor.cif, sequences.fasta, system.cif, and a ligand_files/ folder with sdf files for all ligands in the system', default=None)

args = parser.parse_args()

if args.custom == None:
    import plinder.core.utils.config
    from plinder.core.scores import query_index
    from plinder.core import PlinderSystem
    cfg = plinder.core.get_config()
    print(f"PLINDER local cache directory: {cfg.data.plinder_dir}")
else:
    print(f'Using custom structure database!')
    ANNOTATION_DF = pd.read_csv(f'{args.custom}/annotations.csv')
    ANNOTATION_DATA = {}

    for i, target in enumerate(ANNOTATION_DF['system_id']):
        rls_date = ANNOTATION_DF['release_date'].iloc[i]

        if target not in ANNOTATION_DATA:
            ANNOTATION_DATA[target] = rls_date
            print(target, rls_date)



def get_pocket_resis(rec_sele, lig_sele, cutoff=6.0):
    stored.resi_l = []

    cmd.select('pocket', f'({rec_sele}) and polymer.protein within {cutoff} of ({lig_sele})')
    cmd.iterate('pocket', 'stored.resi_l.append((resi))')

    resi_l = []
    for r in stored.resi_l:
        if int(r) not in resi_l:
            resi_l.append(int(r))

    return resi_l

# Get index of the first and last receptor residues
def get_first_resi(sele):
    stored.all_resi = []
    cmd.iterate(sele, 'stored.all_resi.append(resi)')
    all_resi = list(set(stored.all_resi))
    tmp = list(map(int, all_resi))
    first_resi = min(tmp)
    last_resi = max(tmp)
    #print(first_resi)
    return first_resi, last_resi

# Get list of query sequence residues, and target sequence residues
# that align with the binding site of the query pdb file
def get_d3i_aln_resis(d3i_tsv_foldseek, d3i_tsv_mmseqs, pocket_resi_l, q_first_resi, rec_sele):

    # Read foldseek alignment data
    df = pd.read_csv(d3i_tsv_foldseek, delimiter='\t')
    aln_data = {}
    for i, query in enumerate(df['query']):
        target  = df['target'].iloc[i]
        qaln = df['qaln'].iloc[i]
        taln = df['taln'].iloc[i]
        maln = df['midline'].iloc[i]
        evalue = float(df['evalue'].iloc[i])
        qstart = int(df['qstart'].iloc[i])
        tstart = int(df['tstart'].iloc[i])
        qcov = float(df['qcov'].iloc[i])

        u = df['u'].iloc[i]
        t = df['t'].iloc[i]

        pdbstart = qstart+(q_first_resi-1)

        #print(f'q pdbstart {target}', pdbstart)
        #print(qaln)
        #print(maln)
        #print(taln)
        #print(evalue) # Add an e-value cutoff?

        #if evalue >= 0.001:
        #    continue
        #print(query, target, evalue)

        if target not in aln_data:
            aln_data[target] = {'q_aln_map': [],
                                'pdb_aln_map': [],
                                't_aln_map': [],
                                'qaln': qaln,
                                'maln': maln,
                                'taln': taln,
                                'qstart': qstart,
                                'tstart': tstart,
                                'query': query,
                                'evalue': evalue,
                                'qcov': qcov,
                                'pdbstart': pdbstart,
                                'u': u,
                                't': t,
                                'method': 'foldseek'}

    # Read mmseqs alignment data, replace foldseek data if 
    # qcov is larger
    df = pd.read_csv(d3i_tsv_mmseqs, delimiter='\t')
    for i, query in enumerate(df['query']):
        target  = df['target'].iloc[i]
        qaln = df['qaln'].iloc[i]
        taln = df['taln'].iloc[i]
        maln = df['midline'].iloc[i]
        evalue = float(df['evalue'].iloc[i])
        qstart = int(df['qstart'].iloc[i])
        tstart = int(df['tstart'].iloc[i])
        qcov = float(df['qcov'].iloc[i])

        pdbstart = qstart+(q_first_resi-1)

        replace_data = False
        try:
            if (qcov  > aln_data[target]['qcov']):
                replace_data = True
        except:
            pass

        if (target not in aln_data) or (replace_data == True):
            # Debug:
            #if (replace_data):
            #    print(f'{target} MMSEQS qcov is better: {qcov} > {aln_data[target]["qcov"]}')
            #else:
            #    print(f'{target} only appears in MMSEQS alignment')

            aln_data[target] = {'q_aln_map': [],
                                'pdb_aln_map': [],
                                't_aln_map': [],
                                'qaln': qaln,
                                'maln': maln,
                                'taln': taln,
                                'qstart': qstart,
                                'tstart': tstart,
                                'query': query,
                                'evalue': evalue,
                                'qcov': qcov,
                                'pdbstart': pdbstart,
                                'u': None,
                                't': None,
                                'method': 'mmseqs'}

    # Iterate through all aligned targets
    for target in aln_data:
    #for target in ['rec__2zjf__1__1.A__1.E__1.A']: #Debug
        # Check for matching *aligned* residue positions
        #q_pmap = plinder_seqmap(rec_sele, aln_data[target]['qaln'], aln_data[target]['qstart'], aln_data[target]['pdbstart'])
        aln_resis = []
        q_incr = 0
        t_incr = 0
        for j in range(len(aln_data[target]['qaln'])):
            if aln_data[target]['maln'][j] != ' ':
                q_resi = aln_data[target]['qstart'] + q_incr
                t_resi = aln_data[target]['tstart'] + t_incr
                pdb_resi = aln_data[target]['pdbstart'] + q_incr

                #if q_resi in pocket_resi_l:
                if pdb_resi in pocket_resi_l:
                    #print(f'\t{target} Q_resi:', q_resi, aln_data[target]['maln'][j], 'T_resi', t_resi)
                    aln_resis.append(q_resi)
                    aln_data[target]['q_aln_map'].append(q_resi)
                    aln_data[target]['t_aln_map'].append(t_resi)
                    aln_data[target]['pdb_aln_map'].append(pdb_resi)

            if aln_data[target]['qaln'][j] != '-':
                q_incr += 1
            if aln_data[target]['taln'][j] != '-':
                t_incr += 1
        
    return aln_data

# Map residue indices in the foldseek sequence to residues 
# indices in the receptor PDB
def plinder_seqmap(rec_sele, taln, tstart, pstart):
    print(taln)
    tseq = ''
    for aa in taln:
        #tseq += aa
        if aa != '-':
            tseq += aa
    #print(tseq)

    resi_first, resi_last = get_first_resi(rec_sele)

    stored.resi_data = {}
    cmd.iterate(rec_sele, 'stored.resi_data[int(resi)] = oneletter')
    
    #print(stored.resi_data)

    # Add gaps to the dict
    for i in range(resi_first, resi_last+1):
        if i not in stored.resi_data:
            stored.resi_data[i] = '-'
        #print(i, stored.resi_data[i])
    
    pmap = {} # Map PDB sequence to aln sequence index
    t_i = tstart
    p_i = pstart
    for i, aa in enumerate(tseq):
        t_i = tstart + i
        #print(aa, t_i, p_i, tseq[i], stored.resi_data[p_i])

        while tseq[i] != stored.resi_data[p_i]:
            # Could edit here to allow for nonstandard AA (?) to match?
            p_i += 1
            #print('\tmismatch!', aa, t_i, p_i, tseq[i], stored.resi_data[p_i])
        
        pmap[p_i] = t_i
        p_i += 1


    return pmap   

# Take the aligned target sequence, and find the starting index
# in the protein sequence
def map_plinder_seq_to_d3i(rec_sele, plinder_seq, taln, tstart):
    
    # Get non-gapped sequence for the target
    tseq = ''
    for aa in taln:
        if aa != '-':
            tseq += aa
    print('aln_seq', tseq)
    pstart = plinder_seq.find(tseq)
    if pstart == -1:
        #print('ERROR: No mapping found between pdb100 sequence and plinder sequence :(')
        return None

    resi_first, resi_last = get_first_resi(rec_sele)
    #pdbstart = qstart+(q_first_resi-1)
    pstart = pstart + (resi_first) # Starting index in the PDB file

    #print('\t',tstart, pstart+1)
    #print('\t',tstart, pstart)
    
    return pstart


def main():
    outlines = ['query\ttarget\ttarget_rec_chain\tligand_chain\trelease_date\tpocket_qcov\taln_evalue\trot_mtx\ttrans_vec\taln_method']
    err_log = []

    # Load the query system
    cmd.reinitialize()
    cmd.load(args.query_pdb, 'qrec')
    cmd.load(args.query_lig, 'qlig')
    q_pocket_resis = get_pocket_resis('qrec and polymer.protein', 'qlig')
    q_pocket_size = len(q_pocket_resis)
    print(f'Q pocket residues: {q_pocket_resis}')
    print(f'\t{q_pocket_size} residues available')
    q_first_resi, q_last_resi = get_first_resi('qrec and polymer.protein')
    aln_data = get_d3i_aln_resis(args.d3i_tsv_foldseek, args.d3i_tsv_mmseqs, q_pocket_resis, q_first_resi, 'qrec')
    cmd.reinitialize()
    #return #Debug

    # Search for targets in PLINDER:

    for target in aln_data:
    #for target in ['rec__1iup__1__1.A__1.B__1.A']: #Debug
    #for target in ['rec__2zjf__1__1.A__1.E__1.A']: #Debug
        
        if args.custom == None:
            t_data = target.split('__')
            t_system_id = '__'.join(t_data[1:-1])
            t_chain = t_data[-1]
            print(target, t_system_id, t_chain)
            try:
                plinder_system = PlinderSystem(system_id=t_system_id)
                entry_annotations = plinder_system.entry
                rls_date = entry_annotations['release_date']
                rls_date = datetime.datetime.strptime(rls_date, "%Y-%m-%d")
            except Exception as e:
                err_log.append(f'{t_system_id} annotations not found for this PLINDER system?')
                continue
        
            # Plinder sequence appears to be different from the plinder receptor PDB.
            # Create a fasta for the protein from the plinder receptor.pdb
            entry_path = os.path.dirname(plinder_system.receptor_pdb)
            chain_mapping = f'{entry_path}/chain_mapping.json'
            with open(chain_mapping) as f:
                cmap = json.load(f)

            pdb_chain = cmap[t_chain]
            cmd.reinitialize()
            cmd.load(plinder_system.receptor_pdb, 'rec')
        else:
            # Gotta do some weird string manipulatiosn for RnP :s
            # Deepseek adds _{rec_chain} to the end of the target name, 
            # which conflicts with the PLINDER naming scheme
            t_data = target.split('__')

            rec_chains = t_data[3].split('_')
            lig_chains = t_data[4].split('_')
            
            t_system_id = '__'.join(t_data[1:4]) + '__'
            for llc in lig_chains:
                print(llc, rec_chains)
                if llc not in rec_chains:
                    t_system_id += f'{llc}_'
            
            t_system_id = t_system_id[:-1]
            
            if len(rec_chains) == 1:
                t_chain = rec_chains[0]
            else:
                t_chain = t_data[-1].split('_')[0]

            print(target, t_system_id, t_chain)
            try:
                rls_date = ANNOTATION_DATA[t_system_id]
                rls_date = datetime.datetime.strptime(rls_date, "%Y-%m-%d")
            except:
                rls_date = None
            print(t_system_id, t_system_id in ANNOTATION_DATA, rls_date)

            rec_pdb = f'{args.custom}/structures/{t_system_id}/system.cif'
            cmd.reinitialize()
            cmd.load(rec_pdb, 'rec')
            pdb_chain = t_chain

            ## END IF ##


        rec_seq = cmd.get_fastastr(f'rec and chain {pdb_chain}')
        rec_seq = ''.join(rec_seq.split('\n')[1:])
        #print('pdb_seq', rec_seq)
        pstart = map_plinder_seq_to_d3i(f'rec and chain {pdb_chain}', rec_seq, aln_data[target]['taln'], aln_data[target]['tstart'])
        
        if pstart == None:
            err_log.append(f'{target} pocket_qcov failed.\n\tt_aln: {aln_data[target]["taln"]}\n\tp_seq: {rec_seq}')
            continue

        # Map plinder taln sequence to pdb sequence
        t_pmap = plinder_seqmap(f'(rec and polymer.protein and chain {pdb_chain})', aln_data[target]['taln'], aln_data[target]['tstart'], pstart)

        #if pstart != aln_data[target]['tstart']:
        #    raise ValueError('pstart tstart mismatch!')
        
        # get pocket_residues for plinder ligandss
        if args.custom == None:
            ligand_sdfs = plinder_system.ligand_sdfs
        else:
            ligand_sdfs = {}
            for lsdf in os.listdir(f'{args.custom}/structures/{t_system_id}/ligand_files/'):
                lc = os.path.basename(lsdf)[:-4]
                print(lc, lsdf)
                ligand_sdfs[lc] = f'{args.custom}/structures/{t_system_id}/ligand_files/{lsdf}'
                

        for lc in ligand_sdfs:
            cmd.load(ligand_sdfs[lc], f'lig-{lc}')
            lc_pocket_resis = get_pocket_resis(f'(rec and polymer.protein and chain {pdb_chain})', f'lig-{lc}')
            #print(f'T-{lc} pocket resis:', lc_pocket_resis)

            t_pocket_resis = []
            for resi in lc_pocket_resis:
                try:
                    #print(f'\tT Pocket Convert: {resi} -> {t_pmap[resi]}')
                    t_pocket_resis.append(t_pmap[resi])
                except:
                    print(f'Pocket resi {resi} not included in the foldseek alignment :S')
                    continue


            #print(f'\t{lc} T   pocket:', t_pocket_resis)
            #print(f'\t{lc} pdb pocket:', lc_pocket_resis)
            #print('\tQ aln to Q pocket:', aln_data[target]['q_aln_map'], len(aln_data[target]['q_aln_map']))
            #print('\tT aln to Q pocket:', aln_data[target]['t_aln_map'], len(aln_data[target]['t_aln_map']))
            #print('\tpdb aln to Q pocket:', aln_data[target]['pdb_aln_map'], len(aln_data[target]['pdb_aln_map']))
            
            t_q_overlap = []
            for resi in t_pocket_resis:
                if resi in aln_data[target]['t_aln_map']:   
                    t_q_overlap.append(resi)

            #print(f'\t{lc} T pocket-Q pocket overlap', t_q_overlap)
            n_t_aln_overlap = len(t_q_overlap)
            #print(f'{n_t_aln_overlap} residues in the T binding site align to the Q binding site')

            pocket_qcov = n_t_aln_overlap*100/q_pocket_size

            
            print(f'\t{t_system_id} {lc} pocket_qcov: {pocket_qcov}')

            outlines.append(f'{aln_data[target]["query"]}\t{t_system_id}\t{t_chain}\t{lc}\t{rls_date}\t{pocket_qcov}\t{aln_data[target]["evalue"]}\t{aln_data[target]["u"]}\t{aln_data[target]["t"]}\t{aln_data[target]["method"]}')

    with open(args.outfile, 'w') as fo:
        fo.write('\n'.join(outlines))
    
    
    err_file = '/'.join(os.path.abspath(args.outfile).split('/')[:-1]) + f'/{os.path.basename(args.outfile)[:-4]}.err' 
    
    #print(err_file)
    with open(err_file, 'w') as fo:
        fo.write('\n'.join(err_log))

    #with open(f'json_aln_test.json', 'w') as fo:
    #    json.dump(aln_data, fo, indent=4)


if __name__=='__main__':
    main()
