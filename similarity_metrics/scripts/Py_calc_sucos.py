## Adapt the sucos calculation code from the Runs N Poses publication

import argparse
import os
import glob
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem, rdShapeAlign, rdShapeHelpers
from rdkit.Chem.FeatMaps import FeatMaps

parser = argparse.ArgumentParser()

parser.add_argument('--qcov_file', '-q', help='tsv file with pocket_qcov values')
parser.add_argument('--lig_sdf', '-sdf', help='.sdf file with the query ligand')
parser.add_argument('--outfile', '-o', help='Name of the output .tsv file')
parser.add_argument('--custom', '-c', help='Provide a path to a custom database (i.e. not PLINDER). The data should be formatted like PLINDER. i.e, it contains with a structures/ parent directory, with subdirectories named after each system in the set. Each subdirectory should contain: receptor.cif, sequences.fasta, system.cif, and a ligand_files/ folder with sdf files for all ligands in the system', default=None)

args = parser.parse_args()

if args.custom == None:
    from plinder.core import PlinderSystem

# Adapted from https://github.com/susanhleung/SuCOS
# Initialize feature factory for pharmacophore scoring
FDEF = AllChem.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
)

# Feature map parameters
FEAT_MAP_PARAMS = {k: FeatMaps.FeatMapParams() for k in FDEF.GetFeatureFamilies()}

# Feature types to keep for pharmacophore scoring
PHARMACOPHORE_FEATURES = (
    "Donor",
    "Acceptor",
    "NegIonizable",
    "PosIonizable",
    "ZnBinder",
    "Aromatic",
    "Hydrophobe",
    "LumpedHydrophobe",
)

def get_feature_map_score(
    mol_1: Chem.Mol,
    mol_2: Chem.Mol,
    score_mode: FeatMaps.FeatMapScoreMode = FeatMaps.FeatMapScoreMode.All,
) -> float:
    feat_lists = []
    for molecule in [mol_1, mol_2]:
        raw_feats = FDEF.GetFeaturesForMol(molecule)
        feat_lists.append([
            f for f in raw_feats if f.GetFamily() in PHARMACOPHORE_FEATURES
        ])

    feat_maps = [
        FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=FEAT_MAP_PARAMS)
        for x in feat_lists
    ]
    feat_maps[0].scoreMode = score_mode

    score = feat_maps[0].ScoreFeats(feat_lists[1])
    return score / min(feat_maps[0].GetNumFeatures(), len(feat_lists[1]))

def get_sucos_score(
    mol_1: Chem.Mol,
    mol_2: Chem.Mol,
    score_mode: FeatMaps.FeatMapScoreMode = FeatMaps.FeatMapScoreMode.All,
) -> float:
    fm_score = get_feature_map_score(mol_1, mol_2, score_mode)
    fm_score = np.clip(fm_score, 0, 1)

    protrude_dist = rdShapeHelpers.ShapeProtrudeDist(
        mol_1, mol_2, allowReordering=False
    )
    protrude_dist = np.clip(protrude_dist, 0, 1)

    return 0.5 * fm_score + 0.5 * (1 - protrude_dist)

def align_molecules_crippen(mol_ref, mol_probe, iterations=100):
    crippenO3A = Chem.rdMolAlign.GetCrippenO3A(mol_probe, mol_ref, maxIters=iterations)
    crippenO3A.Align()


def align_molecules(
    reference: Chem.Mol,
    mobile: Chem.Mol,
    max_preiters: int = 100,
    max_postiters: int = 100,
) -> Tuple[float, float, np.ndarray]:
    align_molecules_crippen(reference, mobile, iterations=max_preiters)
    return rdShapeAlign.AlignMol(
        reference,
        mobile,
        max_preiters=max_preiters,
        max_postiters=max_postiters,
    )

def main():
    df = pd.read_csv(args.qcov_file, delimiter='\t')
    err_log = []

    #outlines = ['query\ttarget\ttarget_rec_chain\tligand_chain\trelease_date\taln_evalue\tpocket_qcov\tsucos_protein\tsucos_shape\tsucos_shape_pocket_qcov']
    outlines = ['query\ttarget\ttarget_rec_chain\tligand_chain\trelease_date\taln_evalue\tpocket_qcov\tsucos_shape\tsucos_shape_pocket_qcov']

    
    for i, target in enumerate(df['target']):
        query = df['query'].iloc[i]
        rec_ch = df['target_rec_chain'].iloc[i]
        lig_ch = df['ligand_chain'].iloc[i]
        pocket_qcov = df['pocket_qcov'].iloc[i]
        rls_date = df['release_date'].iloc[i]
        evalue = df['aln_evalue'].iloc[i]
        u_mtx = df['rot_mtx'].iloc[i]
        t_vec = df['trans_vec'].iloc[i]

        print(i, target, rls_date)
        print(f'\tPocket_qcov: {pocket_qcov}')

        if args.custom == None:
            plinder_system = PlinderSystem(system_id=target)
            target_sdfs = plinder_system.ligand_sdfs
            target_sdf = target_sdfs[lig_ch]
        else:
            target_sdf = f'{args.custom}/structures/{target}/ligand_files/{lig_ch}.sdf'
        
        # Apply the foldseek alignment
        '''
        try:
            q_mol = Chem.MolFromMolFile(args.lig_sdf)
            t_mol = Chem.MolFromMolFile(target_sdf)
            print(lig_ch, target_sdf, Chem.MolToSmiles(t_mol))
            rotation = np.array(list(map(float, u_mtx.split(','))))
            translation = np.array(list(map(float, t_vec.split(','))))

            conf = t_mol.GetConformer()
            coords = np.array([
                list(conf.GetAtomPosition(i))
                for i in range(t_mol.GetNumAtoms())
            ])
            rotated_coords = coords @ rotation.reshape(3, 3).T + translation
            for i in range(t_mol.GetNumAtoms()):
                conf.SetAtomPosition(i, rotated_coords[i])
            sucos = get_sucos_score(q_mol, t_mol)
            sucos_pocket_protein = sucos*(pocket_qcov/100)
            sucos_protein = sucos*100
            sucos_pocket_protein = sucos_pocket*100
        except Exception as e:
            err_log.append(f'sucos_protein error for: {target} {lig_ch} {Chem.MolToSmiles(t_mol)}:\n')
            err_log.append(str(e)+ '\n')
            sucos_protein = np.nan
            sucos_pocket_protein = np.nan
        ''' 

        # Use the ligand-based alignment to calculate SuCOS
        q_mol = Chem.MolFromMolFile(args.lig_sdf)
        t_mol = Chem.MolFromMolFile(target_sdf)
        try:
            shape_similarity, color_similarity = align_molecules(q_mol, t_mol)
            print(f'\tShape: {shape_similarity}\tColor: {color_similarity}')
        except Exception as e:
            try:
                err_log.append(f'Alignment error for: {target} {lig_ch} {Chem.MolToSmiles(t_mol)}:\n')
                err_log.append(str(e)+ '\n')
            except:
                err_log.append(f'Alignment error for: {target} {lig_ch} CANNOT_RESOLVE_SMILES:\n')
                err_log.append(str(e)+ '\n')
            shape_similarity = np.nan
            color_similarity = np.nan

        try:
            sucos = get_sucos_score(q_mol, t_mol)
            print(f'\tSuCOS: {sucos}')
            sucos_pocket_shape = sucos*(pocket_qcov/100)
            sucos_shape = sucos*100
            sucos_pocket_shape = sucos_pocket_shape*100
            print(f'\tSuCOS_pocket: {sucos_pocket_shape}')
        except Exception as e:
            err_log.append(f'Alignment error for: {target} {lig_ch} {Chem.MolToSmiles(t_mol)}:\n')
            err_log.append(str(e)+ '\n')
            sucos_shape = np.nan
            sucos_pocket_shape = np.nan
        
        print(sucos_pocket_shape)
        outlines.append(f'{query}\t{target}\t{rec_ch}\t{lig_ch}\t{rls_date}\t{evalue}\t{pocket_qcov}\t{sucos_shape}\t{sucos_pocket_shape}')

    #print('\t', pocket_qcov, sucos, sucos)
        #outlines.append(f'{query}\t{target}\{rec_ch}\t{lig_ch}\t{rls_date}\t{evalue}\t{pocket_qcov}\t{sucos_pocket_protein}\t{sucos_shape}\t{sucos_pocket_shape}')

    with open(args.outfile, 'w') as fo:
        fo.write('\n'.join(outlines))

    err_out = os.path.basename(args.outfile).split('.')
    err_out = '.'.join(err_out[:-1]) + '.err'

    with open(f'{os.path.dirname(os.path.abspath(args.outfile))}/{err_out}', 'w') as fo:
        fo.write(''.join(err_log))



if __name__=='__main__':
    main()
