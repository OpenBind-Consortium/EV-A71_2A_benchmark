# pocket_sucos_code
Code for calculating sucos-pocket similarity from an arbitrary protein-ligand complex. Currently, similarity metrics are
calculated with respect to PLINDER systems. We note that the PLINDER cutoff date is June 2024. 

### Setup the conda environment:
```
conda env create -f environment.yml
```

### Install the PLINDER database, and set the following environment variables:

In order to run the code, you must install the [PLINDER database](https://github.com/plinder-org/plinder).

Set the following environment variables to ensure you are installing PLINDER in a specified local directory.

```
export PLINDER_MOUNT=/path/to/your/plinder/install
export PLINDER_OFFLINE=true
export PLINDER_ITERATION=v2
export PLINDER_RELEASE=2024-06
```

### Create a custom foldseek database from PLINDER receptor structures.

Copy receptor.cif structures installed in the `$PLINDER_MOUNT/plinder/2024-06/v2/systems/`, to a new directory:

```
python3 Py_create_plinder_rec_db.py $PLINDER_MOUNT/plinder/2024-06/v2/systems/ receptor_cifs/
```


Then run the `foldseek createdb` command as specified in the [Foldseek documantation](https://github.com/steineggerlab/foldseek):

```
mkdir receptor_db
cd receptor_db
foldseek createdb ../receptor_cifs plinder_db
```

### To run:

Edit the `FOLDSEEK_DB` variable in `01_Sh_run_foldseek.sh` to point to the path of your custom foldseek database
```
bash 01_Sh_run_foldseek.sh $RECEPTOR_PDB $LIGAND_SDF $OUTDIR
```

### Known Issues:

*Foldseek alignment fails for chains with nonstandard amino acid residues

*Code only designed for single chain proteins

*In internal tests, pocket_qcov values are sometimes inconsistent with those reported in the [Runs N Poses Zenodo page](https://zenodo.org/records/18366081). We suspect this is because the packages used for pocket residue selection may be different.

*PLINDER occaisionally fails to find system annotations for structures within the database.
