INP_PDB=$1
INP_SDF=$2
OUT_DIR=$3
#FOLDSEEK_DB=/lus/lfs1aip2/scratch/s5h/omeir.s5h/databases/foldseek_pdb100/pdb
FOLDSEEK_DB=/lus/lfs1aip2/scratch/s5h/omeir.s5h/databases/plinder/receptor_db/plinder_db

mkdir $OUT_DIR
pdb_n=$(basename ${INP_PDB})
foldseek_aln_out=${OUT_DIR}/foldseek_aln_${pdb_n}.tsv
mmseqs_aln_out=${OUT_DIR}/mmseqs_aln_${pdb_n}.tsv

foldseek easy-search ${INP_PDB} ${FOLDSEEK_DB} ${foldseek_aln_out} ${OUT_DIR}/tmp/ --format-mode 4 --format-output query,target,qaln,taln,qstart,tstart,qend,tend,evalue,pident,qseq,tseq,qcov,u,t -v 1 > ${OUT_DIR}/log_foldseek_search_${pdb_n}.log

PDB_FA=${OUT_DIR}/rec.fa
python3 util_Py_pdb_to_fasta.py ${INP_PDB} ${PDB_FA}

mmseqs easy-search ${PDB_FA} ${FOLDSEEK_DB} ${mmseqs_aln_out} ${OUT_DIR}/tmp/ --format-output query,target,qaln,taln,qstart,tstart,qend,tend,evalue,pident,qseq,tseq,qcov --format-mode 4 -v 1 > ${OUT_DIR}/log_mmseqs_search_${pdb_n}.log

python get_d3i_alignment.py -i ${foldseek_aln_out} --tsv-out --format-output query,target,qaln,taln,qstart,tstart,qend,tend,evalue,pident,qseq,tseq,qcov,u,t > ${OUT_DIR}/d3i_foldseek_aln_${pdb_n}.tsv

python3 get_d3i_alignment.py -i ${mmseqs_aln_out} --tsv-out --format-output query,target,qaln,taln,qstart,tstart,qend,tend,evalue,pident,qseq,tseq,qcov > ${OUT_DIR}/d3i_mmseqs_aln_${pdb_n}.tsv

python3 Py_calc_pocket_qcov.py -i_foldseek=${OUT_DIR}/d3i_foldseek_aln_${pdb_n}.tsv -i_mmseqs=${OUT_DIR}/d3i_mmseqs_aln_${pdb_n}.tsv -pdb=${INP_PDB} -sdf=${INP_SDF} -o=${OUT_DIR}/qcov_${pdb_n}.tsv

python3 Py_calc_sucos.py -q=${OUT_DIR}/qcov_${pdb_n}.tsv  -sdf=${INP_SDF} -o=${OUT_DIR}/sucos_qcov_${pdb_n}.tsv

#rm ${OUT_DIR}/qcov_${pdb_n}.tsv

