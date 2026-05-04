#!/usr/bin/env python3
import argparse
import sys

# All amino acid pairs with positive BLOSUM62 score (excluding identity).
# Extracted from foldseek's frontend JS (data/main.js line 1024).
POSITIVE_PAIRS = {
    "AG", "AS", "DE", "DN", "ED", "EK", "EQ", "FL", "FM", "FW", "FY",
    "GA", "HN", "HQ", "HY", "IL", "IM", "IV", "KE", "KQ", "KR",
    "LF", "LI", "LM", "LV", "MF", "MI", "ML", "MV", "ND", "NH", "NQ", "NS",
    "QE", "QH", "QK", "QN", "QR", "RK", "RQ", "SA", "SN", "ST",
    "TS", "VI", "VL", "VM", "WF", "WY", "YF", "YH", "YW",
}


def compute_midline(qaln, taln):
    """Compute the alignment midline string.

    Returns a string where each position is:
      - the residue letter  if query == target (exact match)
      - '+'                 if the pair has a positive BLOSUM62 score (similar)
      - ' '                 otherwise (mismatch or gap)
    """
    mid = []
    for q, t in zip(qaln, taln):
        if q == "-" or t == "-":
            mid.append(" ")
        elif q == t:
            mid.append(q)
        elif q + t in POSITIVE_PAIRS:
            mid.append("+")
        else:
            mid.append(" ")
    return "".join(mid)


def format_alignment(qaln, taln, midline, qstart, tstart, line_len=80):
    """Format a pairwise alignment block like the foldseek HTML viewer."""
    lines = []
    qpos = int(qstart)
    tpos = int(tstart)
    for offset in range(0, len(qaln), line_len):
        q_chunk = qaln[offset:offset + line_len]
        m_chunk = midline[offset:offset + line_len]
        t_chunk = taln[offset:offset + line_len]

        q_residues = sum(1 for c in q_chunk if c != "-")
        t_residues = sum(1 for c in t_chunk if c != "-")

        pad = max(len(str(qpos + q_residues)), len(str(tpos + t_residues))) + 1
        lines.append(f"Q {qpos:<{pad}} {q_chunk}")
        lines.append(f"  {'':<{pad}} {m_chunk}")
        lines.append(f"T {tpos:<{pad}} {t_chunk}")
        lines.append("")

        qpos += q_residues
        tpos += t_residues
    return "\n".join(lines)


def parse_tsv(path, fmt_fields):
    """Yield rows as dicts keyed by format field names."""
    with open(path) as f:
        next(f) # Skip header
        for line in f:
            vals = line.rstrip("\n").split("\t")
            yield dict(zip(fmt_fields, vals))


def main():
    parser = argparse.ArgumentParser(
        description="Compute alignment midlines from foldseek convertalis output."
    )
    parser.add_argument("--tsv", '-i', help="TSV file from foldseek convertalis. Should be generated with --format-mode 4 if using easy-search")
    parser.add_argument("--outfile", '-o', help="TSV file from foldseek convertalis")
    parser.add_argument(
        "--format-output",
        default="query,target,qaln,taln,qstart,tstart,qend,tend,evalue,pident",
        help="Comma-separated field names matching --format-output used in convertalis "
             "(default: query,target,qaln,taln,qstart,tstart,qend,tend,evalue,pident)",
    )
    parser.add_argument(
        "--line-len", type=int, default=80,
        help="Characters per alignment line (default: 80)",
    )
    parser.add_argument(
        "--tsv-out", action="store_true",
        help="Output as TSV (query, target, qaln, midline, taln) instead of visual alignment",
    )
    args = parser.parse_args()

    fmt_fields = [f.strip() for f in args.format_output.split(",")]
    for required in ("qaln", "taln"):
        if required not in fmt_fields:
            print(f"Error: '{required}' must be in --format-output", file=sys.stderr)
            sys.exit(1)
    
    all_out = []

    if args.tsv_out:
        print(f'query\ttarget\tqstart\tqaln\tqend\tmidline\ttstart\ttaln\ttend\tevalue\tpident\tqcov\tu\tt')
    for row in parse_tsv(args.tsv, fmt_fields):
        qaln = row["qaln"]
        taln = row["taln"]
        qstart = row["qstart"]
        tstart = row["tstart"]
        
        qend = row["qend"]
        tend = row["tend"]
        evalue = row["evalue"]
        pident = row["pident"]
        qcov = row["qcov"]

        try:
            u = row["u"]
            t = row["t"]
        except:
            pass

        midline = compute_midline(qaln, taln)
        
        if args.tsv_out:
            try:
                print("\t".join([
                    row.get("query", ""),
                    row.get("target", ""),
                    qstart,
                    qaln,
                    qend,
                    midline,
                    tstart,
                    taln,
                    tend,
                    evalue,
                    pident,
                    qcov,
                    u,
                    t
                ]))
            except:
                print("\t".join([
                    row.get("query", ""),
                    row.get("target", ""),
                    qstart,
                    qaln,
                    qend,
                    midline,
                    tstart,
                    taln,
                    tend,
                    evalue,
                    pident,
                    qcov
                ]))
        else:
            query = row.get("query", "?")
            target = row.get("target", "?")
            evalue = row.get("evalue", "")
            pident = row.get("pident", "")
            qstart = row.get("qstart", "0")
            tstart = row.get("tstart", "0")

            header = f">{query} vs {target}"
            if evalue:
                header += f"  E={evalue}"
            if pident:
                header += f"  id={pident}"
            
            outlines = format_alignment(qaln, taln, midline, qstart, tstart, args.line_len)
            print(header)
            print(outlines)
            all_out.append(header)
            all_out += outlines




if __name__ == "__main__":
    main()
