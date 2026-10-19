"""
Stage 2b: reconstruct the 125-nt SHAPE-MaP minigene constructs.

GSE254361 probed synthetic 125-nt minigenes in which the pri-miRNA hairpin sits
in the middle, so the reactivity vector is indexed on the *construct*, not on the
miRBase hairpin. Stage 3 needs the construct sequence to build features.

This script:
  1. reads hairpin (miRNA_primary_transcript) coordinates from miRBase hsa.gff3
  2. pulls the GRCh38 sequence for a 125-nt window centred on each hairpin
     through the Ensembl REST API (cached on disk, so re-runs are free)
  3. locates the miRBase hairpin inside that window to get the hairpin offset
  4. writes construct sequences and a mapping table for stage 3
"""
import csv
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RAW = os.path.join(ROOT, "data", "raw")
MIR = os.path.join(ROOT, "data", "mirbase")
CACHE = os.path.join(RAW, "genome_cache")
os.makedirs(CACHE, exist_ok=True)

REST = "https://rest.ensembl.org/sequence/region/human"
CONSTRUCT_LEN = 125
PAD = 30                      # extra sequence fetched each side, for offset search


def log(msg):
    print(f"[construct] {msg}", flush=True)


def read_fasta(path):
    """Yield (id, sequence) from a FASTA file, uppercased with T->U."""
    cur, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if cur is not None:
                    yield cur, "".join(buf).upper().replace("T", "U")
                cur, buf = line[1:].split()[0], []
            elif line:
                buf.append(line)
    if cur is not None:
        yield cur, "".join(buf).upper().replace("T", "U")


def read_gff3_hairpins(path):
    """mirbase_id -> (chrom, start, end, strand) for miRNA_primary_transcript."""
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "miRNA_primary_transcript":
                continue
            attrs = dict(kv.split("=", 1) for kv in f[8].split(";") if "=" in kv)
            mid = attrs.get("Name")
            if mid:
                out[mid] = (f[0], int(f[3]), int(f[4]), f[6])
    return out


def revcomp_rna(s):
    """Reverse complement of an RNA string, used for minus-strand loci."""
    return s[::-1].translate(str.maketrans("ACGUN", "UGCAN"))


def fetch(chrom, start, end):
    """GRCh38 sequence for a 1-based inclusive interval, cached as JSON.
    Genomic (forward) strand is always returned."""
    key = f"{chrom}_{start}_{end}.json"
    p = os.path.join(CACHE, key)
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)["seq"].upper().replace("T", "U")
    url = f"{REST}/{chrom}:{start}-{end}?content-type=text/plain"
    last = None
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"Accept": "text/plain"})
            seq = urllib.request.urlopen(req, timeout=90).read().decode().strip()
            with open(p, "w") as fh:
                json.dump({"chrom": chrom, "start": start, "end": end, "seq": seq}, fh)
            return seq.upper().replace("T", "U")
        except Exception as exc:                     # rate limit / transient network
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"fetch failed {chrom}:{start}-{end}: {last}")


def main():
    hp_seq = dict(read_fasta(os.path.join(MIR, "hairpin.fa")))
    coords = read_gff3_hairpins(os.path.join(MIR, "hsa.gff3"))
    log(f"miRBase hairpins: {len(hp_seq)}; GRCh38 hairpin loci: {len(coords)}")

    with open(os.path.join(PROC, "sequence_mapping.tsv")) as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t")
                if r["dataset"] == "SHAPE" and r.get("mirbase_id")]
    targets = {r["mirbase_id"] for r in rows}
    log(f"SHAPE miRNAs needing constructs: {len(targets)}")

    out, stats = [], {"ok": 0, "no_locus": 0, "no_match": 0, "too_long": 0, "fetch_fail": 0}
    offsets = []

    # resume from a previous partial run if present
    done = {}
    if os.path.exists(os.path.join(PROC, "shape_constructs.tsv")):
        with open(os.path.join(PROC, "shape_constructs.tsv")) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                done[r["mirbase_id"]] = r
            log(f"resuming: {len(done)} previous records found")
    dump = []
    for mid, status in done.items():
        if status["status"] == "ok":
            stats["ok"] += 1
            offsets.append(int(status["offset"]))
        else:
            stats[status["status"]] = stats.get(status["status"], 0) + 1
        dump.append(status)

    for i, mid in enumerate(sorted(t for t in targets if t not in done)):
        loc = coords.get(mid)
        seq = hp_seq.get(mid)
        if loc is None or seq is None:
            stats["no_locus"] += 1
            dump.append({"mirbase_id": mid, "status": "no_locus", "offset": -1})
            continue
        chrom, start, end, strand = loc
        mid_pt = (start + end) // 2
        lo = max(1, mid_pt - CONSTRUCT_LEN // 2 - PAD)
        hi = min(mid_pt + CONSTRUCT_LEN // 2 + PAD, end + PAD)
        try:
            ctx = fetch(chrom, lo, hi)
        except RuntimeError:
            stats["fetch_fail"] += 1
            dump.append({"mirbase_id": mid, "status": "fetch_fail", "offset": -1})
            continue

        probe = seq if strand == "+" else revcomp_rna(seq)
        exp = (start) - lo                           # 0-based hairpin start in ctx
        pos = ctx.find(probe, max(0, exp - 20), exp + 20)
        if pos < 0:
            # tolerate a single database disagreement inside the window
            hits = [j for j in range(max(0, exp - 20), min(len(ctx), exp + 20) - len(probe) + 1)
                    if sum(a != b for a, b in zip(ctx[j:j + len(probe)], probe)) <= 1]
            pos = hits[0] if hits else -1
        if pos < 0:
            stats["no_match"] += 1
            dump.append({"mirbase_id": mid, "status": "no_match", "offset": -1})
            continue

        c0 = pos - ((CONSTRUCT_LEN - len(probe)) // 2)
        c0 = max(0, min(c0, len(ctx) - CONSTRUCT_LEN))
        if len(ctx) - c0 < CONSTRUCT_LEN:
            stats["too_long"] += 1
            dump.append({"mirbase_id": mid, "status": "too_long", "offset": -1})
            continue
        construct = ctx[c0:c0 + CONSTRUCT_LEN]
        offset = pos - c0
        if construct[offset:offset + len(probe)] != probe:
            stats["no_match"] += 1
            dump.append({"mirbase_id": mid, "status": "offset_mismatch", "offset": -1})
            continue

        stats["ok"] += 1
        offsets.append(offset)
        dump.append({"mirbase_id": mid, "status": "ok", "offset": offset,
                    "chrom": chrom, "locus_start": start, "locus_end": end,
                    "strand": strand, "construct": construct,
                    "hairpin_len": len(probe)})
        # incremental checkpoint so a killed run resumes cheaply
        if (i + 1) % 25 == 0:
            with open(os.path.join(PROC, "shape_constructs.tsv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t",
                                   extrasaction="ignore")
                w.writeheader()
                w.writerows(sorted(dump, key=lambda x: x["mirbase_id"]))
            log(f"  checkpoint {i + 1}/new  ok={stats['ok']}")

    p = os.path.join(PROC, "shape_constructs.tsv")
    fields = ["mirbase_id", "status", "offset", "chrom", "locus_start", "locus_end",
              "strand", "hairpin_len", "construct"]
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(dump, key=lambda x: x["mirbase_id"]))
    log(f"wrote {p}: {stats}")
    if offsets:
        import statistics
        log(f"hairpin offset in construct: median={statistics.median(offsets):.0f} "
            f"min={min(offsets)} max={max(offsets)}")

    # how many mapped SHAPE records survive end to end
    keep = {r["mirbase_id"] for r in out if r["status"] == "ok"}
    mapped = {r["mirbase_id"] for r in rows}
    log(f"SHAPE usable: {len(keep & mapped)}/{len(mapped)}")
    dropped = sorted(mapped - keep)
    if dropped:
        log(f"dropped: {dropped[:10]}{' ...' if len(dropped) > 10 else ''}")


if __name__ == "__main__":
    main()
