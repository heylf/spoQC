"""Real-data check: combine_masks vs the verbatim origin/dev function on the first ROWS image rows of
full-scale hqcr/hqpr/hqtr masks. Records every number handed to the figure code (as
test_combine_masks_equivalence does) and prints IDENTICAL or the differing calls.

usage: python tests/realdata_combine_masks.py <spoqc_tmp_dir> <scratch_dir> <rows> [dim_y]
If the tmp dir has no hqtr smoothed mask, the raw hqtr mask stands in under the smoothed names."""
import glob, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyarrow.parquet as pq
SRC, DST, ROWS = sys.argv[1], sys.argv[2], int(sys.argv[3])
DIM_Y = int(sys.argv[4]) if len(sys.argv) > 4 else 35416  # Xenium breast rep1 morphology_focus s0
N = ROWS * DIM_Y
os.makedirs(DST, exist_ok=True)
for suf in ["", "_smoothed"]:
    pq.write_table(pq.read_table(f"{SRC}/hqcr_output_mask{suf}_raw.parquet").slice(0, N), f"{DST}/hqcr_output_mask{suf}_raw.parquet")
    for prefix in ["hqpr_0", "hqtr"]:
        src, dst = f"{SRC}/{prefix}_output_mask{suf}_raw", f"{DST}/{prefix}_output_mask{suf}_raw"
        rename = None
        if not os.path.exists(src):  # hqtr refinement OOM'd: stand in the raw hqtr mask under the smoothed names
            src, rename = f"{SRC}/{prefix}_output_mask_raw", {f"{prefix}_beliefs": f"{prefix}_beliefs{suf}", f"{prefix}_mask": f"{prefix}_mask{suf}"}
            print("STAND-IN", dst, "<-", src)
        os.makedirs(dst, exist_ok=True)
        parts = sorted(glob.glob(f"{src}/part.*.parquet"), key=lambda p: int(re.search(r"part\.(\d+)", p).group(1)))
        left = N
        for p in parts:
            if left <= 0: break
            t = pq.read_table(p); t = t.slice(0, min(left, t.num_rows)); left -= t.num_rows
            if rename: t = t.rename_columns([rename.get(c, c) for c in t.schema.names])
            pq.write_table(t, f"{dst}/{os.path.basename(p)}")
        print(prefix, suf, t.schema.names, flush=True)
import pytest
import test_combine_masks_equivalence as T
T.DIM_X, T.DIM_Y = ROWS, DIM_Y
T.IMAGEDIM = T.helperfuncs.ImageDimStruct(0, 0, DIM_Y, ROWS)
mp = pytest.MonkeyPatch()
t0 = time.time(); old = T._record(T.legacy, mp, DST, DST + "/fig"); t1 = time.time()
new = T._record(T.combine_masks, mp, DST, DST + "/fig", threads=4); t2 = time.time()
mp.undo()
print(f"legacy {t1-t0:.1f}s new {t2-t1:.1f}s calls {len(old)} {len(new)}")
bad = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
print("IDENTICAL" if len(old) == len(new) and not bad else f"DIFFERENT at {bad}")
for c in old:
    if c[0] == "venn": print(c)
