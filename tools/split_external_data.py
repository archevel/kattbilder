"""Rewrite an ONNX model so its initializers live in several external files of
at most CHUNK bytes each (browsers cannot allocate one multi-GB ArrayBuffer).

usage: python3 split_external_data.py in.onnx out_dir prefix [chunk_mb]
Writes out_dir/<prefix>.onnx and out_dir/<prefix>_0.bin, _1.bin, ...
"""
import os, sys
import onnx
from onnx import external_data_helper as edh

src, out_dir, prefix = sys.argv[1:4]
chunk = int(sys.argv[4] if len(sys.argv) > 4 else 900) * 1024 * 1024
os.makedirs(out_dir, exist_ok=True)

model = onnx.load(src)  # loads external data into raw_data
files, cur, cur_name, off = [], None, None, 0

def open_chunk():
    global cur, cur_name, off
    if cur: cur.close()
    cur_name = f"{prefix}_{len(files)}.bin"
    files.append(cur_name)
    cur = open(os.path.join(out_dir, cur_name), "wb")
    off = 0

open_chunk()
n = 0
for t in model.graph.initializer:
    if not t.HasField("raw_data"):
        continue
    data = t.raw_data
    if len(data) < 1024:
        continue
    if off + len(data) > chunk and off > 0:
        open_chunk()
    cur.write(data)
    edh.set_external_data(t, location=cur_name, offset=off, length=len(data))
    t.data_location = onnx.TensorProto.EXTERNAL
    t.ClearField("raw_data")
    off += len(data)
    n += 1
cur.close()
onnx.save_model(model, os.path.join(out_dir, f"{prefix}.onnx"))
sizes = {f: os.path.getsize(os.path.join(out_dir, f)) for f in files}
print(f"{n} tensors ->", {f: round(s / 1e6) for f, s in sizes.items()})
