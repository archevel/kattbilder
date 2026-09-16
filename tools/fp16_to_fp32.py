"""Convert an fp16 ONNX model to fp32 (weights, Cast targets, tensor types).

Needed because Chrome on Linux/NVIDIA does not expose WebGPU shader-f16, and
onnxruntime-web refuses to run fp16 graphs without it.

usage: python3 fp16_to_fp32.py in.onnx out.onnx [external_data_name]
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
ext = sys.argv[3] if len(sys.argv) > 3 else None

model = onnx.load(src)
n_init = n_cast = n_type = n_const = 0


def fix_type(tp):
    global n_type
    if tp.HasField("tensor_type") and tp.tensor_type.elem_type == TensorProto.FLOAT16:
        tp.tensor_type.elem_type = TensorProto.FLOAT
        n_type += 1
    elif tp.HasField("sequence_type"):
        fix_type(tp.sequence_type.elem_type)


def fix_tensor(t):
    if t.data_type == TensorProto.FLOAT16:
        arr = numpy_helper.to_array(t).astype(np.float32)
        t.CopyFrom(numpy_helper.from_array(arr, t.name))
        return True
    return False


def walk(graph):
    global n_init, n_cast, n_const
    for t in graph.initializer:
        n_init += fix_tensor(t)
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        fix_type(vi.type)
    for node in graph.node:
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i == TensorProto.FLOAT16:
                    a.i = TensorProto.FLOAT
                    n_cast += 1
        for a in node.attribute:
            if a.type == onnx.AttributeProto.TENSOR:
                n_const += fix_tensor(a.t)
            elif a.type == onnx.AttributeProto.TENSORS:
                for t in a.tensors:
                    n_const += fix_tensor(t)
            elif a.type == onnx.AttributeProto.GRAPH:
                walk(a.g)
            elif a.type == onnx.AttributeProto.GRAPHS:
                for g in a.graphs:
                    walk(g)


walk(model.graph)
print(f"initializers={n_init} casts={n_cast} types={n_type} const_tensors={n_const}")

# Sanity: no fp16 left anywhere
for t in model.graph.initializer:
    assert t.data_type != TensorProto.FLOAT16, t.name

if ext:
    onnx.save_model(model, dst, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=ext, size_threshold=1024, convert_attribute=False)
else:
    onnx.save_model(model, dst)
print("saved", dst)
