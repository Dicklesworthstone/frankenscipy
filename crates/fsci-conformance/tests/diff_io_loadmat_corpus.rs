#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.io.loadmat`, `savemat` and `whosmat`
//! (frankenscipy-1ksfv.12).
//!
//! Three comparisons against the pinned SciPy 1.17.1:
//!
//! 1. **Corpus.** SciPy's own MAT test data (`scipy/io/matlab/tests/data`: Level 4 and 5 files,
//!    both byte orders, compressed and not, every class, and the corrupt files SciPy's tests
//!    read). For each file and each of seven option sets SciPy's `loadmat` result, or the class of
//!    its exception, is dumped to JSON, and fsci's `loadmat` of the same bytes must give the
//!    identical tree: variable names and order, shapes, dtypes (byte order aside: SciPy returns
//!    `>f8` for a big-endian file, fsci the same values natively), exact values (float bit
//!    patterns), field names, sparse structure, text, and `__header__` / `__version__` /
//!    `__globals__`. A file SciPy rejects must be rejected with the matching `IoError` class:
//!    `UnsupportedFeature` for `NotImplementedError` (v7.3), `InvalidFormat` for everything else.
//!    `whosmat` is compared the same way under two option sets, `matfile_version` for every file,
//!    and `varmats_from_mat` (byte for byte) for every Level 5 file.
//! 2. **Round trip A.** SciPy writes a compressed file (a struct holding a 2x2 double, 'hello',
//!    the nested cell {1, 'x'}, a complex 3x3, a complex 5x5 sparse matrix, an int8 2x3x4, a
//!    logical 2x2 and an empty double); fsci reads it under four option sets.
//! 3. **Round trip B.** fsci writes that content plus a 1-D vector, compressed and not, with
//!    `oned_as` row and column; SciPy reads each file to the tree fsci reads back, fsci reads
//!    back exactly what it wrote, and the uncompressed files equal SciPy's own `savemat` output
//!    for the same content byte for byte after the 116-byte description (which holds a
//!    timestamp).
//!
//! Representation notes, applied identically on both sides of the comparison:
//! - A 0-d value (a squeezed 1x1 array, which SciPy returns as a Python scalar) is compared as a
//!   scalar of Python type float / int / complex / bool / str.
//! - A struct with no fields, which SciPy returns under `struct_as_record` as an object array of
//!   `None`, is `MatStruct` with no field names in fsci; it is emitted as that array of `None`.
//! - Under `simplify_cells` SciPy's dicts and `mat_struct`s are 0-d `MatStruct`s, and its lists
//!   are 1-D cells.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_io::{
    IoError, LoadmatOptions, MatCell, MatChar, MatClass, MatData, MatDtype, MatFile, MatFormat,
    MatNumeric, MatObject, MatOpaque, MatSparse, MatStruct, MatValue, OnedAs, SavematOptions,
    loadmat, matfile_version, savemat, varmats_from_mat, whosmat,
};
use serde::Serialize;
use serde_json::{Map, Value, json};

const PACKET_ID: &str = "FSCI-P2C-017";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// The option sets every corpus file is read under (the `varnames` set's names are chosen by the
/// oracle from the file's own inventory).
const OPTION_SETS: [&str; 7] = [
    "default",
    "squeeze",
    "chars",
    "mat_dtype",
    "simplify",
    "noverify",
    "varnames",
];
const WHOS_SETS: [&str; 2] = ["whos_default", "whos_squeeze_chars"];
/// `matfile_version` for every file, and `varmats_from_mat` for every Level 5 file.
const FILE_ARMS: [&str; 2] = ["matfile_version", "varmats_from_mat"];
/// SciPy 1.17.1 ships 111 files in its MAT test-data directory (110 MAT files and
/// `japanese_utf8.txt`, which SciPy rejects as an unknown MAT type).
const CORPUS_FILES: usize = 111;

/// Shared by the three oracles: SciPy's result dict (or exception) as JSON, with floats as their
/// IEEE-754 bit patterns so the comparison is exact.
const DUMPER: &str = r#"
import io
import json
import os
import sys
import warnings

import numpy as np
import scipy
import scipy.io as sio
import scipy.sparse as sp
from scipy.io.matlab import MatlabFunction, MatlabObject, MatlabOpaque, mat_struct

warnings.simplefilter("ignore")


def fbits(x):
    x = float(x)
    if x != x:
        return "nan"
    return format(int(np.float64(x).view(np.uint64)), "016x")


def base_name(dt):
    return dt.newbyteorder("=").name


def values(a):
    a = np.asarray(a).ravel(order="F")
    if a.dtype.kind == "c":
        return {"dtype": base_name(a.dtype), "real": [fbits(x) for x in a.real],
                "imag": [fbits(x) for x in a.imag]}
    if a.dtype.kind == "f":
        return {"dtype": base_name(a.dtype), "real": [fbits(x) for x in a]}
    if a.dtype.kind == "b":
        return {"dtype": "bool", "real": [bool(x) for x in a]}
    return {"dtype": base_name(a.dtype), "real": [int(x) for x in a]}


def scalar(x):
    if isinstance(x, (bool, np.bool_)):
        return {"kind": "scalar", "type": "bool", "value": bool(x)}
    if isinstance(x, (int, np.integer)):
        return {"kind": "scalar", "type": "int", "value": int(x)}
    if isinstance(x, (float, np.floating)):
        return {"kind": "scalar", "type": "float", "value": fbits(x)}
    if isinstance(x, (complex, np.complexfloating)):
        return {"kind": "scalar", "type": "complex", "re": fbits(x.real), "im": fbits(x.imag)}
    if isinstance(x, str):
        return {"kind": "scalar", "type": "str", "value": x}
    raise TypeError(f"no scalar dump for {type(x).__name__}")


def struct_body(a):
    elems = list(a.ravel(order="F"))
    if a.dtype.names is not None:
        names = list(a.dtype.names)
        items = [[dump(e[n]) for n in names] for e in elems]
    else:
        names = list(elems[0]._fieldnames) if elems and isinstance(elems[0], mat_struct) else []
        items = [[dump(getattr(e, n)) for n in names] for e in elems]
    return {"shape": list(a.shape), "fields": names, "items": items}


def dump(v):
    if v is None:
        return {"kind": "none"}
    if sp.issparse(v):
        fmt = v.format
        c = v.tocsc() if fmt != "csc" else v
        out = {"kind": "sparse", "format": fmt, "shape": [int(n) for n in c.shape],
               "indptr": [int(i) for i in c.indptr], "indices": [int(i) for i in c.indices]}
        out.update(values(c.data))
        return out
    if isinstance(v, mat_struct):
        return {"kind": "dict", "fields": [[f, dump(getattr(v, f))] for f in v._fieldnames]}
    if isinstance(v, dict):
        return {"kind": "dict", "fields": [[k, dump(x)] for k, x in v.items()]}
    if isinstance(v, list):
        return {"kind": "cell", "shape": [len(v)], "items": [dump(x) for x in v]}
    if isinstance(v, MatlabFunction):
        inner = v.view(np.ndarray)
        if inner.ndim == 0 and inner.dtype == object:
            inner = inner[()]
        return {"kind": "function", "inner": dump(inner)}
    if isinstance(v, MatlabOpaque):
        r = v.view(np.ndarray)[0]
        return {"kind": "opaque", "s0": bytes(r["s0"]).hex(), "s1": bytes(r["s1"]).hex(),
                "s2": bytes(r["s2"]).hex(), "arr": dump(r["arr"])}
    if isinstance(v, MatlabObject):
        name = v.classname
        if isinstance(name, bytes):
            name = name.decode("latin1")
        out = {"kind": "object", "classname": name}
        out.update(struct_body(v.view(np.ndarray)))
        return out
    if isinstance(v, np.ndarray):
        if v.dtype.names is not None:
            out = {"kind": "struct"}
            out.update(struct_body(v))
            return out
        if v.dtype == object:
            return {"kind": "cell", "shape": list(v.shape),
                    "items": [dump(x) for x in v.ravel(order="F")]}
        if v.ndim == 0:
            return scalar(v.item())
        if v.dtype.kind == "U":
            return {"kind": "str", "width": v.dtype.itemsize // 4, "shape": list(v.shape),
                    "values": [str(x) for x in v.ravel(order="F")]}
        out = {"kind": "numeric", "shape": list(v.shape)}
        out.update(values(v))
        return out
    return scalar(v)


SYSTEM_KEYS = ("__header__", "__version__", "__globals__")


def load_dump(raw, kwargs):
    try:
        d = sio.loadmat(io.BytesIO(raw), **kwargs)
    except Exception as e:
        return {"error": {"type": type(e).__name__, "message": str(e)}}
    out = {"variables": [[k, dump(v)] for k, v in d.items() if k not in SYSTEM_KEYS]}
    if "__header__" in d:
        out["header"] = {"text": bytes(d["__header__"]).hex(), "version": d["__version__"],
                         "globals": list(d["__globals__"])}
    return out


def whos_dump(raw, kwargs):
    try:
        rows = sio.whosmat(io.BytesIO(raw), **kwargs)
    except Exception as e:
        return {"error": {"type": type(e).__name__, "message": str(e)}}
    return {"whos": [[name, [int(n) for n in shape], cls] for name, shape, cls in rows]}
"#;

const CORPUS_ORACLE: &str = r#"
data_dir = os.path.join(os.path.dirname(sio.matlab.__file__), "tests", "data")
option_sets = {
    "default": {},
    "squeeze": {"squeeze_me": True},
    "chars": {"chars_as_strings": False},
    "mat_dtype": {"mat_dtype": True},
    "simplify": {"simplify_cells": True},
    "noverify": {"verify_compressed_data_integrity": False},
}
whos_sets = {"whos_default": {}, "whos_squeeze_chars": {"squeeze_me": True, "chars_as_strings": False}}
files = []
for name in sorted(os.listdir(data_dir)):
    path = os.path.join(data_dir, name)
    raw = open(path, "rb").read()
    results = {}
    for set_id, kwargs in option_sets.items():
        results[set_id] = {"kwargs": kwargs, "result": load_dump(raw, kwargs)}
    whos = whos_dump(raw, {})
    names = [row[0] for row in whos.get("whos", [])]
    kwargs = {"variable_names": names[-1:] + names[:1] if names else ["absent"]}
    results["varnames"] = {"kwargs": kwargs, "result": load_dump(raw, kwargs)}
    for set_id, kwargs in whos_sets.items():
        results[set_id] = {"kwargs": kwargs, "result": whos_dump(raw, kwargs)}
    try:
        version = [int(n) for n in sio.matlab.matfile_version(io.BytesIO(raw))]
        results["matfile_version"] = {"kwargs": {}, "result": {"version": version}}
    except Exception as e:
        version = None
        results["matfile_version"] = {"kwargs": {}, "result": {"error": {"type": type(e).__name__, "message": str(e)}}}
    # varmats_from_mat reads Level 5 files only (SciPy opens anything with its Level 5 reader).
    if version is not None and version[0] == 1:
        try:
            parts = [[n, s.getvalue().hex()] for n, s in sio.matlab.varmats_from_mat(io.BytesIO(raw))]
            result = {"varmats": parts}
        except Exception as e:
            result = {"error": {"type": type(e).__name__, "message": str(e)}}
        results["varmats_from_mat"] = {"kwargs": {}, "result": result}
    files.append({"file": name, "path": path, "size": len(raw), "results": results})
print(json.dumps({"scipy": scipy.__version__, "numpy": np.__version__, "files": files}))
"#;

/// Round trip A's content, and round trip B's reference `savemat` output of the same content.
const CONTENT_ORACLE: &str = r#"
import scipy.io as sio


def content(with_vector):
    nested = np.empty((1, 2), dtype=object)
    nested[0, 0] = np.array([[1.0]])
    nested[0, 1] = "x"
    c = {
        "st": {"A": np.array([[1.0, 2.0], [3.0, 4.0]])},
        "hello": "hello",
        "nested": nested,
        "cplx": (np.arange(9.0) + 1j * np.arange(9.0, 18.0)).reshape(3, 3),
        "spc": sp.csc_array((np.array([1 + 1j, 2 - 1j, complex(0.0, -3.0), 4, 0.5 + 0.25j]),
                             np.array([0, 2, 4, 1, 3]), np.array([0, 1, 3, 3, 4, 5])),
                            shape=(5, 5)),
        "i8": (np.arange(24, dtype=np.int8) - 12).reshape((2, 3, 4), order="F"),
        "lg": np.array([[True, False], [False, True]]),
        "empty": np.zeros((0, 0)),
    }
    if with_vector:
        c["vec"] = np.arange(3.0)
    return c


q = json.load(sys.stdin)
out = {"scipy": scipy.__version__, "numpy": np.__version__}
if q["mode"] == "a":
    buf = io.BytesIO()
    sio.savemat(buf, content(False), do_compression=True)
    raw = buf.getvalue()
    out["hex"] = raw.hex()
    out["results"] = {set_id: load_dump(raw, kwargs) for set_id, kwargs in q["option_sets"].items()}
else:
    out["reads"] = {blob["id"]: load_dump(bytes.fromhex(blob["hex"]), {}) for blob in q["blobs"]}
    out["reference"] = {}
    for oned in ("row", "column"):
        buf = io.BytesIO()
        sio.savemat(buf, content(True), oned_as=oned)
        out["reference"][oned] = buf.getvalue().hex()
print(json.dumps(out))
"#;

// ── fsci's result as the oracle's JSON ────────────────────────────────────────────────────────

fn fbits(x: f64) -> Value {
    if x.is_nan() {
        json!("nan")
    } else {
        json!(format!("{:016x}", x.to_bits()))
    }
}

fn data_json(data: &MatData) -> Vec<Value> {
    match data {
        MatData::F64(v) => v.iter().map(|&x| fbits(x)).collect(),
        MatData::F32(v) => v.iter().map(|&x| fbits(f64::from(x))).collect(),
        MatData::I8(v) => v.iter().map(|&x| json!(x)).collect(),
        MatData::U8(v) => v.iter().map(|&x| json!(x)).collect(),
        MatData::I16(v) => v.iter().map(|&x| json!(x)).collect(),
        MatData::U16(v) => v.iter().map(|&x| json!(x)).collect(),
        MatData::I32(v) => v.iter().map(|&x| json!(x)).collect(),
        MatData::U32(v) => v.iter().map(|&x| json!(x)).collect(),
        MatData::I64(v) => v.iter().map(|&x| json!(x)).collect(),
        MatData::U64(v) => v.iter().map(|&x| json!(x)).collect(),
        MatData::Bool(v) => v.iter().map(|&x| json!(x)).collect(),
    }
}

fn dtype_json(real: &MatData, complex: bool) -> &'static str {
    match (complex, real.dtype()) {
        (true, MatDtype::F32) => "complex64",
        (true, _) => "complex128",
        (false, dtype) => dtype.name(),
    }
}

/// Values of a (numeric or sparse) array: `dtype`, `real` and, for a complex one, `imag`.
fn values_json(out: &mut Map<String, Value>, real: &MatData, imag: Option<&MatData>) {
    out.insert("dtype".into(), json!(dtype_json(real, imag.is_some())));
    out.insert("real".into(), Value::Array(data_json(real)));
    if let Some(imag) = imag {
        out.insert("imag".into(), Value::Array(data_json(imag)));
    }
}

fn text_json(text: &str) -> Value {
    json!(text.trim_end_matches('\0'))
}

fn scalar_numeric(n: &MatNumeric) -> Value {
    let first = |data: &MatData| data_json(data).into_iter().next().unwrap_or(Value::Null);
    match (&n.imag, n.real.dtype()) {
        (Some(imag), _) => {
            json!({"kind": "scalar", "type": "complex", "re": first(&n.real), "im": first(imag)})
        }
        (None, MatDtype::F64 | MatDtype::F32) => {
            json!({"kind": "scalar", "type": "float", "value": first(&n.real)})
        }
        (None, MatDtype::Bool) => {
            json!({"kind": "scalar", "type": "bool", "value": first(&n.real)})
        }
        (None, _) => json!({"kind": "scalar", "type": "int", "value": first(&n.real)}),
    }
}

/// How a struct is emitted: SciPy's record arrays (the default), or `mat_struct` objects and
/// dicts (`simplify_cells`).
#[derive(Clone, Copy)]
struct DumpMode {
    record: bool,
    level4: bool,
}

const RECORD_V5: DumpMode = DumpMode {
    record: true,
    level4: false,
};

fn struct_json(kind: &str, s: &MatStruct, mode: DumpMode) -> Map<String, Value> {
    let width = s.field_names.len();
    let count: usize = s.dims.iter().product();
    let items: Vec<Value> = (0..count)
        .map(|e| {
            Value::Array(
                (0..width)
                    .map(|f| {
                        s.values
                            .get(e * width + f)
                            .map_or(Value::Null, |v| dump(v, mode))
                    })
                    .collect(),
            )
        })
        .collect();
    let mut out = Map::new();
    out.insert("kind".into(), json!(kind));
    out.insert("shape".into(), json!(s.dims));
    out.insert("fields".into(), json!(s.field_names));
    out.insert("items".into(), Value::Array(items));
    out
}

fn dict_json(s: &MatStruct, element: usize, mode: DumpMode) -> Value {
    let width = s.field_names.len();
    let fields: Vec<Value> = s
        .field_names
        .iter()
        .enumerate()
        .map(|(f, name)| {
            json!([
                name,
                s.values
                    .get(element * width + f)
                    .map_or(Value::Null, |v| dump(v, mode))
            ])
        })
        .collect();
    json!({"kind": "dict", "fields": fields})
}

fn dump(value: &MatValue, mode: DumpMode) -> Value {
    match value {
        MatValue::Numeric(n) if n.dims.is_empty() => scalar_numeric(n),
        MatValue::Numeric(n) => {
            let mut out = Map::new();
            out.insert("kind".into(), json!("numeric"));
            out.insert("shape".into(), json!(n.dims));
            values_json(&mut out, &n.real, n.imag.as_ref());
            Value::Object(out)
        }
        MatValue::Char(c) if c.dims.is_empty() => json!({
            "kind": "scalar",
            "type": "str",
            "value": text_json(&c.chars.iter().collect::<String>()),
        }),
        MatValue::Char(c) => json!({
            "kind": "str",
            "width": 1,
            "shape": c.dims,
            "values": c.chars.iter().map(|ch| text_json(&ch.to_string())).collect::<Vec<_>>(),
        }),
        MatValue::Strings(s) if s.dims.is_empty() => json!({
            "kind": "scalar",
            "type": "str",
            "value": s.strings.first().map_or(json!(""), |t| text_json(t)),
        }),
        MatValue::Strings(s) => json!({
            "kind": "str",
            "width": s.width,
            "shape": s.dims,
            "values": s.strings.iter().map(|t| text_json(t)).collect::<Vec<_>>(),
        }),
        MatValue::Cell(c) => json!({
            "kind": "cell",
            "shape": c.dims,
            "items": c.items.iter().map(|v| dump(v, mode)).collect::<Vec<_>>(),
        }),
        MatValue::Struct(s) if mode.record && s.field_names.is_empty() => {
            // No record dtype has zero fields: SciPy returns an object array of None.
            if s.dims.is_empty() {
                json!({"kind": "none"})
            } else {
                let count: usize = s.dims.iter().product();
                json!({"kind": "cell", "shape": s.dims, "items": vec![json!({"kind": "none"}); count]})
            }
        }
        MatValue::Struct(s) if mode.record => Value::Object(struct_json("struct", s, mode)),
        MatValue::Struct(s) if s.dims.is_empty() => dict_json(s, 0, mode),
        MatValue::Struct(s) => {
            let count: usize = s.dims.iter().product();
            json!({
                "kind": "cell",
                "shape": s.dims,
                "items": (0..count).map(|e| dict_json(s, e, mode)).collect::<Vec<_>>(),
            })
        }
        MatValue::Object(MatObject { class_name, fields }) => {
            let mut out = struct_json("object", fields, mode);
            out.insert("classname".into(), json!(class_name));
            Value::Object(out)
        }
        MatValue::Sparse(s) => {
            let mut out = Map::new();
            out.insert("kind".into(), json!("sparse"));
            out.insert(
                "format".into(),
                json!(if mode.level4 { "coo" } else { "csc" }),
            );
            out.insert("shape".into(), json!([s.rows, s.cols]));
            out.insert("indptr".into(), json!(s.indptr));
            out.insert("indices".into(), json!(s.indices));
            values_json(&mut out, &s.data, s.imag.as_ref());
            Value::Object(out)
        }
        MatValue::Function(inner) => json!({"kind": "function", "inner": dump(inner, mode)}),
        MatValue::Opaque(MatOpaque { s0, s1, s2, arr }) => json!({
            "kind": "opaque",
            "s0": hex(s0),
            "s1": hex(s1),
            "s2": hex(s2),
            "arr": dump(arr, mode),
        }),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .filter_map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok())
        .collect()
}

fn file_json(file: &MatFile, mode: DumpMode) -> Value {
    let variables: Vec<Value> = file
        .variables
        .iter()
        .map(|(name, value)| json!([name, dump(value, mode)]))
        .collect();
    let mut out = Map::new();
    out.insert("variables".into(), Value::Array(variables));
    if let Some(header) = &file.header {
        out.insert(
            "header".into(),
            json!({"text": hex(&header.text), "version": header.version, "globals": header.globals}),
        );
    }
    Value::Object(out)
}

fn options_from(kwargs: &Value) -> LoadmatOptions {
    let flag =
        |key: &str, default: bool| kwargs.get(key).and_then(Value::as_bool).unwrap_or(default);
    LoadmatOptions {
        mat_dtype: flag("mat_dtype", false),
        squeeze_me: flag("squeeze_me", false),
        chars_as_strings: flag("chars_as_strings", true),
        simplify_cells: flag("simplify_cells", false),
        verify_compressed_data_integrity: flag("verify_compressed_data_integrity", true),
        variable_names: kwargs
            .get("variable_names")
            .and_then(Value::as_array)
            .map(|names| {
                names
                    .iter()
                    .filter_map(|n| n.as_str().map(str::to_owned))
                    .collect()
            }),
    }
}

/// The `IoError` class a SciPy exception corresponds to.
fn error_class_matches(scipy_type: &str, error: &IoError) -> bool {
    match error {
        IoError::UnsupportedFeature(_) => scipy_type == "NotImplementedError",
        IoError::InvalidFormat(_) => scipy_type != "NotImplementedError",
        IoError::IoFailed(_) => false,
    }
}

/// Where two JSON trees first differ, for the mismatch report.
fn first_difference(path: &str, scipy: &Value, fsci: &Value) -> Option<String> {
    match (scipy, fsci) {
        (Value::Object(a), Value::Object(b)) => {
            for (key, av) in a {
                match b.get(key) {
                    Some(bv) => {
                        if let Some(d) = first_difference(&format!("{path}.{key}"), av, bv) {
                            return Some(d);
                        }
                    }
                    None => return Some(format!("{path}.{key}: missing on the fsci side")),
                }
            }
            b.keys()
                .find(|k| !a.contains_key(*k))
                .map(|k| format!("{path}.{k}: only on the fsci side"))
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                return Some(format!(
                    "{path}: SciPy has {} entries, fsci {}",
                    a.len(),
                    b.len()
                ));
            }
            a.iter()
                .zip(b)
                .enumerate()
                .find_map(|(i, (av, bv))| first_difference(&format!("{path}[{i}]"), av, bv))
        }
        _ if scipy == fsci => None,
        _ => {
            let show = |v: &Value| {
                let text = v.to_string();
                text.chars().take(160).collect::<String>()
            };
            Some(format!(
                "{path}: SciPy {} vs fsci {}",
                show(scipy),
                show(fsci)
            ))
        }
    }
}

// ── oracle plumbing ───────────────────────────────────────────────────────────────────────────

fn run_oracle(script: &str, query: &Value) -> Option<Value> {
    let source = format!("{DUMPER}\n{script}");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(&source)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn the loadmat oracle: {e}"
            );
            eprintln!("skipping loadmat oracle: python3 not available ({e})");
            return None;
        }
    };
    if let Some(stdin) = child.stdin.as_mut()
        && let Err(e) = stdin.write_all(query.to_string().as_bytes())
    {
        let output = child
            .wait_with_output()
            .expect("wait for the failed loadmat oracle");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "loadmat oracle stdin write failed: {e}; stderr: {stderr}"
        );
        eprintln!("skipping loadmat oracle: stdin write failed ({e})\n{stderr}");
        return None;
    }
    let output = child
        .wait_with_output()
        .expect("wait for the loadmat oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "loadmat oracle failed: {stderr}"
        );
        eprintln!("skipping loadmat oracle: scipy not available\n{stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("parse the loadmat oracle's JSON"))
}

#[derive(Debug, Clone, Serialize)]
struct CaseRow {
    case_id: String,
    arm: String,
    pass: bool,
    detail: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    scipy: String,
    numpy: String,
    files: usize,
    files_matching_every_set: usize,
    compared: BTreeMap<String, ArmCounts>,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    /// Files SciPy rejects under the default options, with both sides' errors.
    default_refusals: Vec<CaseRow>,
    mismatches: Vec<CaseRow>,
}

/// The comparison must be able to see a difference before its silence means anything
/// (the two-arm rule): SciPy's dump of `testdouble_7.4_GLNX86.mat` must hold the nine doubles
/// 0, π/4, …, 2π, fsci's must equal it, and one flipped bit, a signed zero or a changed dtype in
/// fsci's tree must each be reported.
fn detector_controls(files: &[Value]) {
    let control = files
        .iter()
        .find(|f| f["file"] == "testdouble_7.4_GLNX86.mat")
        .expect("the control file is in the corpus");
    let scipy = &control["results"]["default"]["result"];
    let bits = |x: f64| json!(format!("{:016x}", x.to_bits()));
    assert_eq!(
        scipy
            .pointer("/variables/0/1/real")
            .and_then(Value::as_array)
            .map(Vec::len),
        Some(9)
    );
    assert_eq!(
        scipy.pointer("/variables/0/1/real/1"),
        Some(&bits(std::f64::consts::FRAC_PI_4))
    );
    let bytes = fs::read(control["path"].as_str().unwrap_or("")).expect("read the control file");
    let fsci = fsci_load_json(&bytes, &json!({})).expect("fsci reads the control file");
    assert_eq!(
        first_difference("", scipy, &fsci),
        None,
        "the control must match"
    );
    let perturbed = |pointer: &str, value: Value| {
        let mut tree = fsci.clone();
        let slot = tree.pointer_mut(pointer).expect("the control path exists");
        *slot = value;
        first_difference("", scipy, &tree)
    };
    let pi4 = std::f64::consts::FRAC_PI_4.to_bits();
    assert!(perturbed("/variables/0/1/real/1", json!(format!("{:016x}", pi4 ^ 1))).is_some());
    assert!(perturbed("/variables/0/1/real/0", bits(-0.0)).is_some());
    assert!(perturbed("/variables/0/1/dtype", json!("float32")).is_some());
    assert!(perturbed("/variables/0/0", json!("renamed")).is_some());
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    fs::create_dir_all(output_dir()).expect("create the loadmat diff output dir");
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize the loadmat diff log");
    fs::write(path, json).expect("write the loadmat diff log");
}

/// Compare one oracle result (a dump or an exception) with fsci's, recording the case.
fn compare_case(
    ledger: &mut CompareLedger,
    rows: &mut Vec<CaseRow>,
    arm: &str,
    case_id: &str,
    scipy: &Value,
    fsci: Result<Value, IoError>,
) -> bool {
    let (pass, detail) = match (scipy.get("error"), fsci) {
        (Some(error), fsci) => {
            let scipy_type = error.get("type").and_then(Value::as_str).unwrap_or("?");
            let refused = fsci
                .as_ref()
                .err()
                .is_some_and(|e| error_class_matches(scipy_type, e));
            ledger.expected_raise(arm, case_id, refused);
            let detail = match fsci {
                Ok(_) => format!("SciPy raised {scipy_type} but fsci read the file"),
                Err(e) if refused => format!("both refuse: SciPy {scipy_type}, fsci {e}"),
                Err(e) => format!("error class differs: SciPy {scipy_type}, fsci {e:?}"),
            };
            (refused, detail)
        }
        (None, Err(e)) => {
            ledger.rust_failed(arm, case_id, &e.to_string());
            (false, format!("SciPy read the file, fsci refused: {e}"))
        }
        (None, Ok(tree)) => match first_difference("", scipy, &tree) {
            None => {
                ledger.compared(arm, case_id, true);
                (true, "identical".to_owned())
            }
            Some(difference) => {
                ledger.compared(arm, case_id, false);
                (false, difference)
            }
        },
    };
    if !pass {
        rows.push(CaseRow {
            case_id: case_id.to_owned(),
            arm: arm.to_owned(),
            pass,
            detail,
        });
    }
    pass
}

fn fsci_load_json(bytes: &[u8], kwargs: &Value) -> Result<Value, IoError> {
    let options = options_from(kwargs);
    let file = loadmat(bytes, &options)?;
    let mode = DumpMode {
        record: !options.simplify_cells,
        level4: file.version.0 == 0,
    };
    Ok(file_json(&file, mode))
}

fn fsci_whos_json(bytes: &[u8], kwargs: &Value) -> Result<Value, IoError> {
    let rows: Vec<Value> = whosmat(bytes, &options_from(kwargs))?
        .into_iter()
        .map(|info| json!([info.name, info.shape, info.class_name]))
        .collect();
    Ok(json!({"whos": rows}))
}

#[test]
fn diff_io_loadmat_corpus() {
    let Some(oracle) = run_oracle(CORPUS_ORACLE, &json!({})) else {
        return;
    };
    let start = Instant::now();
    let files = oracle["files"].as_array().cloned().unwrap_or_default();
    assert_eq!(
        files.len(),
        CORPUS_FILES,
        "SciPy's MAT test-data directory should hold {CORPUS_FILES} files"
    );
    detector_controls(&files);
    let arms: Vec<&str> = OPTION_SETS
        .iter()
        .chain(WHOS_SETS.iter())
        .chain(FILE_ARMS.iter())
        .copied()
        .collect();
    let mut ledger = CompareLedger::new("diff_io_loadmat_corpus", &arms);
    let mut rows = Vec::new();
    let mut refusals = Vec::new();
    let mut matching = 0usize;
    let mut level5_files = 0usize;
    for entry in &files {
        let name = entry["file"].as_str().unwrap_or("?");
        let path = entry["path"].as_str().unwrap_or("");
        let bytes = fs::read(path).expect("read the corpus file SciPy read");
        assert_eq!(
            Some(bytes.len() as u64),
            entry["size"].as_u64(),
            "{name}: the file changed between the oracle's read and ours"
        );
        let mut all = true;
        for arm in &arms {
            let case = &entry["results"][*arm];
            if case.is_null() {
                // varmats_from_mat is only asked of Level 5 files.
                continue;
            }
            let kwargs = &case["kwargs"];
            let fsci = match *arm {
                "matfile_version" => {
                    matfile_version(&bytes).map(|(major, minor)| json!({"version": [major, minor]}))
                }
                "varmats_from_mat" => {
                    level5_files += 1;
                    varmats_from_mat(&bytes).map(|parts| {
                        json!({"varmats": parts.iter().map(|(n, b)| json!([n, hex(b)])).collect::<Vec<_>>()})
                    })
                }
                whos if whos.starts_with("whos") => fsci_whos_json(&bytes, kwargs),
                _ => fsci_load_json(&bytes, kwargs),
            };
            if *arm == "default"
                && let (Some(error), Err(e)) = (case["result"].get("error"), &fsci)
            {
                refusals.push(CaseRow {
                    case_id: name.to_owned(),
                    arm: (*arm).to_owned(),
                    pass: error_class_matches(error["type"].as_str().unwrap_or("?"), e),
                    detail: format!(
                        "SciPy {}: {} | fsci {e:?}",
                        error["type"].as_str().unwrap_or("?"),
                        error["message"].as_str().unwrap_or("")
                    ),
                });
            }
            all &= compare_case(
                &mut ledger,
                &mut rows,
                arm,
                &format!("{name}/{arm}"),
                &case["result"],
                fsci,
            );
        }
        if all {
            matching += 1;
        }
    }
    eprintln!(
        "diff_io_loadmat_corpus: {matching}/{} files match SciPy {} in every one of the {} arms",
        files.len(),
        oracle["scipy"].as_str().unwrap_or("?"),
        arms.len()
    );
    for row in &refusals {
        eprintln!("loadmat corpus refusal {}: {}", row.case_id, row.detail);
    }
    for row in &rows {
        eprintln!("loadmat corpus mismatch {}: {}", row.case_id, row.detail);
    }
    let log = DiffLog {
        test_id: "diff_io_loadmat_corpus".into(),
        category: "scipy.io.loadmat corpus".into(),
        scipy: oracle["scipy"].as_str().unwrap_or("?").into(),
        numpy: oracle["numpy"].as_str().unwrap_or("?").into(),
        files: files.len(),
        files_matching_every_set: matching,
        compared: ledger.counts().clone(),
        pass: rows.is_empty(),
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        default_refusals: refusals,
        mismatches: rows.clone(),
    };
    emit_log(&log);
    let counts = ledger.finish(level5_files.max(1));
    for arm in &arms {
        let expected = if *arm == "varmats_from_mat" {
            level5_files
        } else {
            CORPUS_FILES
        };
        assert_eq!(
            counts[*arm].compared_cases, expected,
            "arm `{arm}` must compare all {expected} of its files"
        );
    }
    assert!(
        level5_files >= 90,
        "only {level5_files} Level 5 files were split"
    );
    assert_eq!(
        matching,
        files.len(),
        "every corpus file must match in every option set"
    );
}

// ── round trips ───────────────────────────────────────────────────────────────────────────────

fn f64s(values: &[f64]) -> MatData {
    MatData::F64(values.to_vec())
}

/// Round trip A's content as fsci values, in SciPy's write order; `vec` (dims [3]) is added for
/// round trip B, where it is what `oned_as` changes.
fn fsci_content(with_vector: bool) -> Vec<(String, MatValue)> {
    // (np.arange(9.) + 1j * np.arange(9., 18.)).reshape(3, 3), column-major.
    let real: Vec<f64> = (0..9).map(|k| f64::from(3 * (k % 3) + k / 3)).collect();
    let imag: Vec<f64> = real.iter().map(|x| x + 9.0).collect();
    let mut content = vec![
        (
            "st".to_string(),
            MatValue::Struct(MatStruct {
                dims: vec![1, 1],
                field_names: vec!["A".to_string()],
                values: vec![MatValue::Numeric(MatNumeric::new(
                    vec![2, 2],
                    f64s(&[1.0, 3.0, 2.0, 4.0]),
                ))],
            }),
        ),
        ("hello".to_string(), MatValue::Char(MatChar::row("hello"))),
        (
            "nested".to_string(),
            MatValue::Cell(MatCell {
                dims: vec![1, 2],
                items: vec![
                    MatValue::Numeric(MatNumeric::new(vec![1, 1], f64s(&[1.0]))),
                    MatValue::Char(MatChar::row("x")),
                ],
            }),
        ),
        (
            "cplx".to_string(),
            MatValue::Numeric(MatNumeric::complex(vec![3, 3], f64s(&real), f64s(&imag))),
        ),
        (
            "spc".to_string(),
            MatValue::Sparse(MatSparse {
                rows: 5,
                cols: 5,
                logical: false,
                indptr: vec![0, 1, 3, 3, 4, 5],
                indices: vec![0, 2, 4, 1, 3],
                data: f64s(&[1.0, 2.0, 0.0, 4.0, 0.5]),
                imag: Some(f64s(&[1.0, -1.0, -3.0, 0.0, 0.25])),
            }),
        ),
        (
            "i8".to_string(),
            MatValue::Numeric(MatNumeric::new(
                vec![2, 3, 4],
                MatData::I8((-12..12).collect()),
            )),
        ),
        (
            "lg".to_string(),
            // A NumPy bool array is written as uint8 data with the logical flag.
            MatValue::Numeric(MatNumeric {
                dims: vec![2, 2],
                class: MatClass::Uint8,
                logical: true,
                real: MatData::U8(vec![1, 0, 0, 1]),
                imag: None,
            }),
        ),
        (
            "empty".to_string(),
            MatValue::Numeric(MatNumeric::new(vec![0, 0], f64s(&[]))),
        ),
    ];
    if with_vector {
        content.push((
            "vec".to_string(),
            MatValue::Numeric(MatNumeric::new(vec![3], f64s(&[0.0, 1.0, 2.0]))),
        ));
    }
    content
}

fn chars_as_chars() -> LoadmatOptions {
    LoadmatOptions {
        chars_as_strings: false,
        ..LoadmatOptions::default()
    }
}

fn read_options() -> Value {
    json!({
        "default": {},
        "squeeze": {"squeeze_me": true},
        "chars": {"chars_as_strings": false},
        "mat_dtype": {"mat_dtype": true},
    })
}

#[test]
fn diff_io_savemat_roundtrip_scipy_to_fsci() {
    let options = read_options();
    let Some(oracle) = run_oracle(
        CONTENT_ORACLE,
        &json!({"mode": "a", "option_sets": options}),
    ) else {
        return;
    };
    let bytes = unhex(oracle["hex"].as_str().unwrap_or(""));
    let sets: Vec<&str> = options
        .as_object()
        .map(|m| m.keys().map(String::as_str).collect())
        .unwrap_or_default();
    let mut ledger = CompareLedger::new("diff_io_savemat_roundtrip_scipy_to_fsci", &sets);
    let mut rows = Vec::new();
    for set in &sets {
        let fsci = fsci_load_json(&bytes, &options[*set]);
        compare_case(
            &mut ledger,
            &mut rows,
            set,
            &format!("scipy_compressed/{set}"),
            &oracle["results"][*set],
            fsci,
        );
    }
    // And the values themselves, independently of SciPy's reading: fsci reads exactly the content
    // (chars as chars, the form the content is written in), compared bit for bit through the
    // dump, since `==` on floats cannot see a signed zero.
    let file = loadmat(&bytes, &chars_as_chars()).expect("fsci reads SciPy's file");
    let expected = fsci_content(false);
    for (name, value) in &expected {
        assert_eq!(
            file.get(name).map(|v| dump(v, RECORD_V5)),
            Some(dump(value, RECORD_V5)),
            "{name} read from SciPy's compressed file"
        );
    }
    assert_eq!(file.variables.len(), expected.len());
    for row in &rows {
        eprintln!("round trip A mismatch {}: {}", row.case_id, row.detail);
    }
    ledger.finish(1);
}

#[test]
fn diff_io_savemat_roundtrip_fsci_to_scipy() {
    let content = fsci_content(true);
    let mut blobs = Vec::new();
    let mut written = BTreeMap::new();
    for do_compression in [false, true] {
        for (oned_as, oned) in [(OnedAs::Row, "row"), (OnedAs::Column, "column")] {
            let options = SavematOptions {
                do_compression,
                oned_as,
                ..SavematOptions::default()
            };
            let bytes = savemat(&content, &options).expect("fsci writes the content");
            let id = format!(
                "{}_{oned}",
                if do_compression {
                    "compressed"
                } else {
                    "plain"
                }
            );
            blobs.push(json!({"id": id, "hex": hex(&bytes)}));
            written.insert(id, (oned_as, bytes));
        }
    }
    let v4_content: Vec<(String, MatValue)> = content
        .iter()
        .filter(|(name, _)| matches!(name.as_str(), "hello" | "cplx" | "spc" | "empty" | "vec"))
        .cloned()
        .collect();
    let v4 = savemat(
        &v4_content,
        &SavematOptions {
            format: MatFormat::V4,
            ..SavematOptions::default()
        },
    )
    .expect("fsci writes Level 4");
    blobs.push(json!({"id": "level4_row", "hex": hex(&v4)}));
    let Some(oracle) = run_oracle(CONTENT_ORACLE, &json!({"mode": "b", "blobs": blobs})) else {
        return;
    };
    let arms = ["scipy_reads_fsci", "fsci_reads_back", "bytes_equal_scipy"];
    let mut ledger = CompareLedger::new("diff_io_savemat_roundtrip_fsci_to_scipy", &arms);
    let mut rows = Vec::new();
    for (id, (oned_as, bytes)) in &written {
        let fsci = fsci_load_json(bytes, &json!({}));
        compare_case(
            &mut ledger,
            &mut rows,
            "scipy_reads_fsci",
            id,
            &oracle["reads"][id.as_str()],
            fsci,
        );
        // fsci reads back what it wrote, with `vec` shaped by `oned_as`.
        let file = loadmat(bytes, &chars_as_chars()).expect("fsci reads its own file");
        let back = content.iter().all(|(name, value)| {
            let expected = match (name.as_str(), value) {
                ("vec", MatValue::Numeric(v)) => MatValue::Numeric(MatNumeric {
                    dims: if *oned_as == OnedAs::Row {
                        vec![1, 3]
                    } else {
                        vec![3, 1]
                    },
                    ..v.clone()
                }),
                _ => value.clone(),
            };
            file.get(name).map(|v| dump(v, RECORD_V5)) == Some(dump(&expected, RECORD_V5))
        });
        ledger.compared(
            "fsci_reads_back",
            id,
            back && file.variables.len() == content.len(),
        );
        if !back {
            rows.push(CaseRow {
                case_id: id.clone(),
                arm: "fsci_reads_back".into(),
                pass: false,
                detail: "fsci did not read back what it wrote".into(),
            });
        }
        if id.starts_with("plain") {
            let oned = if *oned_as == OnedAs::Row {
                "row"
            } else {
                "column"
            };
            let reference = unhex(oracle["reference"][oned].as_str().unwrap_or(""));
            let same = reference.len() == bytes.len() && reference.get(116..) == bytes.get(116..);
            ledger.compared("bytes_equal_scipy", id, same);
            if !same {
                let at = reference
                    .iter()
                    .zip(bytes.iter())
                    .skip(116)
                    .position(|(a, b)| a != b)
                    .map_or(reference.len().min(bytes.len()), |i| i + 116);
                rows.push(CaseRow {
                    case_id: id.clone(),
                    arm: "bytes_equal_scipy".into(),
                    pass: false,
                    detail: format!(
                        "differs from SciPy's savemat at byte {at} (lengths {} vs {})",
                        reference.len(),
                        bytes.len()
                    ),
                });
            }
        }
    }
    let fsci_v4 = fsci_load_json(&v4, &json!({}));
    compare_case(
        &mut ledger,
        &mut rows,
        "scipy_reads_fsci",
        "level4_row",
        &oracle["reads"]["level4_row"],
        fsci_v4,
    );
    for row in &rows {
        eprintln!(
            "round trip B mismatch {} ({}): {}",
            row.case_id, row.arm, row.detail
        );
    }
    ledger.finish(2);
}
