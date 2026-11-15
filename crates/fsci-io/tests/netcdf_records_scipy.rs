#![forbid(unsafe_code)]
//! Bidirectional live-SciPy checks for classic NetCDF unlimited record variables.
//! SciPy independently writes CDF-1/CDF-2 files for Rust to read and reads Rust's
//! CDF-1 output. Compare dimensions, shapes, primitive types, attributes, and
//! every value (including floating-point bits), allowing variable header order
//! to differ. Empty files and the lone Byte/Char/Short padding exception are
//! separate cases, not inferred from a Rust-to-Rust round trip.
//!
//! `FSCI_REQUIRE_SCIPY_ORACLE=1` requires SciPy 1.17.1 / NumPy 2.4.3;
//! `FSCI_PYTHON` selects the interpreter. A missing interpreter or SciPy is an
//! explicit skip only when the live oracle is not required. All other oracle
//! failures fail the test, and all three arms must compare all 21 cases.

use std::io::Write;
use std::process::{Command, Stdio};

use fsci_io::{
    NetcdfAttribute, NetcdfDimension, NetcdfFile, NetcdfValue, NetcdfVariable, read_netcdf_classic,
    write_netcdf_classic,
};
use serde_json::{Map, Value, json};

const ORACLE: &str = r#"
import io
import json
import os
import sys
try:
    import numpy as np
    import scipy
except ImportError as error:
    print(json.dumps({'missing': str(error)}))
    sys.exit(0)
from scipy.io import netcdf_file

if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') is not None:
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3'), \
        f'required oracle is SciPy 1.17.1 / NumPy 2.4.3, got {scipy.__version__} / {np.__version__}'

TYPE_CODES = {'Byte': 'b', 'Char': 'c', 'Short': 'h', 'Int': 'i',
              'Float': 'f', 'Double': 'd'}
DTYPES = {'Byte': 'i1', 'Short': 'i2', 'Int': 'i4', 'Float': 'f4', 'Double': 'f8'}

def array(value):
    kind = value['type']
    if kind == 'Char':
        return np.frombuffer(value['text'].encode('utf-8'), dtype='S1').copy()
    if kind in ('Float', 'Double'):
        width = 4 if kind == 'Float' else 8
        return np.array([int(x, 16) for x in value['bits']], dtype=f'u{width}').view(f'f{width}')
    return np.array(value['values'], dtype=DTYPES[kind])

def value_snapshot(value):
    if isinstance(value, bytes):
        return {'type': 'Char', 'text': value.decode('utf-8')}
    data = np.asarray(value)
    if data.dtype.kind == 'S':
        return {'type': 'Char', 'text': data.tobytes().decode('utf-8')}
    data = data.astype(data.dtype.newbyteorder('='), copy=False).ravel()
    kind = {('i', 1): 'Byte', ('i', 2): 'Short', ('i', 4): 'Int',
            ('f', 4): 'Float', ('f', 8): 'Double'}[(data.dtype.kind, data.dtype.itemsize)]
    if data.dtype.kind == 'f':
        width = data.dtype.itemsize
        return {'type': kind, 'bits': [format(int(x), f'0{2*width}x') for x in data.view(f'u{width}')]}
    return {'type': kind, 'values': [int(x) for x in data]}

def attrs_snapshot(attrs):
    return {name: value_snapshot(value) for name, value in attrs.items()}

def snapshot(raw):
    with netcdf_file(io.BytesIO(raw), 'r', mmap=False) as source:
        return {'dimensions': [[name, length] for name, length in source.dimensions.items()],
                'attributes': attrs_snapshot(source._attributes),
                'variables': {name: {'dimensions': list(var.dimensions),
                                     'shape': list(var.shape),
                                     'attributes': attrs_snapshot(var._attributes),
                                     'data': value_snapshot(var.data)}
                              for name, var in source.variables.items()}}

def set_attrs(target, attrs):
    for name, value in attrs.items():
        setattr(target, name, value['text'].encode('utf-8') if value['type'] == 'Char' else array(value))

def encode(spec, version):
    buffer = io.BytesIO()
    target = netcdf_file(buffer, 'w', version=version)
    for name, length in spec['dimensions']:
        target.createDimension(name, length)
    set_attrs(target, spec['attributes'])
    for name, var in spec['variables'].items():
        result = target.createVariable(name, TYPE_CODES[var['data']['type']], tuple(var['dimensions']))
        set_attrs(result, var['attributes'])
        data = array(var['data']).reshape(var['shape'])
        if data.size:
            if data.ndim:
                result[:] = data
            else:
                result.data[...] = data
    target.flush()
    raw = buffer.getvalue()
    target.close()
    return raw

rows = []
for case in json.load(sys.stdin):
    reads = []
    for version in (1, 2):
        raw = encode(case['spec'], version)
        reads.append({'version': version, 'bytes': raw.hex(), 'snapshot': snapshot(raw)})
    rows.append({'id': case['id'], 'read': reads,
                 'write': snapshot(bytes.fromhex(case['bytes']))})
print(json.dumps({'versions': [scipy.__version__, np.__version__], 'cases': rows}))
"#;

fn value_snapshot(value: &NetcdfValue) -> Value {
    match value {
        NetcdfValue::Byte(values) => json!({"type": "Byte", "values": values}),
        NetcdfValue::Char(text) => json!({"type": "Char", "text": text}),
        NetcdfValue::Short(values) => json!({"type": "Short", "values": values}),
        NetcdfValue::Int(values) => json!({"type": "Int", "values": values}),
        NetcdfValue::Float(values) => json!({"type": "Float", "bits": values.iter()
            .map(|x| format!("{:08x}", x.to_bits())).collect::<Vec<_>>()}),
        NetcdfValue::Double(values) => json!({"type": "Double", "bits": values.iter()
            .map(|x| format!("{:016x}", x.to_bits())).collect::<Vec<_>>()}),
    }
}

fn attributes_snapshot(attributes: &[NetcdfAttribute]) -> Value {
    Value::Object(
        attributes
            .iter()
            .map(|attr| (attr.name.clone(), value_snapshot(&attr.value)))
            .collect(),
    )
}

fn value_len(value: &NetcdfValue) -> usize {
    match value {
        NetcdfValue::Byte(v) => v.len(),
        NetcdfValue::Char(v) => v.len(),
        NetcdfValue::Short(v) => v.len(),
        NetcdfValue::Int(v) => v.len(),
        NetcdfValue::Float(v) => v.len(),
        NetcdfValue::Double(v) => v.len(),
    }
}

fn snapshot(file: &NetcdfFile) -> Value {
    let variables: Map<String, Value> = file
        .variables
        .iter()
        .map(|var| {
            let slab_len: usize = var
                .dim_ids
                .iter()
                .map(|&id| file.dimensions[id].len.unwrap_or(1))
                .product();
            let records = value_len(&var.data) / slab_len;
            let shape: Vec<_> = var
                .dim_ids
                .iter()
                .map(|&id| file.dimensions[id].len.unwrap_or(records))
                .collect();
            let dimensions: Vec<_> = var
                .dim_ids
                .iter()
                .map(|&id| file.dimensions[id].name.as_str())
                .collect();
            (
                var.name.clone(),
                json!({"dimensions": dimensions, "shape": shape,
            "attributes": attributes_snapshot(&var.attributes), "data": value_snapshot(&var.data)}),
            )
        })
        .collect();
    json!({"dimensions": file.dimensions.iter().map(|dim| json!([dim.name, dim.len])).collect::<Vec<_>>(),
           "attributes": attributes_snapshot(&file.attributes), "variables": variables})
}

fn variable(name: &str, dim_ids: &[usize], data: NetcdfValue) -> NetcdfVariable {
    NetcdfVariable {
        name: name.into(),
        dim_ids: dim_ids.into(),
        attributes: Vec::new(),
        data,
    }
}

fn cases() -> Vec<(String, NetcdfFile)> {
    let primitives = [
        ("byte", NetcdfValue::Byte(vec![i8::MIN, 0, i8::MAX])),
        ("char", NetcdfValue::Char("aZ!".into())),
        ("short", NetcdfValue::Short(vec![i16::MIN, 0, i16::MAX])),
        ("int", NetcdfValue::Int(vec![i32::MIN, 0, i32::MAX])),
        (
            "float",
            NetcdfValue::Float(vec![-0.0, f32::INFINITY, f32::from_bits(0x7fc0_1234)]),
        ),
        (
            "double",
            NetcdfValue::Double(vec![
                -0.0,
                f64::NEG_INFINITY,
                f64::from_bits(0x7ff8_0000_0000_1234),
            ]),
        ),
    ];
    let mut cases = Vec::new();
    for (name, data) in primitives {
        for records in [0, 1, 3] {
            let mut data = data.clone();
            match &mut data {
                NetcdfValue::Byte(v) => v.truncate(records),
                NetcdfValue::Char(v) => v.truncate(records),
                NetcdfValue::Short(v) => v.truncate(records),
                NetcdfValue::Int(v) => v.truncate(records),
                NetcdfValue::Float(v) => v.truncate(records),
                NetcdfValue::Double(v) => v.truncate(records),
            }
            cases.push((
                format!("single_{name}_{records}"),
                NetcdfFile {
                    dimensions: vec![NetcdfDimension {
                        name: "time".into(),
                        len: None,
                    }],
                    attributes: Vec::new(),
                    variables: vec![variable("sample", &[0], data)],
                },
            ));
        }
    }
    for records in [0, 1, 2] {
        let mut file = NetcdfFile {
            dimensions: vec![
                NetcdfDimension {
                    name: "time".into(),
                    len: None,
                },
                NetcdfDimension {
                    name: "station".into(),
                    len: Some(3),
                },
            ],
            attributes: vec![NetcdfAttribute {
                name: "title".into(),
                value: NetcdfValue::Char("record data".into()),
            }],
            variables: vec![
                variable(
                    "temperature",
                    &[0, 1],
                    NetcdfValue::Double(
                        [280.0, 281.5, 282.25, 283.0, 284.5, 285.25][..3 * records].into(),
                    ),
                ),
                variable("station_id", &[1], NetcdfValue::Int(vec![10, 20, 30])),
                variable("flags", &[0], NetcdfValue::Byte([-1, 7][..records].into())),
                variable("offset", &[], NetcdfValue::Float(vec![-0.0])),
                variable(
                    "counts",
                    &[0, 1],
                    NetcdfValue::Short([-2, 3, 4, 100, -200, 300][..3 * records].into()),
                ),
                variable(
                    "labels",
                    &[0, 1],
                    NetcdfValue::Char("abcXYZ"[..3 * records].into()),
                ),
                variable(
                    "quality",
                    &[0],
                    NetcdfValue::Int([1000, -2000][..records].into()),
                ),
                variable(
                    "wind",
                    &[0],
                    NetcdfValue::Float([1.25, -2.5][..records].into()),
                ),
            ],
        };
        file.variables[0].attributes.push(NetcdfAttribute {
            name: "units".into(),
            value: NetcdfValue::Char("K".into()),
        });
        file.variables[4].attributes.push(NetcdfAttribute {
            name: "valid_range".into(),
            value: NetcdfValue::Short(vec![-200, 300]),
        });
        cases.push((format!("mixed_{records}"), file));
    }
    cases
}

fn scipy_source_fixture(id: &str, file: &NetcdfFile) -> NetcdfFile {
    let mut source = file.clone();
    if id == "mixed_2" {
        // SciPy 1.17.1 sorts fixed scalar shape () after record variables,
        // then writes it after the FIRST record, overwriting record two.
        // Reproduced with Int [11, 22] plus scalar Float -0: SciPy itself
        // reads [11, -2147483648] back. Use a one-element fixed array only
        // for this SciPy-written source, so the source is not corrupt.
        // Rust-written mixed_2 still contains the scalar and must be read
        // correctly by SciPy. mixed_1 exercises its valid trailing scalar.
        let axis = source.dimensions.len();
        source.dimensions.push(NetcdfDimension {
            name: "scalar_axis".into(),
            len: Some(1),
        });
        source
            .variables
            .iter_mut()
            .find(|var| var.name == "offset")
            .expect("the mixed fixture includes its fixed offset")
            .dim_ids = vec![axis];
    }
    source
}

fn unhex(text: &str) -> Vec<u8> {
    assert!(text.len().is_multiple_of(2));
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("oracle emits hexadecimal bytes"))
        .collect()
}

#[test]
fn netcdf_unlimited_records_match_live_scipy_in_both_directions() {
    let cases = cases();
    let query: Vec<_> = cases
        .iter()
        .map(|(id, file)| {
            let bytes = write_netcdf_classic(file).expect("Rust writes a valid record case");
            let encoded: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
            json!({"id": id, "spec": snapshot(&scipy_source_fixture(id, file)), "bytes": encoded})
        })
        .collect();
    let required = std::env::var_os("FSCI_REQUIRE_SCIPY_ORACLE").is_some();
    let python = std::env::var_os("FSCI_PYTHON").unwrap_or_else(|| "python3".into());
    let mut child = match Command::new(&python)
        .arg("-c")
        .arg(ORACLE)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) if !required && error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("skipping live NetCDF oracle: {python:?} is unavailable");
            return;
        }
        Err(error) => panic!("start the live SciPy NetCDF oracle: {error}"),
    };
    child
        .stdin
        .take()
        .unwrap()
        .write_all(serde_json::to_string(&query).unwrap().as_bytes())
        .expect("send the NetCDF cases to SciPy");
    let output = child
        .wait_with_output()
        .expect("wait for the NetCDF oracle");
    assert!(
        output.status.success(),
        "NetCDF oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let response: Value = serde_json::from_slice(&output.stdout).expect("parse the NetCDF oracle");
    if let Some(error) = response.get("missing") {
        assert!(!required, "required live NetCDF oracle is missing: {error}");
        eprintln!("skipping live NetCDF oracle: {error}");
        return;
    }
    eprintln!(
        "NetCDF oracle {python:?}: SciPy {} / NumPy {}",
        response["versions"][0], response["versions"][1]
    );
    let rows = response["cases"]
        .as_array()
        .expect("oracle returns its case rows");
    assert_eq!(
        rows.len(),
        cases.len(),
        "the oracle must return every requested case"
    );
    let mut compared = [0usize; 3];
    assert_eq!(
        cases.len(),
        21,
        "the frozen record corpus must stay complete"
    );
    for ((id, file), row) in cases.iter().zip(rows) {
        assert_eq!(row["id"], *id);
        let expected = snapshot(file);
        let write_pass = row["write"] == expected;
        assert!(
            write_pass,
            "SciPy reading Rust output for {id}: expected {expected}, got {}",
            row["write"]
        );
        compared[2] += 1;
        let expected_read = snapshot(&scipy_source_fixture(id, file));
        let reads = row["read"].as_array().expect("both CDF versions");
        assert_eq!(reads.len(), 2);
        for (index, read) in reads.iter().enumerate() {
            assert_eq!(read["version"], index + 1);
            assert_eq!(
                read["snapshot"], expected_read,
                "SciPy independently writes the requested data for {id}"
            );
            let bytes = unhex(read["bytes"].as_str().unwrap());
            let rust = snapshot(
                &read_netcdf_classic(&bytes)
                    .unwrap_or_else(|error| panic!("{id} CDF-{}: {error}", index + 1)),
            );
            let pass = rust == read["snapshot"];
            assert!(
                pass,
                "Rust reading SciPy CDF-{} for {id}: expected {}, got {rust}",
                index + 1,
                read["snapshot"]
            );
            compared[index] += 1;
        }
    }
    assert_eq!(
        compared,
        [21, 21, 21],
        "every case must compare in all three arms"
    );
    eprintln!(
        "NetCDF compared: CDF-1 reads={}, CDF-2 reads={}, CDF-1 writes={}",
        compared[0], compared[1], compared[2]
    );
}
