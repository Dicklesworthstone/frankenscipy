//! `frankenscipy-22van` gate (b): does hoisting the spline knobs let the compiler SPECIALISE
//! the per-pixel loop, or does it only remove some cheap atomic loads?
//!
//! WHY THIS IS A SEPARATE BINARY AND NOT A TOGGLE. The bead pre-registered that the
//! specialisation half must be an A/B of two SHIPPING binaries. That looked like caution; it
//! is a hard requirement, and the reason is concrete. `SplineFlags::resolve_or_reread` — the
//! same-binary A/B knob `NDIMAGE_SPLINE_FLAG_HOIST_DISABLE` — is consulted inside
//! `sample_interpolated`, i.e. ONCE PER PIXEL. So in the default build BOTH arms of that
//! toggle carry a per-pixel atomic load, neither can specialise, and a same-binary A/B is
//! structurally incapable of measuring the effect the bead is about. It would report a null
//! and the null would mean nothing.
//!
//! The `spline-flags-const` feature compiles both knobs to constants, so every atomic leaves
//! the per-pixel path. Timing THIS binary built with the feature against the same binary built
//! without it, alternated inside one window, is the comparison that can answer gate (b).
//!
//! It prints one line per replicate and nothing else; the pairing and the statistics are done
//! by the caller, which alternates the two binaries ABBAABBA rather than running one and then
//! the other (window drift on this host exceeds the effect being measured).

use fsci_ndimage::{BoundaryMode, NdArray, shift};
use std::hint::black_box;
use std::io::Read;
use std::time::Instant;

/// Which arm this binary IS, decided at compile time. Printed so a row can never attribute a
/// sample to the wrong arm — the two builds are otherwise byte-identical in their output.
const ARM: &str = if cfg!(feature = "spline-flags-const") {
    "const"
} else {
    "atomic"
};

/// SHA-256 of this running executable, so the two arms are distinguishable in the record.
fn elf_sha256() -> String {
    let Ok(mut f) = std::fs::File::open("/proc/self/exe") else {
        return "unavailable".into();
    };
    // Vendored FNV-style digest would not be a SHA; shell out to the coreutils hash instead is
    // not available here, so read-and-hash with a tiny SHA-256. Keep it dependency-free.
    let mut buf = Vec::new();
    if f.read_to_end(&mut buf).is_err() {
        return "unavailable".into();
    }
    sha256_hex(&buf)
}

fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(v);
        }
    }
    h.iter().map(|w| format!("{w:08x}")).collect()
}

fn main() {
    let env_usize = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let w = env_usize("VAN22_W", 256);
    let reps = env_usize("VAN22_REPS", 3);
    let min_of = env_usize("VAN22_MIN_OF", 7);
    let replicates = env_usize("VAN22_REPLICATES", 5);

    // Deterministic, non-separable fixture. Values are irrational-ish so no coincidence in the
    // data can make the spline weights degenerate.
    let data: Vec<f64> = (0..(w * w) as u64)
        .map(|i| ((i as f64) * 0.618_033_988_749_894_9).sin() * 3.0 + (i as f64) * 0.25)
        .collect();
    let input = NdArray::new(data, vec![w, w]).expect("fixture");

    // NON-INTEGER shifts on both axes: an integer shift takes the cardinal fast path and never
    // reaches the spline sampler, so the arms would run identical code and the row would be
    // vacuous. Order 3 gives a 4-tap support per axis, so the flags are consulted heavily.
    let shifts = [0.37f64, -0.61];

    println!(
        "arm={ARM} elf_sha256={} w={w} reps={reps} min_of={min_of}",
        elf_sha256()
    );

    // Warm-up outside every timer: first call pays allocation and page-in.
    let warm = shift(&input, &shifts, 3, BoundaryMode::Reflect, 0.0).expect("warmup");
    // Fold the output so nothing can be optimised away, and print it: the two arms MUST agree
    // bit-for-bit, and a caller comparing the two lines gets that check for free.
    let checksum: u64 = warm
        .data
        .iter()
        .fold(0u64, |acc, v| acc.rotate_left(1) ^ v.to_bits());
    println!(
        "arm={ARM} output_checksum={checksum:016x} pixels={}",
        warm.size()
    );

    for rep in 0..replicates {
        let mut best = f64::INFINITY;
        for _ in 0..min_of {
            let t0 = Instant::now();
            for _ in 0..reps {
                let out = shift(black_box(&input), &shifts, 3, BoundaryMode::Reflect, 0.0)
                    .expect("shift");
                black_box(out.data[0]);
            }
            let dt = t0.elapsed().as_secs_f64();
            if dt < best {
                best = dt;
            }
        }
        println!("arm={ARM} rep={rep} seconds={best:.9}");
    }
}
