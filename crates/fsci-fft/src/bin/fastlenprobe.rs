//! scipy-parity probe for next_fast_len/prev_fast_len (bead ia47s).
//!
//! scipy 1.17.1 reference values (probed via scipy.fft.next_fast_len /
//! prev_fast_len on the pinned incumbent, real=False then real=True):
//! ```text
//! real=False: next 23->24  127->128  257->264  1023->1024
//!             prev 23->22   127->126  257->256  1023->1008
//! real=True : next 23->24  127->128  257->270  1023->1024
//!             prev 23->20   127->125  257->250  1023->1000
//! ```
#![forbid(unsafe_code)]

use fsci_fft::{next_fast_len, prev_fast_len};

fn main() {
    for t in [23usize, 127, 257, 1023] {
        println!(
            "PROBE next({t}) real=false = {} | scipy 24/128/264/1024",
            next_fast_len(t, false)
        );
        println!(
            "PROBE next({t}) real=true  = {} | scipy 24/128/270/1024",
            next_fast_len(t, true)
        );
        println!(
            "PROBE prev({t}) real=false = {} | scipy 22/126/256/1008",
            prev_fast_len(t, false)
        );
        println!(
            "PROBE prev({t}) real=true  = {} | scipy 20/125/250/1000",
            prev_fast_len(t, true)
        );
    }
}
