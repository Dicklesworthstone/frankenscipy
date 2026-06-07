//! Quasi-Monte Carlo low-discrepancy sequences.
//!
//! Compact `scipy.stats.qmc` surface for deterministic integration tests:
//! Halton, Sobol, Latin Hypercube, Poisson-disk sampling, scale, and the
//! centered/L2-star/mixture/wraparound discrepancy metrics.

use crate::StatsError;

/// First 32 primes, sufficient for the Halton sequence in dimensions ≤ 32.
/// scipy's qmc.Halton uses the same prime list.
const HALTON_PRIMES: &[u64] = &[
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131,
];

/// Common stateful engine surface for QMC samplers.
pub trait QmcEngine {
    /// Dimension of each emitted point.
    fn dimension(&self) -> usize;

    /// Rewind to the engine's initial state.
    fn reset(&mut self);

    /// Advance the engine by `n` points without returning them.
    fn fast_forward(&mut self, n: u64);

    /// Draw `n` points in row-major order.
    fn sample(&mut self, n: usize) -> Vec<f64>;
}

/// Halton low-discrepancy sequence sampler in `[0, 1)^d`.
///
/// For each dimension `i`, samples are the radical inverse of the index in
/// base `p_i`, where `p_i` is the i-th prime. Discrepancy of an `N`-point
/// Halton set scales as `O(log(N)^d / N)` — much better than i.i.d. uniform's
/// `O(1/sqrt(N))` — making Halton useful for low-dimensional Monte Carlo
/// integration.
///
/// The sequence is deterministic; `reset` rewinds to index 0. The first
/// emitted point is the origin, matching `scipy.stats.qmc.Halton(...,
/// scramble=False).random(...)`. `skip(k)` advances by `k` samples without
/// materializing them, matching scipy.stats.qmc.Halton().fast_forward(k).
#[derive(Debug, Clone)]
pub struct HaltonSampler {
    primes: Vec<u64>,
    next_index: u64,
}

impl HaltonSampler {
    /// Construct a Halton sampler for the given dimension.
    ///
    /// Returns `StatsError::InvalidArgument` if `dimension == 0` or exceeds
    /// the bundled prime table (32 dimensions). The 32-prime cutoff matches
    /// scipy's default — Halton's discrepancy degrades sharply past that.
    pub fn new(dimension: usize) -> Result<Self, StatsError> {
        if dimension == 0 {
            return Err(StatsError::InvalidArgument(
                "Halton dimension must be ≥ 1".to_string(),
            ));
        }
        if dimension > HALTON_PRIMES.len() {
            return Err(StatsError::InvalidArgument(format!(
                "Halton dimension {dimension} exceeds bundled prime table ({} primes)",
                HALTON_PRIMES.len()
            )));
        }
        Ok(Self {
            primes: HALTON_PRIMES[..dimension].to_vec(),
            next_index: 0,
        })
    }

    /// Dimension of the sequence.
    pub fn dimension(&self) -> usize {
        self.primes.len()
    }

    /// Index of the next sample to be emitted.
    pub fn next_index(&self) -> u64 {
        self.next_index
    }

    /// Rewind the sampler so the next call to `sample` returns sample index 0
    /// (the origin), matching SciPy's unscrambled Halton prefix.
    pub fn reset(&mut self) {
        self.next_index = 0;
    }

    /// Advance the sampler by `k` samples without materializing them. Useful
    /// when partitioning a long sequence across workers.
    pub fn skip(&mut self, k: u64) {
        self.next_index = self.next_index.saturating_add(k);
    }

    /// Draw the next `n` samples, returned as `n * dimension` values in
    /// row-major order: `out[i * d + j]` is the j-th coordinate of the i-th
    /// sample.
    pub fn sample(&mut self, n: usize) -> Vec<f64> {
        if self.primes.as_slice() == [2, 3, 5, 7] {
            return self.sample_4d(n);
        }

        let d = self.primes.len();
        let mut out = Vec::with_capacity(n.saturating_mul(d));
        for _ in 0..n {
            let idx = self.next_index;
            for &prime in &self.primes {
                out.push(radical_inverse_fast(idx, prime));
            }
            self.next_index = self.next_index.saturating_add(1);
        }
        out
    }

    fn sample_4d(&mut self, n: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(n.saturating_mul(4));
        for _ in 0..n {
            let idx = self.next_index;
            out.push(radical_inverse_const::<2>(idx));
            out.push(radical_inverse_const::<3>(idx));
            out.push(radical_inverse_const::<5>(idx));
            out.push(radical_inverse_const::<7>(idx));
            self.next_index = self.next_index.saturating_add(1);
        }
        out
    }
}

impl QmcEngine for HaltonSampler {
    fn dimension(&self) -> usize {
        HaltonSampler::dimension(self)
    }

    fn reset(&mut self) {
        HaltonSampler::reset(self);
    }

    fn fast_forward(&mut self, n: u64) {
        self.skip(n);
    }

    fn sample(&mut self, n: usize) -> Vec<f64> {
        HaltonSampler::sample(self, n)
    }
}

/// Radical inverse of `index` in base `prime`.
///
/// φ_b(n) = sum over i of d_i / b^(i+1) where n = sum d_i * b^i.
///
/// Numerically: starts with `f = 1/prime`, accumulates `result += f * d`
/// where `d` is the next digit, then `f /= prime`. Loop exits when `n` is
/// zero. The result lies strictly inside `(0, 1)` for any positive `index`.
fn radical_inverse(mut index: u64, prime: u64) -> f64 {
    let inv_prime = 1.0_f64 / prime as f64;
    let mut f = inv_prime;
    let mut result = 0.0_f64;
    while index > 0 {
        let digit = index % prime;
        result += digit as f64 * f;
        index /= prime;
        f *= inv_prime;
    }
    result
}

#[inline]
fn radical_inverse_const<const PRIME: u64>(mut index: u64) -> f64 {
    let inv_prime = 1.0_f64 / PRIME as f64;
    let mut f = inv_prime;
    let mut result = 0.0_f64;
    while index > 0 {
        let digit = index % PRIME;
        result += digit as f64 * f;
        index /= PRIME;
        f *= inv_prime;
    }
    result
}

/// Dispatch a bundled Halton prime to its `radical_inverse_const` specialisation
/// so the per-digit `% prime` / `/ prime` become compile-time division
/// strength-reduction (multiply-by-magic) instead of full runtime `div`
/// instructions. Byte-identical: `radical_inverse_const::<P>` is the exact same
/// float computation as `radical_inverse(index, P)`. Primes are the 32 entries of
/// HALTON_PRIMES; the wildcard keeps the runtime path as a safety net.
#[inline]
fn radical_inverse_fast(index: u64, prime: u64) -> f64 {
    match prime {
        2 => radical_inverse_const::<2>(index),
        3 => radical_inverse_const::<3>(index),
        5 => radical_inverse_const::<5>(index),
        7 => radical_inverse_const::<7>(index),
        11 => radical_inverse_const::<11>(index),
        13 => radical_inverse_const::<13>(index),
        17 => radical_inverse_const::<17>(index),
        19 => radical_inverse_const::<19>(index),
        23 => radical_inverse_const::<23>(index),
        29 => radical_inverse_const::<29>(index),
        31 => radical_inverse_const::<31>(index),
        37 => radical_inverse_const::<37>(index),
        41 => radical_inverse_const::<41>(index),
        43 => radical_inverse_const::<43>(index),
        47 => radical_inverse_const::<47>(index),
        53 => radical_inverse_const::<53>(index),
        59 => radical_inverse_const::<59>(index),
        61 => radical_inverse_const::<61>(index),
        67 => radical_inverse_const::<67>(index),
        71 => radical_inverse_const::<71>(index),
        73 => radical_inverse_const::<73>(index),
        79 => radical_inverse_const::<79>(index),
        83 => radical_inverse_const::<83>(index),
        89 => radical_inverse_const::<89>(index),
        97 => radical_inverse_const::<97>(index),
        101 => radical_inverse_const::<101>(index),
        103 => radical_inverse_const::<103>(index),
        107 => radical_inverse_const::<107>(index),
        109 => radical_inverse_const::<109>(index),
        113 => radical_inverse_const::<113>(index),
        127 => radical_inverse_const::<127>(index),
        131 => radical_inverse_const::<131>(index),
        other => radical_inverse(index, other),
    }
}

// ══════════════════════════════════════════════════════════════════════
// Sobol sampling
// ══════════════════════════════════════════════════════════════════════

/// Sobol low-discrepancy sequence sampler for dimensions `1..=32`.
///
/// Dimensions 0 and 1 use locally-generated direction tables; dimensions 2..=31
/// use SciPy's Joe–Kuo `_sv` direction numbers, so the unscrambled sequence is
/// bit-for-bit identical to `scipy.stats.qmc.Sobol(d, scramble=False)` across all
/// supported dimensions (e.g. the d=2 prefix `(0,0), (1/2,1/2), (3/4,1/4), ...`).
///
/// `with_digital_shift` applies a deterministic xor shift to each coordinate.
/// This is not a full Owen tree permutation, but it is the standard
/// random-shifted digital net primitive and preserves the low-discrepancy
/// structure while moving the origin off the boundary.
#[derive(Debug, Clone)]
pub struct SobolSampler {
    dimension: usize,
    next_index: u64,
    digital_shift: Vec<u64>,
}

impl SobolSampler {
    /// Construct an unscrambled Sobol sampler.
    pub fn new(dimension: usize) -> Result<Self, StatsError> {
        Self::with_shift_words(dimension, vec![0; dimension])
    }

    /// Construct a deterministic digital-shifted Sobol sampler.
    pub fn with_digital_shift(dimension: usize, seed: u64) -> Result<Self, StatsError> {
        let shifts = (0..dimension)
            .map(|idx| splitmix64(seed.wrapping_add(idx as u64)))
            .collect();
        Self::with_shift_words(dimension, shifts)
    }

    fn with_shift_words(dimension: usize, digital_shift: Vec<u64>) -> Result<Self, StatsError> {
        if dimension == 0 {
            return Err(StatsError::InvalidArgument(
                "Sobol dimension must be ≥ 1".to_string(),
            ));
        }
        if dimension > SOBOL_MAX_DIM {
            return Err(StatsError::InvalidArgument(format!(
                "Sobol supports dimensions 1..={SOBOL_MAX_DIM} in this QMC surface"
            )));
        }
        Ok(Self {
            dimension,
            next_index: 0,
            digital_shift,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn next_index(&self) -> u64 {
        self.next_index
    }

    pub fn reset(&mut self) {
        self.next_index = 0;
    }

    pub fn skip(&mut self, k: u64) {
        self.next_index = self.next_index.saturating_add(k);
    }

    pub fn sample(&mut self, n: usize) -> Vec<f64> {
        if self.dimension == 2 {
            return self.sample_2d(n);
        }

        // Incremental Gray-code recurrence (as in sample_2d), generalised to any
        // dimension. Instead of recomputing sobol_bits(idx, dim) from scratch for
        // every (sample, dim) — an O(64²) Gray-loop-times-sobol_direction cost
        // since sobol_direction itself recomputes the direction number per call —
        // carry a running `bits[dim]` and flip a single direction word per step.
        // Byte-identical: the direction words sobol_bits uses for dimension `dim`
        // are exactly `direction_table(dim)` (dims 0,1 have their own tables; dims
        // 2.. use the scipy `_sv` direction numbers), and gray(idx+1) differs from
        // gray(idx) in exactly the `trailing_zeros(idx+1)`-th bit, so the carried
        // value equals sobol_bits(idx, dim) bit-for-bit at every step.
        let d = self.dimension;
        let dir = |dim: usize| -> &'static [u64; 64] { direction_table(dim) };
        let mut out = Vec::with_capacity(n.saturating_mul(d));
        let mut idx = self.next_index;
        let mut bits: Vec<u64> = (0..d).map(|dim| sobol_bits(idx, dim)).collect();
        // `dim` indexes bits / digital_shift / the per-dim direction table in
        // lockstep, so a range loop reads clearest.
        #[allow(clippy::needless_range_loop)]
        for _ in 0..n {
            for dim in 0..d {
                out.push(bits_to_unit(bits[dim] ^ self.digital_shift[dim]));
            }
            let next_idx = idx.saturating_add(1);
            if next_idx != idx {
                let bit = next_idx.trailing_zeros() as usize;
                for dim in 0..d {
                    bits[dim] ^= dir(dim)[bit];
                }
            }
            idx = next_idx;
        }
        self.next_index = idx;
        out
    }

    fn sample_2d(&mut self, n: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(n.saturating_mul(2));
        let mut idx = self.next_index;
        let mut bits0 = sobol_bits(idx, 0);
        let mut bits1 = sobol_bits(idx, 1);
        let shift0 = self.digital_shift[0];
        let shift1 = self.digital_shift[1];

        for _ in 0..n {
            out.push(bits_to_unit(bits0 ^ shift0));
            out.push(bits_to_unit(bits1 ^ shift1));

            let next_idx = idx.saturating_add(1);
            if next_idx != idx {
                let bit = next_idx.trailing_zeros() as usize;
                bits0 ^= SOBOL_DIRECTION_TABLES[0][bit];
                bits1 ^= SOBOL_DIRECTION_TABLES[1][bit];
            }
            idx = next_idx;
        }

        self.next_index = idx;
        out
    }
}

impl QmcEngine for SobolSampler {
    fn dimension(&self) -> usize {
        SobolSampler::dimension(self)
    }

    fn reset(&mut self) {
        SobolSampler::reset(self);
    }

    fn fast_forward(&mut self, n: u64) {
        self.skip(n);
    }

    fn sample(&mut self, n: usize) -> Vec<f64> {
        SobolSampler::sample(self, n)
    }
}

const SOBOL_DIRECTION_TABLES: [[u64; 64]; 2] = [sobol_direction_table(0), sobol_direction_table(1)];

const fn sobol_direction_table(dimension: usize) -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut bit = 0usize;
    while bit < 64 {
        table[bit] = sobol_direction_const(dimension, bit);
        bit += 1;
    }
    table
}

const fn sobol_direction_const(dimension: usize, bit: usize) -> u64 {
    let mut direction = 1u64 << 63;
    if dimension == 0 {
        return direction >> bit;
    }

    let mut n = 0usize;
    while n < bit {
        direction ^= direction >> 1;
        n += 1;
    }
    direction
}

/// Highest Sobol dimension count supported by this surface.
///
/// Dimensions 0,1 use the locally-generated direction tables; dimensions 2..31
/// use SciPy's Joe–Kuo `_sv` direction numbers (see [`SOBOL_SV_EXT`]).
pub(crate) const SOBOL_MAX_DIM: usize = 32;

/// SciPy `scipy.stats.qmc.Sobol._sv` direction numbers for dimensions 2..=31,
/// as 30-bit integers (MSB at 2^29). Dimensions 0 and 1 are omitted because the
/// local `sobol_direction_const` tables already reproduce them exactly (verified
/// `sobol_direction_const(dim, bit) == _sv[dim][bit] << 34` for bits 0..29).
#[rustfmt::skip]
const SOBOL_SV30_DIMS_2_31: [[u32; 30]; 30] = [
    [536870912, 805306368, 402653184, 603979776, 973078528, 385875968, 595591168, 826277888, 438304768, 657457152, 999817216, 358875136, 538574848, 807862272, 406552576, 605372416, 975183872, 389033984, 597170176, 828646400, 437926400, 656873216, 1002152832, 357921088, 536885792, 805312304, 402662296, 603992420, 973085210, 385885991],
    [536870912, 805306368, 134217728, 335544320, 1040187392, 486539264, 679477248, 616562688, 908066816, 156237824, 376963072, 968097792, 503447552, 755171328, 545292288, 817971200, 136568832, 340905984, 1056606208, 494291968, 673276416, 609457408, 922347392, 158784320, 371195936, 961544240, 511180808, 766771220, 537002046, 805503005],
    [536870912, 268435456, 134217728, 738197504, 1040187392, 922746880, 511705088, 658505728, 379584512, 200278016, 676855808, 1009516544, 916586496, 468779008, 542670848, 271499264, 144826368, 754085888, 1054435328, 929870848, 503351808, 654495488, 377744768, 188970688, 681697312, 1022521360, 920217608, 460108844, 536906302, 268619575],
    [536870912, 268435456, 402653184, 201326592, 838860800, 150994944, 360710144, 1052770304, 941621248, 470810624, 706215936, 84672512, 665976832, 935919616, 766869504, 586072064, 301998080, 419434496, 226498560, 851446784, 169882112, 353372416, 1066931584, 1003241152, 529676320, 735648784, 128821784, 669173004, 900859826, 784934857],
    [536870912, 805306368, 671088640, 872415232, 369098752, 620756992, 260046848, 952107008, 799014912, 149946368, 126353408, 1019478016, 295567360, 434176000, 504463360, 555335680, 832446464, 702623744, 907126784, 354022400, 664679936, 216077568, 965846912, 769248448, 138287520, 68230640, 1041866760, 287174660, 429918270, 502268945],
    [536870912, 268435456, 671088640, 335544320, 570425344, 150994944, 75497472, 188743680, 497025024, 663748608, 34078720, 419692544, 747241472, 524615680, 1068007424, 781336576, 109649920, 306630656, 825059328, 954000384, 1033896448, 932184320, 705168000, 218366272, 243925536, 373620880, 992510024, 634536116, 455680474, 903271033],
    [536870912, 268435456, 671088640, 335544320, 167772160, 889192448, 444596224, 473956352, 236978176, 370147328, 981991424, 205783040, 640286720, 34930688, 814383104, 961101824, 1017946112, 508694528, 1051265024, 794608640, 103416320, 303366400, 411730560, 759775552, 917282976, 726799184, 606669224, 857523652, 134873826, 67436641],
    [536870912, 268435456, 939524096, 738197504, 637534208, 620756992, 578813952, 381681664, 216006656, 913309696, 478674944, 264503296, 812515328, 700121088, 350322688, 175980544, 42901504, 113946624, 887404544, 444587008, 982385152, 717947136, 317293440, 1064911552, 402694752, 1006744144, 504183336, 151428460, 76456654, 868921959],
    [536870912, 268435456, 671088640, 67108864, 33554432, 452984832, 662700032, 146800640, 367001600, 728760320, 535298048, 611581952, 308412416, 867500032, 570589184, 184795136, 260268032, 214298624, 401348608, 813575168, 946037248, 750121216, 126098048, 415902784, 1038180896, 795930800, 502175992, 1065217228, 903873406, 997196295],
    [536870912, 268435456, 134217728, 201326592, 369098752, 721420288, 629145600, 180355072, 891289600, 38797312, 950534144, 345243648, 327811072, 835125248, 413630464, 77185024, 461692928, 1036808192, 245770240, 798041088, 1041162752, 923496704, 998353024, 768140480, 111805280, 597099440, 672105176, 470663276, 504291890, 655061653],
    [536870912, 805306368, 671088640, 335544320, 1040187392, 587202560, 947912704, 213909504, 65011712, 139460608, 627572736, 396099584, 906100736, 118030336, 780500992, 1003339776, 90284032, 664530944, 503764992, 319628288, 277062144, 415429376, 1038834304, 727508288, 501219808, 458228016, 904397064, 257687948, 199361502, 744030901],
    [536870912, 805306368, 402653184, 603979776, 234881024, 822083584, 276824064, 683671552, 1012924416, 714080256, 1060634624, 558104576, 939655168, 335740928, 369197056, 352468992, 511762432, 432214016, 752945152, 38964224, 56346112, 198617344, 121238400, 893719616, 771751968, 16777264, 142606360, 213909540, 845152270, 462422065],
    [536870912, 268435456, 134217728, 1006632960, 704643072, 352321536, 645922816, 658505728, 127926272, 389021696, 524812288, 591659008, 153223168, 477167616, 988315648, 494387200, 451452928, 225726464, 317253632, 829731840, 954751488, 611771136, 510377088, 53989440, 432709664, 335892496, 109078536, 927167548, 262208042, 724742933],
    [536870912, 805306368, 134217728, 872415232, 905969664, 822083584, 293601280, 557842432, 694157312, 498073600, 728236032, 447479808, 191496192, 716111872, 57311232, 514605056, 896131072, 800018432, 619247616, 250430464, 227175936, 324837120, 652269696, 166802880, 764046880, 593272624, 786487432, 1039218164, 462056982, 308059905],
    [536870912, 268435456, 134217728, 1006632960, 234881024, 83886080, 1031798784, 432013312, 601882624, 336592896, 581435392, 66846720, 78249984, 721223680, 1059749888, 171884544, 793403392, 49188864, 327268352, 214109184, 375490048, 835931392, 256584832, 317502144, 461473312, 818105616, 409152648, 886092540, 852198958, 187583765],
    [536870912, 805306368, 134217728, 1006632960, 436207616, 419430400, 226492416, 457179136, 274726912, 940572672, 884473856, 654049280, 54132736, 348061696, 383418368, 200065024, 671293440, 201412608, 302118912, 621011968, 394296832, 37987584, 501388672, 592468672, 618298912, 518825264, 931141000, 843094780, 367222330, 525530409],
    [536870912, 268435456, 671088640, 335544320, 637534208, 1023410176, 729808896, 784334848, 970981376, 628097024, 117964800, 873201664, 921305088, 360513536, 1071677440, 141574144, 76079104, 248418304, 688576512, 218905600, 325690880, 314374912, 193007232, 1052936128, 299976224, 829582096, 554423976, 151960532, 493412358, 1002344493],
    [536870912, 805306368, 939524096, 738197504, 771751936, 251658240, 864026624, 272629760, 140509184, 342884352, 39321600, 559677440, 1016987648, 598147072, 402685952, 469811200, 369156096, 587247616, 494974976, 524303360, 1004588544, 70271232, 171450752, 892096960, 1053165920, 50038128, 614363768, 1067929244, 234881026, 1056964611],
    [536870912, 805306368, 939524096, 872415232, 436207616, 251658240, 578813952, 339738624, 710934528, 930086912, 384303104, 242483200, 629014528, 364183552, 41975808, 608223232, 308338688, 57724928, 216557568, 24394752, 134121984, 32854272, 685779328, 321920448, 76069792, 218425808, 697761272, 348220116, 92143618, 632619011],
    [536870912, 268435456, 402653184, 872415232, 234881024, 587202560, 528482304, 473956352, 840957952, 19922944, 115867648, 786169856, 312868864, 820969472, 696287232, 457195520, 1000366080, 175165440, 625489920, 279743488, 971931136, 54751488, 267274368, 71517376, 107936672, 251865968, 634294936, 829677500, 243662850, 48168961],
    [536870912, 805306368, 671088640, 603979776, 33554432, 419430400, 444596224, 574619648, 694157312, 852492288, 101187584, 727973888, 736231424, 482148352, 157319168, 47235072, 772317184, 258248704, 702679040, 96297984, 333507072, 546326784, 124257664, 1035230016, 803868704, 786024848, 939672968, 1009597428, 235446274, 1063555075],
    [536870912, 805306368, 134217728, 872415232, 301989888, 587202560, 897581056, 239075328, 895483904, 210763776, 749207552, 478412800, 351666176, 548601856, 853573632, 298893312, 608706560, 710201344, 524175360, 329747456, 1061049856, 598210816, 926840192, 396829248, 624843424, 883420688, 280915416, 988227532, 635314178, 913821699],
    [536870912, 805306368, 134217728, 335544320, 905969664, 1023410176, 260046848, 624951296, 601882624, 256901120, 1019740160, 196870144, 728891392, 42139648, 583565312, 314359808, 452075520, 247681024, 950556672, 95171584, 173678080, 795854080, 217241472, 63113536, 1059613984, 887892976, 532339272, 490705740, 1069161986, 759153923],
    [536870912, 268435456, 671088640, 738197504, 637534208, 687865856, 511705088, 893386752, 10485760, 403701760, 343408640, 572260352, 587857920, 301006848, 49840128, 727465984, 766156800, 216117248, 771758080, 352379904, 276879872, 809526528, 941660800, 72401984, 173550560, 256662896, 936545960, 738011012, 898292226, 414278913],
    [536870912, 805306368, 671088640, 201326592, 100663296, 218103808, 578813952, 658505728, 434110464, 546308096, 272105472, 406585344, 609353728, 173080576, 190349312, 804044800, 92971008, 1056780288, 959617024, 813822976, 134261248, 1006660864, 771809152, 16811584, 612422368, 708889072, 996179144, 131085828, 165153282, 950820099],
    [536870912, 268435456, 939524096, 872415232, 33554432, 318767104, 8388608, 759169024, 228589568, 816840704, 84410368, 30670848, 117309440, 336396288, 304644096, 723795968, 886497280, 791269376, 519997440, 806161408, 673735168, 203695360, 911662720, 292118208, 333356576, 771604048, 545279864, 1027651572, 899711490, 78705921],
    [536870912, 805306368, 939524096, 335544320, 436207616, 318767104, 494927872, 1035993088, 228589568, 902823936, 569901056, 1003225088, 686424064, 894500864, 149585920, 89243648, 820666368, 290557952, 719980544, 38919168, 929725952, 1066478336, 986335360, 172096576, 456314720, 831329136, 869821640, 81577524, 993185634, 26022771],
    [536870912, 268435456, 402653184, 603979776, 838860800, 486539264, 343932928, 12582912, 987758592, 466616320, 421003264, 918290432, 98959360, 173998080, 694583296, 1050624000, 965140480, 474992640, 102639616, 926131200, 764967424, 645703424, 654709120, 898598976, 41850336, 354648752, 685779576, 382750668, 362321378, 310149809],
    [536870912, 805306368, 671088640, 872415232, 771751936, 16777216, 461373440, 633339904, 1017118720, 997195776, 81264640, 388235264, 919994368, 209256448, 41320448, 856276992, 11411456, 288165888, 66623488, 966005760, 648967680, 338510592, 511222912, 693419712, 797847520, 194901584, 1031799480, 549471412, 555749346, 733013587],
    [536870912, 805306368, 939524096, 201326592, 436207616, 989855744, 142606336, 180355072, 228589568, 659554304, 445120512, 94109696, 326762496, 214106112, 618299392, 546553856, 248668160, 600829952, 168298496, 861674496, 1020963328, 479459072, 752024704, 349149760, 411573920, 43798576, 972724552, 822305852, 1006272162, 906207283],
];

/// Dimensions 2..=31 expanded to the 64-bit direction convention used here:
/// `_sv[dim][bit] << 34` maps SciPy's 30-bit numbers (MSB at 2^29) onto our
/// MSB-at-2^63 words. Bits 30..63 stay zero, capping these dimensions at 2^30
/// samples — exactly SciPy's own Sobol limit.
const SOBOL_SV_EXT: [[u64; 64]; 30] = build_sobol_sv_ext();

const fn build_sobol_sv_ext() -> [[u64; 64]; 30] {
    let mut out = [[0u64; 64]; 30];
    let mut d = 0usize;
    while d < 30 {
        let mut b = 0usize;
        while b < 30 {
            out[d][b] = (SOBOL_SV30_DIMS_2_31[d][b] as u64) << 34;
            b += 1;
        }
        d += 1;
    }
    out
}

/// 64-bit direction words for Sobol dimension `dimension` (`< SOBOL_MAX_DIM`).
fn direction_table(dimension: usize) -> &'static [u64; 64] {
    if dimension < 2 {
        &SOBOL_DIRECTION_TABLES[dimension]
    } else {
        &SOBOL_SV_EXT[dimension - 2]
    }
}

fn sobol_bits(index: u64, dimension: usize) -> u64 {
    if dimension < SOBOL_MAX_DIM {
        return sobol_bits_from_directions(index, direction_table(dimension));
    }

    let mut gray = index ^ (index >> 1);
    let mut bit = 0usize;
    let mut value = 0u64;
    while gray != 0 {
        if gray & 1 == 1 {
            value ^= sobol_direction(dimension, bit);
        }
        gray >>= 1;
        bit += 1;
    }
    value
}

fn sobol_bits_from_directions(index: u64, directions: &[u64; 64]) -> u64 {
    let mut gray = index ^ (index >> 1);
    let mut bit = 0usize;
    let mut value = 0u64;
    while gray != 0 {
        if gray & 1 == 1 {
            value ^= directions[bit];
        }
        gray >>= 1;
        bit += 1;
    }
    value
}

fn sobol_direction(dimension: usize, bit: usize) -> u64 {
    let mut direction = 1u64 << 63;
    if dimension == 0 {
        return direction >> bit.min(63);
    }

    for _ in 0..bit.min(63) {
        direction ^= direction >> 1;
    }
    direction
}

fn bits_to_unit(bits: u64) -> f64 {
    unit_interval_from_u64(bits)
}

// ══════════════════════════════════════════════════════════════════════
// Centered L2 discrepancy
// ══════════════════════════════════════════════════════════════════════

/// Compute the centered L2 discrepancy CD²(P) of an `n × d` point set in
/// `[0, 1)^d`, where `sample` is row-major (`sample[i * d + j]` is the j-th
/// coordinate of the i-th point).
///
/// Matches `scipy.stats.qmc.discrepancy(sample, method='CD')`. The formula is
///
///   CD²(P) = (13/12)^d
///          - (2/N) Σ_i Π_k [ 1 + 0.5·|x_i^k − 0.5|
///                              - 0.5·(x_i^k − 0.5)² ]
///          + (1/N²) Σ_{i,j} Π_k [ 1 + 0.5·|x_i^k − 0.5|
///                                    + 0.5·|x_j^k − 0.5|
///                                    - 0.5·|x_i^k − x_j^k| ]
///
/// Returns `Err(StatsError::InvalidArgument)` when `dimension == 0`, when
/// `sample.len()` is not a multiple of `dimension`, or when any coordinate is
/// outside `[0, 1]`.
#[derive(Clone, Copy)]
struct DiscrepancyPoint2 {
    x0: f64,
    x1: f64,
    centered0: f64,
    centered1: f64,
    abs0: f64,
    abs1: f64,
}

fn discrepancy_points_2d(sample: &[f64], n: usize) -> Vec<DiscrepancyPoint2> {
    let mut points = Vec::with_capacity(n);
    for row in sample.chunks_exact(2) {
        let x0 = row[0];
        let x1 = row[1];
        let centered0 = x0 - 0.5;
        let centered1 = x1 - 0.5;
        points.push(DiscrepancyPoint2 {
            x0,
            x1,
            centered0,
            centered1,
            abs0: centered0.abs(),
            abs1: centered1.abs(),
        });
    }
    points
}

fn centered_discrepancy_2d(sample: &[f64], n: usize) -> f64 {
    let points = discrepancy_points_2d(sample, n);
    let leading = (13.0_f64 / 12.0).powi(2);

    let mut single = 0.0_f64;
    for point in &points {
        let mut prod = 1.0_f64;
        prod *= 1.0 + 0.5 * point.abs0 - 0.5 * point.centered0 * point.centered0;
        prod *= 1.0 + 0.5 * point.abs1 - 0.5 * point.centered1 * point.centered1;
        single += prod;
    }

    let mut double = 0.0_f64;
    for point_i in &points {
        for point_j in &points {
            let mut prod = 1.0_f64;
            prod *= 1.0 + 0.5 * point_i.abs0 + 0.5 * point_j.abs0
                - 0.5 * (point_i.x0 - point_j.x0).abs();
            prod *= 1.0 + 0.5 * point_i.abs1 + 0.5 * point_j.abs1
                - 0.5 * (point_i.x1 - point_j.x1).abs();
            double += prod;
        }
    }

    let n_f = n as f64;
    leading - 2.0 / n_f * single + double / (n_f * n_f)
}

fn mixture_discrepancy_2d(sample: &[f64], n: usize) -> f64 {
    let points = discrepancy_points_2d(sample, n);
    let leading = (19.0_f64 / 12.0).powi(2);

    let mut single = 0.0_f64;
    for point in &points {
        let mut prod = 1.0_f64;
        prod *= 5.0 / 3.0 - 0.25 * point.abs0 - 0.25 * point.centered0 * point.centered0;
        prod *= 5.0 / 3.0 - 0.25 * point.abs1 - 0.25 * point.centered1 * point.centered1;
        single += prod;
    }

    let mut double = 0.0_f64;
    for point_i in &points {
        for point_j in &points {
            let delta0 = point_i.x0 - point_j.x0;
            let d0 = delta0.abs();
            let delta1 = point_i.x1 - point_j.x1;
            let d1 = delta1.abs();
            let mut prod = 1.0_f64;
            prod *= 15.0 / 8.0 - 0.25 * point_i.abs0 - 0.25 * point_j.abs0 - 0.75 * d0
                + 0.5 * delta0.powi(2);
            prod *= 15.0 / 8.0 - 0.25 * point_i.abs1 - 0.25 * point_j.abs1 - 0.75 * d1
                + 0.5 * delta1.powi(2);
            double += prod;
        }
    }

    let n_f = n as f64;
    leading - 2.0 / n_f * single + double / (n_f * n_f)
}

fn l2_star_discrepancy_2d(sample: &[f64], n: usize) -> f64 {
    let points = discrepancy_points_2d(sample, n);
    let leading = (1.0_f64 / 3.0).powi(2);
    let two_pow_one_minus_d = 2.0_f64.powi(-1);

    let mut single = 0.0_f64;
    for point in &points {
        let mut prod = 1.0_f64;
        prod *= 1.0 - point.x0 * point.x0;
        prod *= 1.0 - point.x1 * point.x1;
        single += prod;
    }

    let mut double = 0.0_f64;
    for point_i in &points {
        for point_j in &points {
            let mut prod = 1.0_f64;
            prod *= 1.0 - point_i.x0.max(point_j.x0);
            prod *= 1.0 - point_i.x1.max(point_j.x1);
            double += prod;
        }
    }

    let n_f = n as f64;
    (leading - two_pow_one_minus_d / n_f * single + double / (n_f * n_f)).sqrt()
}

fn wraparound_discrepancy_2d(sample: &[f64], n: usize) -> f64 {
    let points = discrepancy_points_2d(sample, n);
    let leading = -(4.0_f64 / 3.0).powi(2);

    let mut double = 0.0_f64;
    for point_i in &points {
        for point_j in &points {
            let d0 = (point_i.x0 - point_j.x0).abs();
            let d1 = (point_i.x1 - point_j.x1).abs();
            let mut prod = 1.0_f64;
            prod *= 1.5 - d0 * (1.0 - d0);
            prod *= 1.5 - d1 * (1.0 - d1);
            double += prod;
        }
    }

    let n_f = n as f64;
    leading + double / (n_f * n_f)
}

pub fn centered_discrepancy(sample: &[f64], dimension: usize) -> Result<f64, StatsError> {
    if dimension == 0 {
        return Err(StatsError::InvalidArgument(
            "centered_discrepancy: dimension must be ≥ 1".to_string(),
        ));
    }
    if !sample.len().is_multiple_of(dimension) {
        return Err(StatsError::InvalidArgument(format!(
            "centered_discrepancy: sample.len() {} not a multiple of dimension {dimension}",
            sample.len()
        )));
    }
    let n = sample.len() / dimension;
    if n == 0 {
        return Ok((13.0_f64 / 12.0).powi(dimension as i32));
    }
    for (idx, &v) in sample.iter().enumerate() {
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return Err(StatsError::InvalidArgument(format!(
                "centered_discrepancy: sample[{idx}] = {v} outside [0, 1]"
            )));
        }
    }
    if dimension == 2 {
        return Ok(centered_discrepancy_2d(sample, n));
    }

    let leading = (13.0_f64 / 12.0).powi(dimension as i32);

    // Single-sum term Σ_i Π_k [1 + 0.5|x − 0.5| - 0.5 (x − 0.5)²].
    let mut single = 0.0_f64;
    for i in 0..n {
        let mut prod = 1.0_f64;
        for k in 0..dimension {
            let x = sample[i * dimension + k];
            let centered = x - 0.5;
            prod *= 1.0 + 0.5 * centered.abs() - 0.5 * centered * centered;
        }
        single += prod;
    }

    // Double-sum term Σ_{i,j} Π_k [...]. Quadratic in n; QMC users typically
    // call this on samples with n ≤ a few thousand.
    let mut double = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let mut prod = 1.0_f64;
            for k in 0..dimension {
                let xi = sample[i * dimension + k];
                let xj = sample[j * dimension + k];
                prod *=
                    1.0 + 0.5 * (xi - 0.5).abs() + 0.5 * (xj - 0.5).abs() - 0.5 * (xi - xj).abs();
            }
            double += prod;
        }
    }

    let n_f = n as f64;
    Ok(leading - 2.0 / n_f * single + double / (n_f * n_f))
}

/// Compute the mixture L2 discrepancy MD²(P) of an `n × d` point set in
/// `[0, 1)^d`, with `sample` row-major.
///
/// Matches `scipy.stats.qmc.discrepancy(sample, method='MD')`. The formula
///
///   MD² = (19/12)^d
///       − (2/N) Σ_i  Π_k [ 5/3 − 1/4·|xi^k − 1/2| − 1/4·(xi^k − 1/2)² ]
///       + (1/N²) Σ_{i,j} Π_k [ 15/8 − 1/4·|xi^k − 1/2| − 1/4·|xj^k − 1/2|
///                                  − 3/4·|xi^k − xj^k|
///                                  + 1/2·(xi^k − xj^k)² ]
///
/// is a "mix" of centered- and wraparound-style weighting and is one of the
/// four standard L2 discrepancy metrics in scipy.stats.qmc.
pub fn mixture_discrepancy(sample: &[f64], dimension: usize) -> Result<f64, StatsError> {
    if dimension == 0 {
        return Err(StatsError::InvalidArgument(
            "mixture_discrepancy: dimension must be ≥ 1".to_string(),
        ));
    }
    if !sample.len().is_multiple_of(dimension) {
        return Err(StatsError::InvalidArgument(format!(
            "mixture_discrepancy: sample.len() {} not a multiple of dimension {dimension}",
            sample.len()
        )));
    }
    let n = sample.len() / dimension;
    if n == 0 {
        return Ok((19.0_f64 / 12.0).powi(dimension as i32));
    }
    for (idx, &v) in sample.iter().enumerate() {
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return Err(StatsError::InvalidArgument(format!(
                "mixture_discrepancy: sample[{idx}] = {v} outside [0, 1]"
            )));
        }
    }
    if dimension == 2 {
        return Ok(mixture_discrepancy_2d(sample, n));
    }
    let leading = (19.0_f64 / 12.0).powi(dimension as i32);
    // Single-sum term.
    let mut single = 0.0_f64;
    for i in 0..n {
        let mut prod = 1.0_f64;
        for k in 0..dimension {
            let x = sample[i * dimension + k];
            let centered = x - 0.5;
            prod *= 5.0 / 3.0 - 0.25 * centered.abs() - 0.25 * centered * centered;
        }
        single += prod;
    }
    // Double-sum term.
    let mut double = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let mut prod = 1.0_f64;
            for k in 0..dimension {
                let xi = sample[i * dimension + k];
                let xj = sample[j * dimension + k];
                let d = (xi - xj).abs();
                prod *= 15.0 / 8.0 - 0.25 * (xi - 0.5).abs() - 0.25 * (xj - 0.5).abs() - 0.75 * d
                    + 0.5 * (xi - xj).powi(2);
            }
            double += prod;
        }
    }
    let n_f = n as f64;
    Ok(leading - 2.0 / n_f * single + double / (n_f * n_f))
}

/// Linearly scale a unit-cube `[0, 1)^d` design into the box `[l_bounds, u_bounds]`.
///
/// `sample` is row-major (`sample[i * d + j]` is the j-th coordinate of the
/// i-th point). The result has the same shape; coordinate `j` is mapped via
///   `out_j = l_bounds[j] + (u_bounds[j] - l_bounds[j]) * sample_j`.
///
/// Matches the *forward* direction of `scipy.stats.qmc.scale(sample,
/// l_bounds, u_bounds)`. Returns `Err(StatsError::InvalidArgument)` when the
/// bound vectors disagree with `dimension`, when any non-NaN coordinate is
/// outside `[0, 1]`, or when any `u_bounds[j] < l_bounds[j]`.
pub fn scale(
    sample: &[f64],
    dimension: usize,
    l_bounds: &[f64],
    u_bounds: &[f64],
) -> Result<Vec<f64>, StatsError> {
    if dimension == 0 {
        return Err(StatsError::InvalidArgument(
            "qmc::scale: dimension must be ≥ 1".to_string(),
        ));
    }
    if l_bounds.len() != dimension || u_bounds.len() != dimension {
        return Err(StatsError::InvalidArgument(format!(
            "qmc::scale: bounds vectors of length ({}, {}) do not match dimension {dimension}",
            l_bounds.len(),
            u_bounds.len()
        )));
    }
    if !sample.len().is_multiple_of(dimension) {
        return Err(StatsError::InvalidArgument(format!(
            "qmc::scale: sample.len() {} not a multiple of dimension {dimension}",
            sample.len()
        )));
    }
    for k in 0..dimension {
        if !l_bounds[k].is_finite() || !u_bounds[k].is_finite() {
            return Err(StatsError::InvalidArgument(format!(
                "qmc::scale: non-finite bound at dim {k}"
            )));
        }
        if u_bounds[k] < l_bounds[k] {
            return Err(StatsError::InvalidArgument(format!(
                "qmc::scale: u_bounds[{k}]={} < l_bounds[{k}]={}",
                u_bounds[k], l_bounds[k]
            )));
        }
    }
    let n = sample.len() / dimension;
    let mut out = Vec::with_capacity(sample.len());
    for i in 0..n {
        for k in 0..dimension {
            let value = sample[i * dimension + k];
            if !value.is_nan() && !(0.0..=1.0).contains(&value) {
                return Err(StatsError::InvalidArgument(
                    "qmc::scale: sample is not in unit hypercube".to_string(),
                ));
            }
            let scale_k = u_bounds[k] - l_bounds[k];
            out.push(l_bounds[k] + scale_k * value);
        }
    }
    Ok(out)
}

/// Update the centered L2 discrepancy `existing_disc` to reflect a new point
/// appended to `existing_sample`, in O(N·d) instead of an O((N+1)²·d) full
/// recomputation.
///
/// Matches `scipy.stats.qmc.update_discrepancy(x_new, sample, initial_disc)`:
/// the discrepancy of the augmented design is the prior `existing_disc` plus
/// three correction terms, each normalized by `N = n_old + 1`,
///
///   disc1 = −(2/N)·s1(z),
///   disc2 = (2/N²)·Σ_i s2(x_i, z),
///   disc3 = (1/N²)·s2(z, z),
///
/// with `s1(x) = Π_k [1 + ½|x_k−½| − ½(x_k−½)²]` and
/// `s2(a,b) = Π_k [1 + ½|a_k−½| + ½|b_k−½| − ½|a_k−b_k|]`.
pub fn update_centered_discrepancy(
    existing_sample: &[f64],
    dimension: usize,
    existing_disc: f64,
    new_point: &[f64],
) -> Result<f64, StatsError> {
    if dimension == 0 {
        return Err(StatsError::InvalidArgument(
            "update_centered_discrepancy: dimension must be ≥ 1".to_string(),
        ));
    }
    if !existing_sample.len().is_multiple_of(dimension) {
        return Err(StatsError::InvalidArgument(format!(
            "update_centered_discrepancy: existing_sample.len() {} not a multiple of dimension {dimension}",
            existing_sample.len()
        )));
    }
    if new_point.len() != dimension {
        return Err(StatsError::InvalidArgument(format!(
            "update_centered_discrepancy: new_point length {} != dimension {dimension}",
            new_point.len()
        )));
    }
    for &v in existing_sample.iter().chain(new_point.iter()) {
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return Err(StatsError::InvalidArgument(format!(
                "update_centered_discrepancy: value {v} outside [0, 1]"
            )));
        }
    }
    let n_old = existing_sample.len() / dimension;
    let n_new = (n_old + 1) as f64;

    // s1(x) = Π_k [1 + ½|x_k−½| − ½(x_k−½)²].
    let s1 = |point: &[f64]| -> f64 {
        let mut prod = 1.0_f64;
        for &coordinate in point.iter().take(dimension) {
            let centered = coordinate - 0.5;
            prod *= 1.0 + 0.5 * centered.abs() - 0.5 * centered * centered;
        }
        prod
    };
    // s2(a,b) = Π_k [1 + ½|a_k−½| + ½|b_k−½| − ½|a_k−b_k|].
    let s2_pair = |a: &[f64], b: &[f64]| -> f64 {
        let mut prod = 1.0_f64;
        for k in 0..dimension {
            prod *= 1.0 + 0.5 * (a[k] - 0.5).abs() + 0.5 * (b[k] - 0.5).abs()
                - 0.5 * (a[k] - b[k]).abs();
        }
        prod
    };

    // scipy's additive correction (see the doc comment above). The cross
    // loop is empty for an initially empty sample, so this also handles the
    // singleton-extension case uniformly.
    let s1z = s1(new_point);
    let mut cross = 0.0_f64;
    for i in 0..n_old {
        let row = &existing_sample[i * dimension..(i + 1) * dimension];
        cross += s2_pair(row, new_point);
    }
    let self_term = s2_pair(new_point, new_point);

    let disc1 = -2.0 / n_new * s1z;
    let disc2 = 2.0 / (n_new * n_new) * cross;
    let disc3 = self_term / (n_new * n_new);
    Ok(existing_disc + disc1 + disc2 + disc3)
}

/// Compute the L2-star discrepancy SD(P) of an `n × d` point set in
/// `[0, 1)^d`, with `sample` row-major.
///
/// Matches `scipy.stats.qmc.discrepancy(sample, method='L2-star')`, which
/// returns the (non-squared) discrepancy — the square root of the
/// Hickernell closed form:
///
///   SD² = (1/3)^d
///       − (2^(1−d) / N) Σ_i Π_k (1 − (xi^k)²)
///       + (1/N²) Σ_{i,j} Π_k (1 − max(xi^k, xj^k))
///   SD  = √(SD²)
///
/// SD weights how well the design covers the lower-left orthants
/// `[0, x]^d` and is the most common deterministic-sequence quality
/// measure in the QMC literature.
pub fn l2_star_discrepancy(sample: &[f64], dimension: usize) -> Result<f64, StatsError> {
    if dimension == 0 {
        return Err(StatsError::InvalidArgument(
            "l2_star_discrepancy: dimension must be ≥ 1".to_string(),
        ));
    }
    if !sample.len().is_multiple_of(dimension) {
        return Err(StatsError::InvalidArgument(format!(
            "l2_star_discrepancy: sample.len() {} not a multiple of dimension {dimension}",
            sample.len()
        )));
    }
    let n = sample.len() / dimension;
    if n == 0 {
        // No points: SD² collapses to the leading (1/3)^d term.
        return Ok((1.0_f64 / 3.0).powi(dimension as i32).sqrt());
    }
    for (idx, &v) in sample.iter().enumerate() {
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return Err(StatsError::InvalidArgument(format!(
                "l2_star_discrepancy: sample[{idx}] = {v} outside [0, 1]"
            )));
        }
    }
    if dimension == 2 {
        return Ok(l2_star_discrepancy_2d(sample, n));
    }
    let leading = (1.0_f64 / 3.0).powi(dimension as i32);
    let two_pow_one_minus_d = 2.0_f64.powi(1 - dimension as i32);
    // Single-sum Σ_i Π_k (1 - x_i^k²).
    let mut single = 0.0_f64;
    for i in 0..n {
        let mut prod = 1.0_f64;
        for k in 0..dimension {
            let x = sample[i * dimension + k];
            prod *= 1.0 - x * x;
        }
        single += prod;
    }
    // Double-sum Σ_ij Π_k (1 - max(x_i^k, x_j^k)).
    let mut double = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let mut prod = 1.0_f64;
            for k in 0..dimension {
                let xi = sample[i * dimension + k];
                let xj = sample[j * dimension + k];
                prod *= 1.0 - xi.max(xj);
            }
            double += prod;
        }
    }
    let n_f = n as f64;
    // scipy.stats.qmc.discrepancy returns the square root of the SD² form.
    Ok((leading - two_pow_one_minus_d / n_f * single + double / (n_f * n_f)).sqrt())
}

/// Compute the wraparound L2 discrepancy WD²(P) of an `n × d` point set in
/// `[0, 1)^d`, with `sample` row-major (`sample[i * d + j]` is coordinate j
/// of point i).
///
/// Matches `scipy.stats.qmc.discrepancy(sample, method='WD')`. The formula
///
///   WD² = -(4/3)^d
///       + (1/N²) Σ_{i,j} Π_k [ 3/2 - |x_i^k − x_j^k| · (1 − |x_i^k − x_j^k|) ]
///
/// is invariant under cyclic shifts of any dimension's coordinates — so it
/// captures the design's quality on the unit *torus*, complementing the
/// centered discrepancy which weights points near the cube's center.
///
/// Returns `Err(StatsError::InvalidArgument)` for the same reasons as
/// `centered_discrepancy`.
pub fn wraparound_discrepancy(sample: &[f64], dimension: usize) -> Result<f64, StatsError> {
    if dimension == 0 {
        return Err(StatsError::InvalidArgument(
            "wraparound_discrepancy: dimension must be ≥ 1".to_string(),
        ));
    }
    if !sample.len().is_multiple_of(dimension) {
        return Err(StatsError::InvalidArgument(format!(
            "wraparound_discrepancy: sample.len() {} not a multiple of dimension {dimension}",
            sample.len()
        )));
    }
    let n = sample.len() / dimension;
    if n == 0 {
        return Ok(-(4.0_f64 / 3.0).powi(dimension as i32));
    }
    for (idx, &v) in sample.iter().enumerate() {
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return Err(StatsError::InvalidArgument(format!(
                "wraparound_discrepancy: sample[{idx}] = {v} outside [0, 1]"
            )));
        }
    }
    if dimension == 2 {
        return Ok(wraparound_discrepancy_2d(sample, n));
    }
    let leading = -(4.0_f64 / 3.0).powi(dimension as i32);
    let mut double = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let mut prod = 1.0_f64;
            for k in 0..dimension {
                let xi = sample[i * dimension + k];
                let xj = sample[j * dimension + k];
                let d = (xi - xj).abs();
                prod *= 1.5 - d * (1.0 - d);
            }
            double += prod;
        }
    }
    let n_f = n as f64;
    Ok(leading + double / (n_f * n_f))
}

// ══════════════════════════════════════════════════════════════════════
// Latin Hypercube Sampling
// ══════════════════════════════════════════════════════════════════════

/// Latin Hypercube Sampling (LHS) over `[0, 1)^d`.
///
/// For each dimension, the unit interval is partitioned into `n` equal-width
/// strata; exactly one sample is placed in each stratum. The per-dimension
/// stratum order is a uniform random permutation. This is the construction
/// used by `scipy.stats.qmc.LatinHypercube`.
///
/// Two modes:
/// - `centered = true` (default): each sample sits at the midpoint of its
///   stratum, i.e. `(perm_j[i] + 0.5) / n`. Deterministic given the seed.
/// - `centered = false`: each sample is drawn uniformly inside its stratum,
///   i.e. `(perm_j[i] + u) / n` for `u ∈ [0, 1)`.
#[derive(Debug, Clone)]
pub struct LatinHypercubeSampler {
    dimension: usize,
    centered: bool,
    rng_state: u64,
}

impl LatinHypercubeSampler {
    /// Construct a centered LHS sampler in `dimension`-dimensional unit
    /// hypercube, seeded so subsequent `sample(n)` calls are reproducible.
    pub fn new(dimension: usize, seed: u64) -> Result<Self, StatsError> {
        if dimension == 0 {
            return Err(StatsError::InvalidArgument(
                "LatinHypercube dimension must be ≥ 1".to_string(),
            ));
        }
        Ok(Self {
            dimension,
            centered: true,
            // Derive the live RNG state from the seed via splitmix mixing so
            // seed=0 produces a non-trivial state.
            rng_state: splitmix64(seed.wrapping_add(0x9E37_79B9_7F4A_7C15)),
        })
    }

    /// Switch between centered (cell midpoint) and randomized (uniform within
    /// cell) modes. Returns the previous setting.
    pub fn set_centered(&mut self, centered: bool) -> bool {
        let prev = self.centered;
        self.centered = centered;
        prev
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Draw an `n × d` LHS design, row-major (`out[i*d + j]` is sample i,
    /// dimension j).
    pub fn sample(&mut self, n: usize) -> Vec<f64> {
        let d = self.dimension;
        let mut out = vec![0.0_f64; n.saturating_mul(d)];
        if n == 0 {
            return out;
        }
        let inv_n = 1.0_f64 / n as f64;
        for j in 0..d {
            let perm = self.permutation(n);
            for i in 0..n {
                let stratum = perm[i] as f64;
                let u = if self.centered {
                    0.5
                } else {
                    self.next_uniform()
                };
                out[i * d + j] = (stratum + u) * inv_n;
            }
        }
        out
    }

    fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut perm: Vec<usize> = (0..n).collect();
        // Fisher-Yates with the internal splitmix RNG.
        for i in (1..n).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        perm
    }

    fn next_u64(&mut self) -> u64 {
        self.rng_state = self.rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        splitmix64(self.rng_state)
    }

    fn next_uniform(&mut self) -> f64 {
        unit_interval_from_u64(self.next_u64())
    }
}

// ══════════════════════════════════════════════════════════════════════
// Poisson-disk sampling
// ══════════════════════════════════════════════════════════════════════

/// Two-dimensional Poisson-disk sampler over the unit square.
///
/// This is a deterministic dart-throwing implementation: candidate points are
/// drawn from a SplitMix stream and accepted only when they are at least
/// `radius` away from every previous point. It is intentionally compact and
/// bounded for conformance and metamorphic tests; callers requesting more
/// points than the radius permits may receive fewer than `n` points.
#[derive(Debug, Clone)]
pub struct PoissonDiskSampler {
    radius: f64,
    rng_state: u64,
}

impl PoissonDiskSampler {
    pub fn new_2d(radius: f64, seed: u64) -> Result<Self, StatsError> {
        if !radius.is_finite() || radius <= 0.0 || radius >= 1.0 {
            return Err(StatsError::InvalidArgument(
                "PoissonDisk radius must be finite and in (0, 1)".to_string(),
            ));
        }
        Ok(Self {
            radius,
            rng_state: splitmix64(seed.wrapping_add(0xA5A5_A5A5_5A5A_5A5A)),
        })
    }

    pub fn dimension(&self) -> usize {
        2
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Draw up to `n` points in row-major `[x0, y0, x1, y1, ...]` order.
    pub fn sample(&mut self, n: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(n.saturating_mul(2));
        if n == 0 {
            return out;
        }

        let min_dist2 = self.radius * self.radius;
        let max_attempts = n.saturating_mul(2_000).max(2_000);
        for _ in 0..max_attempts {
            if out.len() / 2 == n {
                break;
            }
            let x = self.next_uniform();
            let y = self.next_uniform();
            if poisson_candidate_is_valid(&out, x, y, min_dist2) {
                out.push(x);
                out.push(y);
            }
        }
        out
    }

    fn next_u64(&mut self) -> u64 {
        self.rng_state = self.rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        splitmix64(self.rng_state)
    }

    fn next_uniform(&mut self) -> f64 {
        unit_interval_from_u64(self.next_u64())
    }
}

fn poisson_candidate_is_valid(sample: &[f64], x: f64, y: f64, min_dist2: f64) -> bool {
    for point in sample.chunks_exact(2) {
        let dx = x - point[0];
        let dy = y - point[1];
        if dx.mul_add(dx, dy * dy) < min_dist2 {
            return false;
        }
    }
    true
}

fn unit_interval_from_u64(bits: u64) -> f64 {
    // 53-bit mantissa division: standard f64 [0,1) sample.
    (bits >> 11) as f64 * (1.0_f64 / (1u64 << 53) as f64)
}

fn splitmix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Method selector for [`geometric_discrepancy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometricDiscrepancyMethod {
    /// Minimum pairwise Euclidean distance (default).
    MinDist,
    /// Mean edge length of the minimum spanning tree built on the sample.
    Mst,
}

/// Geometric discrepancy of a sample. Higher values correspond to greater
/// sample uniformity (the inverse polarity from the standard discrepancies
/// in this module).
///
/// Matches `scipy.stats.qmc.geometric_discrepancy(sample, method, metric='euclidean')`.
/// The `metric` parameter on the scipy side is currently fixed to euclidean
/// here — sufficient for the QMC sample-quality use cases. The sample buffer
/// is laid out row-major as `n × dimension` (the same convention the rest of
/// the qmc module uses for its `sample(n)` outputs).
///
/// `MinDist` returns `min_{i<j} ||x_i − x_j||`. `Mst` builds the
/// fully-connected weighted graph on the sample with edge weights = pairwise
/// Euclidean distance and returns the mean edge length of its minimum
/// spanning tree (Prim's algorithm; n ≤ a few thousand is the comfortable
/// regime — the algorithm is O(n²) in time and memory).
pub fn geometric_discrepancy(
    sample: &[f64],
    dimension: usize,
    method: GeometricDiscrepancyMethod,
) -> Result<f64, StatsError> {
    if dimension == 0 {
        return Err(StatsError::InvalidArgument("dimension must be >= 1".into()));
    }
    if !sample.len().is_multiple_of(dimension) {
        return Err(StatsError::InvalidArgument(format!(
            "sample length {} is not a multiple of dimension {dimension}",
            sample.len()
        )));
    }
    let n = sample.len() / dimension;
    if n < 2 {
        return Err(StatsError::InvalidArgument(
            "sample must contain at least two points".into(),
        ));
    }
    // Sample must lie in the unit hypercube — scipy raises a ValueError for
    // out-of-range inputs. Mirror that.
    if sample
        .iter()
        .any(|v| !(0.0..=1.0).contains(v) || !v.is_finite())
    {
        return Err(StatsError::InvalidArgument(
            "sample must lie in [0, 1] with finite coordinates".into(),
        ));
    }

    let row = |i: usize| &sample[i * dimension..(i + 1) * dimension];
    let euclid = |i: usize, j: usize| -> f64 {
        let a = row(i);
        let b = row(j);
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    };

    match method {
        GeometricDiscrepancyMethod::MinDist => {
            let mut best = f64::INFINITY;
            for i in 0..n {
                for j in (i + 1)..n {
                    let d = euclid(i, j);
                    if d > 0.0 && d < best {
                        best = d;
                    }
                }
            }
            if !best.is_finite() {
                // All pairs collapsed onto a duplicate — match scipy's behaviour
                // of returning the smallest (zero) distance.
                Ok(0.0)
            } else {
                Ok(best)
            }
        }
        GeometricDiscrepancyMethod::Mst => {
            // Prim's algorithm on the dense weighted graph. O(n²) time and
            // O(n) auxiliary memory. Returns the mean MST edge length.
            let mut in_tree = vec![false; n];
            let mut min_edge = vec![f64::INFINITY; n];
            in_tree[0] = true;
            for (j, edge) in min_edge.iter_mut().enumerate().take(n).skip(1) {
                *edge = euclid(0, j);
            }
            let mut total = 0.0_f64;
            for _ in 1..n {
                // Pick the cheapest fringe vertex.
                let mut u = usize::MAX;
                let mut best = f64::INFINITY;
                for j in 0..n {
                    if !in_tree[j] && min_edge[j] < best {
                        best = min_edge[j];
                        u = j;
                    }
                }
                if u == usize::MAX {
                    // Sample reduces to fewer than 2 distinct points; fall back.
                    return Ok(0.0);
                }
                in_tree[u] = true;
                total += best;
                // Relax the fringe.
                for j in 0..n {
                    if !in_tree[j] {
                        let w = euclid(u, j);
                        if w < min_edge[j] {
                            min_edge[j] = w;
                        }
                    }
                }
            }
            Ok(total / (n - 1) as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halton_construct_rejects_zero_dimension() {
        let err = HaltonSampler::new(0).expect_err("zero-dim must fail");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn halton_construct_rejects_excess_dimension() {
        let err = HaltonSampler::new(33).expect_err("33-dim must fail");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn halton_sample_first_2d_canonical() {
        // Canonical first-five Halton points in 2D (bases 2 and 3):
        //   index 0 → (0, 0)
        //   index 1 → (1/2, 1/3)
        //   index 2 → (1/4, 2/3)
        //   index 3 → (3/4, 1/9)
        //   index 4 → (1/8, 4/9)
        let mut h = HaltonSampler::new(2).unwrap();
        let s = h.sample(5);
        let expected = [
            (0.0, 0.0),
            (1.0 / 2.0, 1.0 / 3.0),
            (1.0 / 4.0, 2.0 / 3.0),
            (3.0 / 4.0, 1.0 / 9.0),
            (1.0 / 8.0, 4.0 / 9.0),
        ];
        for (i, (ex, ey)) in expected.iter().enumerate() {
            assert!((s[i * 2] - ex).abs() < 1e-15, "x[{i}]");
            assert!((s[i * 2 + 1] - ey).abs() < 1e-15, "y[{i}]");
        }
    }

    #[test]
    fn halton_sample_in_unit_hypercube() {
        let mut h = HaltonSampler::new(5).unwrap();
        let n = 200;
        let s = h.sample(n);
        for v in &s {
            assert!(*v >= 0.0 && *v < 1.0, "value {v} out of [0,1)");
        }
        assert_eq!(s.len(), n * 5);
    }

    #[test]
    fn halton_metamorphic_skip_equivalent_to_consume() {
        let mut a = HaltonSampler::new(3).unwrap();
        let mut b = HaltonSampler::new(3).unwrap();
        let _ = a.sample(7);
        b.skip(7);
        let from_a = a.sample(5);
        let from_b = b.sample(5);
        assert_eq!(from_a, from_b);
    }

    #[test]
    fn halton_metamorphic_reset_replays_prefix() {
        let mut h = HaltonSampler::new(2).unwrap();
        let first = h.sample(10);
        h.reset();
        let second = h.sample(10);
        assert_eq!(first, second);
    }

    #[test]
    fn halton_metamorphic_low_discrepancy_beats_uniform() {
        // Halton's star discrepancy is bounded by C·log(N)^d / N. For d=2 and
        // N=1024, that bound is ~0.07 with C ≈ 1; uniform i.i.d. has
        // E[D*] ~ 1/sqrt(N) ≈ 0.031 expected but the *worst-case* L∞ box
        // discrepancy of any deterministic uniform point set is much higher.
        // A safe relation: for the unit-square box [0, 0.5]^2, the Halton
        // count fraction must agree with the box volume (0.25) within 5%
        // for N ≥ 200, while a deterministic offset-grid sample might not.
        let mut h = HaltonSampler::new(2).unwrap();
        let s = h.sample(1024);
        let in_box = (0..1024)
            .filter(|&i| s[i * 2] < 0.5 && s[i * 2 + 1] < 0.5)
            .count();
        let frac = in_box as f64 / 1024.0;
        assert!(
            (frac - 0.25).abs() < 0.05,
            "Halton box-count fraction {frac} should be within 5% of 0.25"
        );
    }

    #[test]
    fn halton_index_advances_with_each_sample() {
        let mut h = HaltonSampler::new(2).unwrap();
        assert_eq!(h.next_index(), 0);
        let _ = h.sample(3);
        assert_eq!(h.next_index(), 3);
    }

    #[test]
    fn halton_4d_specialization_matches_generic_reference_bits() {
        fn reference(start: u64, n: usize) -> (Vec<f64>, u64) {
            let mut out = Vec::with_capacity(n * 4);
            let mut idx = start;
            for _ in 0..n {
                out.push(radical_inverse(idx, 2));
                out.push(radical_inverse(idx, 3));
                out.push(radical_inverse(idx, 5));
                out.push(radical_inverse(idx, 7));
                idx = idx.saturating_add(1);
            }
            (out, idx)
        }

        for start in [0, 1, 4_095, 4_096, 1_000_003, u64::MAX - 2] {
            let mut h = HaltonSampler::new(4).unwrap();
            h.next_index = start;
            let got = h.sample(5);
            let got_next = h.next_index();
            let (expected, expected_next) = reference(start, 5);

            assert_eq!(
                got_next, expected_next,
                "next_index mismatch at start={start}"
            );
            assert_eq!(
                got.len(),
                expected.len(),
                "length mismatch at start={start}"
            );
            for (i, (actual, expected)) in got.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "4D Halton specialization changed bits at start={start}, value={i}"
                );
            }
        }
    }

    #[test]
    fn halton_dimension_query() {
        let h = HaltonSampler::new(7).unwrap();
        assert_eq!(h.dimension(), 7);
    }

    #[test]
    fn sobol_construct_rejects_unsupported_dimensions() {
        assert!(matches!(
            SobolSampler::new(0),
            Err(StatsError::InvalidArgument(_))
        ));
        // Dimensions 1..=32 are supported; only past the cap is rejected.
        assert!(SobolSampler::new(3).is_ok());
        assert!(SobolSampler::new(SOBOL_MAX_DIM).is_ok());
        assert!(matches!(
            SobolSampler::new(SOBOL_MAX_DIM + 1),
            Err(StatsError::InvalidArgument(_))
        ));
    }

    #[test]
    fn sobol_multidim_matches_scipy_unscrambled_reference() {
        // scipy.stats.qmc.Sobol(d, scramble=False).random(4), the multi-dimension
        // direction numbers we now embed (dims 2..=31 from scipy's _sv). These are
        // exact dyadic rationals, so equality is bit-exact.
        let d3: [[f64; 3]; 4] = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.75],
        ];
        let mut s3 = SobolSampler::new(3).unwrap();
        let got3 = s3.sample(4);
        for (i, row) in d3.iter().enumerate() {
            for (j, &want) in row.iter().enumerate() {
                assert_eq!(got3[i * 3 + j].to_bits(), want.to_bits(), "d3 [{i}][{j}]");
            }
        }

        let d5: [[f64; 5]; 4] = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.75, 0.25, 0.25, 0.25, 0.75],
            [0.25, 0.75, 0.75, 0.75, 0.25],
        ];
        let mut s5 = SobolSampler::new(5).unwrap();
        let got5 = s5.sample(4);
        for (i, row) in d5.iter().enumerate() {
            for (j, &want) in row.iter().enumerate() {
                assert_eq!(got5[i * 5 + j].to_bits(), want.to_bits(), "d5 [{i}][{j}]");
            }
        }
    }

    #[test]
    fn sobol_sample_first_2d_canonical() {
        let mut sobol = SobolSampler::new(2).unwrap();
        let sample = sobol.sample(5);
        let expected = [
            (0.0, 0.0),
            (0.5, 0.5),
            (0.75, 0.25),
            (0.25, 0.75),
            (0.375, 0.375),
        ];
        for (i, (x, y)) in expected.iter().enumerate() {
            assert!((sample[i * 2] - x).abs() < 1e-15, "x[{i}]");
            assert!((sample[i * 2 + 1] - y).abs() < 1e-15, "y[{i}]");
        }
    }

    #[test]
    fn sobol_cached_direction_bits_match_recurrence() {
        let indices = [0, 1, 2, 3, 4, 31, 32, 4_095, 4_096, 1_000_003, u64::MAX - 2];
        for dimension in 0..2 {
            for index in indices {
                let mut gray = index ^ (index >> 1);
                let mut bit = 0usize;
                let mut expected = 0u64;
                while gray != 0 {
                    if gray & 1 == 1 {
                        expected ^= sobol_direction(dimension, bit);
                    }
                    gray >>= 1;
                    bit += 1;
                }
                assert_eq!(sobol_bits(index, dimension), expected);
            }
        }

        let mut shifted = SobolSampler::with_digital_shift(2, 99).unwrap();
        shifted.skip(4_095);
        let sample = shifted.sample(8);
        let mut expected = Vec::with_capacity(sample.len());
        for idx in 4_095..4_103 {
            for dim in 0..2 {
                expected.push(bits_to_unit(
                    sobol_bits(idx, dim) ^ splitmix64(99 + dim as u64),
                ));
            }
        }
        assert_eq!(sample, expected);
    }

    #[test]
    fn sobol_2d_incremental_matches_direct_bits() {
        for (start, n) in [(0_u64, 16_usize), (1, 16), (4_095, 17), (u64::MAX - 2, 5)] {
            let shift0 = splitmix64(0x5eed_u64);
            let shift1 = splitmix64(0x5eed_u64.wrapping_add(1));
            let shifts = [shift0, shift1];
            let mut sampler = SobolSampler::with_shift_words(2, vec![shift0, shift1]).unwrap();
            sampler.skip(start);

            let sample = sampler.sample(n);
            let mut expected = Vec::with_capacity(n.saturating_mul(2));
            let mut idx = start;
            for _ in 0..n {
                for (dimension, &shift) in shifts.iter().enumerate() {
                    expected.push(bits_to_unit(sobol_bits(idx, dimension) ^ shift));
                }
                idx = idx.saturating_add(1);
            }

            assert_eq!(sample, expected);
            assert_eq!(sampler.next_index(), idx);
        }
    }

    #[test]
    fn sobol_metamorphic_skip_equivalent_to_consume() {
        let mut a = SobolSampler::new(2).unwrap();
        let mut b = SobolSampler::new(2).unwrap();
        let _ = a.sample(11);
        b.skip(11);
        assert_eq!(a.sample(8), b.sample(8));
    }

    #[test]
    fn sobol_digital_shift_is_reproducible_and_bounded() {
        let mut a = SobolSampler::with_digital_shift(2, 99).unwrap();
        let mut b = SobolSampler::with_digital_shift(2, 99).unwrap();
        let shifted = a.sample(16);
        assert_eq!(shifted, b.sample(16));
        assert_ne!(shifted, SobolSampler::new(2).unwrap().sample(16));
        for value in shifted {
            assert!((0.0..1.0).contains(&value), "shifted value {value}");
        }
    }

    #[test]
    fn qmc_engine_trait_dispatches_halton_and_sobol() {
        fn draw_prefix(engine: &mut dyn QmcEngine, n: usize) -> Vec<f64> {
            engine.sample(n)
        }

        let mut halton = HaltonSampler::new(2).unwrap();
        let mut sobol = SobolSampler::new(2).unwrap();
        assert_eq!(draw_prefix(&mut halton, 2), vec![0.0, 0.0, 0.5, 1.0 / 3.0]);
        assert_eq!(draw_prefix(&mut sobol, 2), vec![0.0, 0.0, 0.5, 0.5]);
    }

    #[test]
    fn lhs_construct_rejects_zero_dimension() {
        let err = LatinHypercubeSampler::new(0, 0).expect_err("zero-dim must fail");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn lhs_metamorphic_one_per_stratum_centered() {
        // Centered LHS: every sample's coordinate is (k + 0.5) / n for some
        // integer k ∈ [0, n). Each dimension must hit every k exactly once.
        let n = 10;
        let d = 4;
        let mut lhs = LatinHypercubeSampler::new(d, 42).unwrap();
        let s = lhs.sample(n);
        for j in 0..d {
            let mut strata: Vec<i64> = (0..n)
                .map(|i| (s[i * d + j] * n as f64 - 0.5).round() as i64)
                .collect();
            strata.sort_unstable();
            assert_eq!(
                strata,
                (0..n as i64).collect::<Vec<_>>(),
                "dim {j}: each stratum must appear exactly once"
            );
        }
    }

    #[test]
    fn lhs_metamorphic_seed_reproducibility() {
        let mut a = LatinHypercubeSampler::new(3, 1234).unwrap();
        let mut b = LatinHypercubeSampler::new(3, 1234).unwrap();
        assert_eq!(a.sample(20), b.sample(20));
    }

    #[test]
    fn lhs_metamorphic_distinct_seeds_diverge() {
        let mut a = LatinHypercubeSampler::new(3, 1).unwrap();
        let mut b = LatinHypercubeSampler::new(3, 2).unwrap();
        assert_ne!(a.sample(20), b.sample(20));
    }

    #[test]
    fn lhs_randomized_in_unit_hypercube() {
        let mut lhs = LatinHypercubeSampler::new(4, 7).unwrap();
        lhs.set_centered(false);
        let s = lhs.sample(50);
        for v in &s {
            assert!(*v >= 0.0 && *v < 1.0, "value {v} out of [0,1)");
        }
        assert_eq!(s.len(), 50 * 4);
    }

    #[test]
    fn lhs_randomized_one_per_stratum_invariant() {
        // Randomized LHS still places exactly one sample per stratum per
        // dimension; only the within-cell offset is random. Stratum index
        // is `floor(value * n)`.
        let n = 16;
        let d = 3;
        let mut lhs = LatinHypercubeSampler::new(d, 12345).unwrap();
        lhs.set_centered(false);
        let s = lhs.sample(n);
        for j in 0..d {
            let mut strata: Vec<usize> =
                (0..n).map(|i| (s[i * d + j] * n as f64) as usize).collect();
            strata.sort_unstable();
            assert_eq!(
                strata,
                (0..n).collect::<Vec<_>>(),
                "dim {j}: every stratum hit exactly once"
            );
        }
    }

    #[test]
    fn lhs_zero_samples_returns_empty_design() {
        let mut lhs = LatinHypercubeSampler::new(3, 0).unwrap();
        assert!(lhs.sample(0).is_empty());
    }

    #[test]
    fn poisson_disk_rejects_invalid_radius() {
        for radius in [0.0, -0.1, 1.0, f64::NAN] {
            assert!(
                matches!(
                    PoissonDiskSampler::new_2d(radius, 0),
                    Err(StatsError::InvalidArgument(_))
                ),
                "radius {radius} should fail"
            );
        }
    }

    #[test]
    fn poisson_disk_samples_are_separated_and_reproducible() {
        let mut a = PoissonDiskSampler::new_2d(0.08, 123).unwrap();
        let mut b = PoissonDiskSampler::new_2d(0.08, 123).unwrap();
        let sample = a.sample(24);
        assert_eq!(sample, b.sample(24));
        assert_eq!(sample.len(), 48);
        for value in &sample {
            assert!((0.0..1.0).contains(value), "value {value}");
        }
        for i in 0..24 {
            for j in i + 1..24 {
                let dx = sample[i * 2] - sample[j * 2];
                let dy = sample[i * 2 + 1] - sample[j * 2 + 1];
                let dist = dx.mul_add(dx, dy * dy).sqrt();
                assert!(dist >= 0.08 - 1e-15, "distance {dist} too small");
            }
        }
    }

    #[test]
    fn discrepancy_rejects_zero_dimension() {
        let err = centered_discrepancy(&[], 0).expect_err("d=0 must fail");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn discrepancy_rejects_misshaped_sample() {
        let err =
            centered_discrepancy(&[0.1, 0.2, 0.3], 2).expect_err("3 values for d=2 must fail");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn discrepancy_rejects_out_of_range_value() {
        let err = centered_discrepancy(&[0.5, 1.5], 2).expect_err("1.5 must be rejected");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn discrepancy_empty_sample_is_leading_constant() {
        // With n=0 the single and double sums vanish and only the (13/12)^d
        // leading term remains.
        let d = 3;
        let cd = centered_discrepancy(&[], d).unwrap();
        let expected = (13.0_f64 / 12.0).powi(d as i32);
        assert!(
            (cd - expected).abs() < 1e-15,
            "empty-sample CD² should equal (13/12)^d = {expected}, got {cd}"
        );
    }

    #[test]
    fn discrepancy_metamorphic_halton_beats_uniform_grid_2d() {
        // For 64 points in [0,1)^2, a Halton design has CD² much smaller
        // than a deterministic offset uniform grid (which has rows of points
        // exactly aligned, blowing up the double-sum term).
        let n = 64;
        let d = 2;
        let mut h = HaltonSampler::new(d).unwrap();
        let halton = h.sample(n);
        // Aligned grid: every point at (0.5, 0.5) — pathological, maximally
        // bad CD² because all single-sum products are identical and the
        // double-sum is dominated by the single-distance pair.
        let aligned = vec![0.5_f64; n * d];
        let cd_halton = centered_discrepancy(&halton, d).unwrap();
        let cd_aligned = centered_discrepancy(&aligned, d).unwrap();
        assert!(
            cd_halton < cd_aligned,
            "Halton CD²={cd_halton} should beat aligned-point CD²={cd_aligned}"
        );
        // Halton CD² for n=64, d=2 should be small (<0.01).
        assert!(
            cd_halton.abs() < 0.01,
            "Halton CD² {cd_halton} unexpectedly large"
        );
    }

    #[test]
    fn scale_rejects_mismatched_bounds() {
        let err = scale(&[0.5, 0.5], 2, &[0.0], &[1.0]).expect_err("bounds shape must match dim");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn scale_rejects_inverted_bounds() {
        let err =
            scale(&[0.5, 0.5], 2, &[2.0, 0.0], &[1.0, 1.0]).expect_err("u<l must be rejected");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn scale_rejects_coordinates_outside_unit_hypercube() {
        for sample in [
            &[-0.1_f64][..],
            &[1.5],
            &[f64::INFINITY],
            &[f64::NEG_INFINITY],
        ] {
            let err = scale(sample, 1, &[0.0], &[1.0]).expect_err("outside unit cube");
            assert!(matches!(err, StatsError::InvalidArgument(_)));
        }
    }

    #[test]
    fn scale_preserves_scipy_nan_propagation() {
        let out = scale(&[f64::NAN], 1, &[2.0], &[5.0]).expect("NaN propagates");
        assert!(out[0].is_nan());
    }

    #[test]
    fn scale_identity_with_unit_bounds() {
        let sample = [0.0_f64, 0.5, 0.99, 0.25, 0.75, 0.5];
        let out = scale(&sample, 2, &[0.0, 0.0], &[1.0, 1.0]).unwrap();
        for (a, b) in sample.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn scale_metamorphic_inverse_recovers_unit() {
        // forward: x ↦ l + (u-l)·x; inverse: y ↦ (y-l)/(u-l).
        let sample = [0.1_f64, 0.4, 0.7, 0.9, 0.5, 0.3];
        let l = [-2.0_f64, 5.0];
        let u = [3.0_f64, 8.5];
        let scaled = scale(&sample, 2, &l, &u).unwrap();
        for i in 0..3 {
            for k in 0..2 {
                let recovered = (scaled[i * 2 + k] - l[k]) / (u[k] - l[k]);
                let expected = sample[i * 2 + k];
                assert!(
                    (recovered - expected).abs() < 1e-12,
                    "roundtrip at ({i},{k}): got {recovered}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn scale_zero_width_bound_is_collapsed_dimension() {
        // u == l yields the constant value at that dimension.
        let sample = [0.1_f64, 0.5, 0.9];
        let out = scale(&sample, 1, &[2.0], &[2.0]).unwrap();
        for v in &out {
            assert_eq!(*v, 2.0);
        }
    }

    #[test]
    fn update_centered_disc_matches_scipy() {
        // scipy.stats.qmc.update_discrepancy is an additive correction to
        // the prior discrepancy, not a full recompute, so it does NOT equal
        // centered_discrepancy of the augmented design. Reference values
        // from scipy seeded with discrepancy(sample, method='CD').
        let sample2 = [0.1, 0.2, 0.5, 0.7, 0.9, 0.3, 0.3, 0.85, 0.65, 0.45];
        let init2 = centered_discrepancy(&sample2, 2).unwrap();
        let got2 = update_centered_discrepancy(&sample2, 2, init2, &[0.4, 0.6]).unwrap();
        assert!(
            (got2 - (-0.014_156_180_555_556_32)).abs() < 1e-12,
            "2D update = {got2}"
        );

        let sample3 = [
            0.1, 0.2, 0.3, 0.5, 0.7, 0.15, 0.9, 0.3, 0.8, 0.35, 0.6, 0.55,
        ];
        let init3 = centered_discrepancy(&sample3, 3).unwrap();
        let got3 = update_centered_discrepancy(&sample3, 3, init3, &[0.45, 0.25, 0.7]).unwrap();
        assert!(
            (got3 - 0.026_812_939_380_786_77).abs() < 1e-12,
            "3D update = {got3}"
        );
    }

    #[test]
    fn update_centered_disc_singleton_starts_from_empty() {
        // Adding a point to an empty sample must equal the CD² of the
        // singleton design.
        let new_point = [0.4_f64, 0.7];
        // CD² of empty sample is the (13/12)^d leading constant.
        let empty_cd = centered_discrepancy(&[], 2).unwrap();
        let after = update_centered_discrepancy(&[], 2, empty_cd, &new_point).unwrap();
        let direct = centered_discrepancy(&new_point, 2).unwrap();
        assert!((after - direct).abs() < 1e-12, "{after} != {direct}");
    }

    #[test]
    fn update_centered_disc_rejects_dimension_mismatch() {
        let err = update_centered_discrepancy(&[0.5, 0.5], 2, 0.0, &[0.5])
            .expect_err("new_point length must match dim");
        assert!(matches!(err, StatsError::InvalidArgument(_)));
    }

    #[test]
    fn l2_star_rejects_invalid_inputs() {
        assert!(matches!(
            l2_star_discrepancy(&[], 0),
            Err(StatsError::InvalidArgument(_))
        ));
        assert!(matches!(
            l2_star_discrepancy(&[0.5, 0.5, 0.5], 2),
            Err(StatsError::InvalidArgument(_))
        ));
        assert!(matches!(
            l2_star_discrepancy(&[1.5], 1),
            Err(StatsError::InvalidArgument(_))
        ));
    }

    #[test]
    fn l2_star_empty_sample_is_sqrt_one_third_to_d() {
        for d in 1..=4 {
            let sd = l2_star_discrepancy(&[], d).unwrap();
            let expected = (1.0_f64 / 3.0).powi(d as i32).sqrt();
            assert!((sd - expected).abs() < 1e-15, "d={d}: got {sd}");
        }
    }

    #[test]
    fn l2_star_metamorphic_single_corner_point() {
        // For n=1 at the corner (0,…,0), the single product Π (1 - 0²) = 1
        // and the double product Π (1 - max(0,0)) = 1, so
        // SD² = (1/3)^d − 2^(1−d) + 1 and scipy returns SD = √(SD²).
        for d in 1..=4 {
            let sample = vec![0.0_f64; d];
            let sd = l2_star_discrepancy(&sample, d).unwrap();
            let sd2 = (1.0_f64 / 3.0).powi(d as i32) - 2.0_f64.powi(1 - d as i32) + 1.0;
            let expected = sd2.sqrt();
            assert!(
                (sd - expected).abs() < 1e-13,
                "d={d}: SD={sd}, expected={expected}"
            );
        }
    }

    #[test]
    fn l2_star_metamorphic_halton_beats_corner_clustered_2d() {
        let n = 64;
        let d = 2;
        let mut h = HaltonSampler::new(d).unwrap();
        let halton = h.sample(n);
        // All-zeros design: every point at the lower-left corner. SD is
        // dominated by huge contributions (single sum=N, double sum=N²).
        let clustered = vec![0.0_f64; n * d];
        let sd_halton = l2_star_discrepancy(&halton, d).unwrap();
        let sd_cluster = l2_star_discrepancy(&clustered, d).unwrap();
        assert!(
            sd_halton < sd_cluster,
            "Halton SD={sd_halton} should beat corner-clustered SD={sd_cluster}"
        );
    }

    #[test]
    fn mixture_rejects_invalid_inputs() {
        assert!(matches!(
            mixture_discrepancy(&[], 0),
            Err(StatsError::InvalidArgument(_))
        ));
        assert!(matches!(
            mixture_discrepancy(&[0.1, 0.2, 0.3], 2),
            Err(StatsError::InvalidArgument(_))
        ));
        assert!(matches!(
            mixture_discrepancy(&[1.5], 1),
            Err(StatsError::InvalidArgument(_))
        ));
    }

    #[test]
    fn mixture_empty_sample_is_leading_constant() {
        let d = 3;
        let md = mixture_discrepancy(&[], d).unwrap();
        let expected = (19.0_f64 / 12.0).powi(d as i32);
        assert!(
            (md - expected).abs() < 1e-15,
            "empty MD² should equal (19/12)^d = {expected}, got {md}"
        );
    }

    #[test]
    fn mixture_metamorphic_single_point_at_center() {
        // For n=1 at the center of every dimension, |xi-1/2| = 0 and the
        // single/double products reduce to (5/3)^d / 1 and (15/8)^d / 1.
        // MD² = (19/12)^d - 2·(5/3)^d + (15/8)^d.
        for d in 1..=4 {
            let sample = vec![0.5_f64; d];
            let md = mixture_discrepancy(&sample, d).unwrap();
            let leading = (19.0_f64 / 12.0).powi(d as i32);
            let single = (5.0_f64 / 3.0).powi(d as i32);
            let double = (15.0_f64 / 8.0).powi(d as i32);
            let expected = leading - 2.0 * single + double;
            assert!(
                (md - expected).abs() < 1e-13,
                "d={d}: center-point MD²={md}, expected={expected}"
            );
        }
    }

    #[test]
    fn mixture_metamorphic_halton_beats_aligned_2d() {
        let n = 64;
        let d = 2;
        let mut h = HaltonSampler::new(d).unwrap();
        let halton = h.sample(n);
        let aligned = vec![0.5_f64; n * d];
        let md_halton = mixture_discrepancy(&halton, d).unwrap();
        let md_aligned = mixture_discrepancy(&aligned, d).unwrap();
        assert!(
            md_halton < md_aligned,
            "Halton MD²={md_halton} should beat aligned MD²={md_aligned}"
        );
    }

    #[test]
    fn wraparound_rejects_invalid_inputs() {
        assert!(matches!(
            wraparound_discrepancy(&[], 0),
            Err(StatsError::InvalidArgument(_))
        ));
        assert!(matches!(
            wraparound_discrepancy(&[0.1, 0.2, 0.3], 2),
            Err(StatsError::InvalidArgument(_))
        ));
        assert!(matches!(
            wraparound_discrepancy(&[0.5, -0.1], 2),
            Err(StatsError::InvalidArgument(_))
        ));
    }

    #[test]
    fn wraparound_metamorphic_single_point_closed_form() {
        // For n=1 in d dimensions the double sum is exactly Π (3/2) = (3/2)^d
        // regardless of the point's coordinates (because xi == xj makes
        // every distance zero). Thus WD² = (3/2)^d - (4/3)^d.
        for d in 1..=4 {
            let mut sample = vec![0.0_f64; d];
            for (k, coordinate) in sample.iter_mut().enumerate().take(d) {
                *coordinate = 0.1 + 0.07 * k as f64;
            }
            let wd = wraparound_discrepancy(&sample, d).unwrap();
            let expected = 1.5_f64.powi(d as i32) - (4.0_f64 / 3.0).powi(d as i32);
            assert!(
                (wd - expected).abs() < 1e-15,
                "d={d}: WD²={wd}, expected={expected}"
            );
        }
    }

    #[test]
    fn wraparound_metamorphic_halton_beats_aligned_points() {
        let n = 64;
        let d = 2;
        let mut h = HaltonSampler::new(d).unwrap();
        let halton = h.sample(n);
        let aligned = vec![0.5_f64; n * d];
        let wd_halton = wraparound_discrepancy(&halton, d).unwrap();
        let wd_aligned = wraparound_discrepancy(&aligned, d).unwrap();
        assert!(
            wd_halton < wd_aligned,
            "Halton WD²={wd_halton} should beat aligned WD²={wd_aligned}"
        );
    }

    #[test]
    fn wraparound_metamorphic_invariant_under_cyclic_shift() {
        // The wraparound L2 discrepancy is, by construction, invariant
        // under any per-dimension cyclic shift on the unit torus —
        // because the kernel K(d) = 3/2 − d(1−d) satisfies K(d) = K(1−d)
        // and shifting all points uniformly preserves pairwise
        // wraparound distances. Verify across shifts in {0.0, 0.13, 0.5,
        // 0.87} for a 4-point 3D Halton sample.
        let d = 3;
        let n = 4;
        let mut h = HaltonSampler::new(d).unwrap();
        let original = h.sample(n);
        let baseline = wraparound_discrepancy(&original, d).unwrap();
        for &shift in &[0.0_f64, 0.13, 0.5, 0.87] {
            let shifted: Vec<f64> = original.iter().map(|x| (x + shift).fract()).collect();
            let wd = wraparound_discrepancy(&shifted, d).unwrap();
            assert!(
                (wd - baseline).abs() < 1e-12,
                "shift={shift}: WD² changed from {baseline} to {wd} — kernel must be torus-invariant"
            );
        }
    }

    #[test]
    fn wraparound_empty_sample_is_negative_leading_constant() {
        // Empty sample: only the -(4/3)^d term survives.
        let d = 2;
        let wd = wraparound_discrepancy(&[], d).unwrap();
        let expected = -(4.0_f64 / 3.0).powi(d as i32);
        assert!((wd - expected).abs() < 1e-15);
    }

    /// Anchor `geometric_discrepancy` against scipy.stats.qmc.geometric_discrepancy
    /// on a deterministic 8-point 2-D Halton fixture (frankenscipy-b8s95).
    #[test]
    fn geometric_discrepancy_matches_scipy_halton_2d() {
        // Coordinates from scipy.stats.qmc.Halton(d=2, seed=42).random(8).
        let sample = vec![
            0.5513058732778853,
            0.15176798070427107,
            0.05130587327788534,
            0.8184346473709377,
            0.8013058732778853,
            0.4851013140376044,
            0.301_305_873_277_885_3,
            0.262879090904271,
            0.6763058732778853,
            0.9295457584820488,
            0.17630587327788534,
            0.5962124251487155,
            0.9263058732778853,
            0.04065687070427107,
            0.426_305_873_277_885_3,
            0.7073235362709377,
        ];
        let mindist =
            geometric_discrepancy(&sample, 2, GeometricDiscrepancyMethod::MinDist).unwrap();
        let mst = geometric_discrepancy(&sample, 2, GeometricDiscrepancyMethod::Mst).unwrap();
        // scipy reports 0.25496610764841404 / 0.3286278710953668 to ~16 digits.
        assert!(
            (mindist - 0.25496610764841404).abs() < 1e-10,
            "mindist {mindist} vs scipy 0.25497"
        );
        assert!(
            (mst - 0.3286278710953668).abs() < 1e-10,
            "mst {mst} vs scipy 0.32863"
        );
    }

    #[test]
    fn geometric_discrepancy_validates_inputs() {
        // Empty/single-point samples are rejected.
        let single = vec![0.5_f64, 0.5];
        assert!(
            geometric_discrepancy(&single, 2, GeometricDiscrepancyMethod::MinDist).is_err(),
            "single-point sample must be rejected"
        );
        // Out-of-unit-cube coords are rejected.
        let out = vec![0.5_f64, 1.5];
        assert!(
            geometric_discrepancy(&out, 1, GeometricDiscrepancyMethod::MinDist).is_err(),
            "out-of-unit-cube must be rejected"
        );
        // dimension=0 rejected.
        assert!(
            geometric_discrepancy(&[0.5_f64], 0, GeometricDiscrepancyMethod::MinDist).is_err(),
            "dimension=0 must be rejected"
        );
        // Length not multiple of dimension rejected.
        assert!(
            geometric_discrepancy(&[0.1_f64, 0.2, 0.3], 2, GeometricDiscrepancyMethod::MinDist)
                .is_err(),
            "ragged sample must be rejected"
        );
    }

    #[test]
    fn discrepancy_lhs_is_lower_than_random_offset_1d() {
        // 1D LHS: each cell has exactly one centered sample. CD² ought to
        // be smaller than a degenerate sample where all points cluster
        // (e.g. all at 0.25).
        let n = 32;
        let d = 1;
        let mut lhs = LatinHypercubeSampler::new(d, 0xc0ffee).unwrap();
        let sample = lhs.sample(n);
        let cd_lhs = centered_discrepancy(&sample, d).unwrap();
        let clustered = vec![0.25_f64; n];
        let cd_cluster = centered_discrepancy(&clustered, d).unwrap();
        assert!(
            cd_lhs < cd_cluster,
            "LHS CD²={cd_lhs} should beat clustered CD²={cd_cluster}"
        );
    }
}
