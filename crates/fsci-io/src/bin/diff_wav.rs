use fsci_io::wav_read;
fn main() {
    let bytes: &[u8] = &[
        82, 73, 70, 70, 46, 0, 0, 0, 87, 65, 86, 69, 102, 109, 116, 32, 16, 0, 0, 0, 1, 0, 1, 0,
        64, 31, 0, 0, 128, 62, 0, 0, 2, 0, 16, 0, 100, 97, 116, 97, 10, 0, 0, 0, 0, 0, 0, 64, 0,
        192, 255, 127, 0, 128,
    ];
    let w = wav_read(bytes).unwrap();
    println!(
        "rate={} ch={} bps={} data={:?}",
        w.sample_rate, w.channels, w.bits_per_sample, w.data
    );
}
