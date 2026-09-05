//! Probe: fsci contingency_table labels on the diff-test sparse fixture.
fn main() {
    let x: Vec<usize> = vec![0, 0, 0, 2, 2, 5];
    let y: Vec<usize> = vec![1, 1, 3, 1, 3, 1];
    let (t, rows, cols) = fsci_stats::contingency_table(&x, &y);
    println!("PROBE rows={rows:?} cols={cols:?} table={t:?}");
    println!("EXPECT rows=[0, 2, 5] cols=[1, 3]");
}
