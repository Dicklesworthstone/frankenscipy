use fsci_io::mmread;
fn t(tag: &str, s: &str) {
    match mmread(s) {
        Ok(m) => println!("{tag}:{}x{}|{:?}", m.rows, m.cols, m.data),
        Err(e) => println!("{tag}:ERR:{e:?}"),
    }
}
fn main() {
    t(
        "sym3",
        "%%MatrixMarket matrix array real symmetric\n%\n3 3\n1\n2\n3\n4\n5\n6\n",
    );
    t(
        "skew3",
        "%%MatrixMarket matrix array real skew-symmetric\n%\n3 3\n-2\n-3\n-5\n",
    );
    t(
        "gen2",
        "%%MatrixMarket matrix array real general\n%\n2 3\n1\n4\n2\n5.5\n3\n-6\n",
    );
}
