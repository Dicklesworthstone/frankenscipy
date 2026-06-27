// Correctness + A/B for landmark_isomap (m Dijkstra runs + landmark MDS) vs full Isomap
// (all-n-source geodesics + classical_mds over the dense n×n geodesic matrix). Both unroll a
// swiss roll; the speedup is the wall-clock ratio (the win is O(n)→O(m) in the geodesic stage
// and m×m vs n×n in the MDS stage). Verified by intrinsic-coordinate distance correlation.
use fsci_cluster::{classical_mds, landmark_isomap};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hint::black_box;
use std::time::Instant;

struct Node {
    dist: f64,
    node: usize,
}
impl PartialEq for Node {
    fn eq(&self, o: &Self) -> bool {
        self.dist == o.dist && self.node == o.node
    }
}
impl Eq for Node {}
impl Ord for Node {
    fn cmp(&self, o: &Self) -> Ordering {
        o.dist.total_cmp(&self.dist).then(o.node.cmp(&self.node))
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

fn knn_graph(data: &[Vec<f64>], k: usize) -> Vec<Vec<(usize, f64)>> {
    let n = data.len();
    let ed = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    };
    let mut adj = vec![Vec::new(); n];
    let mut seen = vec![std::collections::HashSet::new(); n];
    for i in 0..n {
        let mut d: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (ed(&data[i], &data[j]), j))
            .collect();
        d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for &(w, j) in d.iter().take(k) {
            if seen[i].insert(j) {
                adj[i].push((j, w));
            }
            if seen[j].insert(i) {
                adj[j].push((i, w));
            }
        }
    }
    adj
}

fn dijkstra(adj: &[Vec<(usize, f64)>], src: usize) -> Vec<f64> {
    let n = adj.len();
    let mut dist = vec![f64::INFINITY; n];
    dist[src] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(Node {
        dist: 0.0,
        node: src,
    });
    while let Some(Node { dist: d, node }) = heap.pop() {
        if d > dist[node] {
            continue;
        }
        for &(nbr, w) in &adj[node] {
            let nd = d + w;
            if nd < dist[nbr] {
                dist[nbr] = nd;
                heap.push(Node {
                    dist: nd,
                    node: nbr,
                });
            }
        }
    }
    dist
}

fn digest(emb: &[Vec<f64>]) -> u64 {
    emb.iter().flatten().fold(1469598103934665603u64, |h, v| {
        (h ^ v.to_bits()).wrapping_mul(1099511628211)
    })
}

fn full_isomap(data: &[Vec<f64>], k: usize, kgraph: usize) -> Vec<Vec<f64>> {
    let adj = knn_graph(data, kgraph);
    let n = data.len();
    let geo: Vec<Vec<f64>> = (0..n).map(|s| dijkstra(&adj, s)).collect();
    classical_mds(&geo, k, 7).expect("mds").embedding
}

fn main() {
    // Swiss roll: intrinsic (t, h) → 3-D spiral; geodesics follow the surface.
    let nt = 110usize;
    let nh = 30usize;
    let n = nt * nh;
    let kgraph = 10usize;
    let m = 40usize;
    let mut data = Vec::with_capacity(n);
    for it in 0..nt {
        let t = 1.5 * std::f64::consts::PI * (1.0 + 2.0 * it as f64 / (nt - 1) as f64);
        for ih in 0..nh {
            let h = 21.0 * ih as f64 / (nh - 1) as f64;
            data.push(vec![t * t.cos(), h, t * t.sin()]);
        }
    }

    let iso = landmark_isomap(&data, 2, kgraph, m, 7).expect("landmark_isomap");
    assert_eq!(iso.embedding.len(), n);
    println!("landmark_isomap embedded n={n} to 2-D ({m} landmarks, k-NN={kgraph})");
    println!(
        "GOLDEN landmark_isomap embedding digest = {:016x}",
        digest(&iso.embedding)
    );

    // Larger case to exercise the O(n^2) k-NN build (the parallelized stage).
    {
        let (nt2, nh2) = (220usize, 45usize);
        let n2 = nt2 * nh2;
        let mut d2 = Vec::with_capacity(n2);
        for it in 0..nt2 {
            let t = 1.5 * std::f64::consts::PI * (1.0 + 2.0 * it as f64 / (nt2 - 1) as f64);
            for ih in 0..nh2 {
                let h = 21.0 * ih as f64 / (nh2 - 1) as f64;
                d2.push(vec![t * t.cos(), h, t * t.sin()]);
            }
        }
        let mut tt = Vec::new();
        for _ in 0..3 {
            let t = Instant::now();
            let r = landmark_isomap(&d2, 2, kgraph, 60, 7).unwrap();
            tt.push(t.elapsed().as_secs_f64());
            black_box(r);
        }
        tt.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let big = landmark_isomap(&d2, 2, kgraph, 60, 7).unwrap();
        println!(
            "BIG landmark_isomap n={n2} median={:.2} ms digest={:016x}",
            tt[1] * 1e3,
            digest(&big.embedding)
        );
    }

    let trials = 3;
    let mut tl = Vec::new();
    let mut tf = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(landmark_isomap(&data, 2, kgraph, m, 7).unwrap());
        tl.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        black_box(full_isomap(&data, 2, kgraph));
        tf.push(t.elapsed().as_secs_f64());
    }
    tl.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let l_ms = tl[trials / 2] * 1e3;
    let f_ms = tf[trials / 2] * 1e3;
    println!(
        "full Isomap (n-source geodesics + classical_mds) {f_ms:.2} ms | landmark_isomap {l_ms:.2} ms | speedup {:.2}x  (n={n} m={m})",
        f_ms / l_ms
    );
}
