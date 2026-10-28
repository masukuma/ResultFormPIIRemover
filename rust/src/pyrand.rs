//! Python's `random.Random`, bit for bit.
//!
//! The characters hidden in each ID or phone number are chosen at random from
//! a seed that is recorded in the manifest. `rfpii verify --rerun` repeats a
//! redaction from that seed, so the choice must come out the same whichever
//! implementation made the manifest: this is CPython's Mersenne Twister
//! (`init_by_array` seeding from an integer) and its `randbelow` / `sample`
//! algorithms, unchanged.

const N: usize = 624;
const M: usize = 397;

pub struct PyRandom {
    mt: [u32; N],
    index: usize,
}

impl PyRandom {
    /// `random.Random(seed)` for a non-negative integer seed.
    pub fn new(seed: u64) -> Self {
        // CPython splits abs(seed) into 32-bit words, least significant first;
        // zero is the single word [0].
        let mut key = vec![(seed & 0xffff_ffff) as u32];
        if seed >> 32 != 0 {
            key.push((seed >> 32) as u32);
        }
        let mut r = PyRandom { mt: [0; N], index: N };
        r.init_by_array(&key);
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            let prev = self.mt[i - 1];
            self.mt[i] = 1812433253u32.wrapping_mul(prev ^ (prev >> 30)).wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let (mut i, mut j) = (1usize, 0usize);
        let mut k = N.max(key.len());
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x8000_0000;
        self.index = N;
    }

    fn genrand_u32(&mut self) -> u32 {
        const MAG01: [u32; 2] = [0, 0x9908_b0df];
        if self.index >= N {
            for kk in 0..N {
                let y = (self.mt[kk] & 0x8000_0000) | (self.mt[(kk + 1) % N] & 0x7fff_ffff);
                self.mt[kk] = self.mt[(kk + M) % N] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `random()`: a float in [0, 1) with 53 random bits.
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// `getrandbits(k)` for k <= 64.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        assert!(k <= 64);
        if k == 0 {
            return 0;
        }
        if k <= 32 {
            return (self.genrand_u32() >> (32 - k)) as u64;
        }
        // least significant word first, the last one cut to the remaining bits
        let lo = self.genrand_u32() as u64;
        let hi = (self.genrand_u32() >> (64 - k)) as u64;
        lo | (hi << 32)
    }

    /// `_randbelow(n)`: uniform in [0, n), by rejection on bit_length(n) bits.
    pub fn randbelow(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let k = 64 - n.leading_zeros();
        let mut r = self.getrandbits(k);
        while r >= n {
            r = self.getrandbits(k);
        }
        r
    }

    /// `sample(range(n), k)`.
    pub fn sample_range(&mut self, n: usize, k: usize) -> Vec<usize> {
        assert!(k <= n, "sample larger than population");
        let mut setsize = 21usize;
        if k > 5 {
            // 4 ** ceil(log(k * 3, 4))
            let e = ((k as f64 * 3.0).ln() / 4f64.ln()).ceil() as u32;
            setsize += 4usize.pow(e);
        }
        let mut result = Vec::with_capacity(k);
        if n <= setsize {
            let mut pool: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = self.randbelow((n - i) as u64) as usize;
                result.push(pool[j]);
                pool[j] = pool[n - i - 1];
            }
        } else {
            let mut selected = std::collections::HashSet::new();
            for _ in 0..k {
                let mut j = self.randbelow(n as u64) as usize;
                while selected.contains(&j) {
                    j = self.randbelow(n as u64) as usize;
                }
                selected.insert(j);
                result.push(j);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_cpython() {
        // python3 -c "import random; r = random.Random(7); print(r.random(), r.getrandbits(32), r.sample(range(10), 3))"
        let mut r = PyRandom::new(7);
        assert_eq!(r.random(), 0.32383276483316237);
        assert_eq!(r.getrandbits(32), 647892279);
        assert_eq!(r.sample_range(10, 3), vec![6, 0, 1]);
        // a seed over 32 bits: two key words
        let mut r = PyRandom::new(0x1234_5678_9abc_def0);
        assert_eq!(r.random(), 0.849063363578306);
        let mut r = PyRandom::new(5);
        let s: Vec<Vec<usize>> = (0..3).map(|_| r.sample_range(12, 3)).collect();
        assert_eq!(s, vec![vec![9, 4, 5], vec![11, 10, 8], vec![0, 7, 3]]);
        assert_eq!(r.getrandbits(40), 58621899349);
    }
}
