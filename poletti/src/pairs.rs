use std::ops::{Deref, DerefMut};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::util::{Token, Word};

#[derive(Default)]
pub(crate) struct Count {
    pub(crate) val: usize,
    pub(crate) idxs: FxHashSet<usize>,
}

#[derive(Eq, Ord, PartialEq, PartialOrd, Hash, Default, Clone, Copy)]
pub(crate) struct TokenPair(Token, Token);

impl TokenPair {
    fn new(word: &Word, i: usize) -> Self {
        Self(word.tokens[i], word.tokens[i + 1])
    }
}

type CountMap = FxHashMap<TokenPair, Count>;

struct PairCounts<Log = ()> {
    counts: CountMap,
    log: Log,
}

trait Log: Default {
    fn record(&mut self, token_pair: TokenPair);
    fn drain(&mut self) -> impl IntoIterator<Item = TokenPair> + use<Self>;
}

impl Log for () {
    fn drain(&mut self) -> impl IntoIterator<Item = TokenPair> + use<> {
        []
    }
    fn record(&mut self, token_pair: TokenPair) {}
}

impl Log for FxHashSet<TokenPair> {
    fn record(&mut self, token_pair: TokenPair) {
        self.insert(token_pair);
    }

    fn drain(&mut self) -> impl IntoIterator<Item = TokenPair> + use<> {
        std::mem::take(self)
    }
}

impl<L: Log> PairCounts<L> {
    fn new_logger<L2: Log>(self) -> PairCounts<L2> {
        PairCounts {
            counts: self.counts,
            log: L2::default(),
        }
    }
    fn from_words(words: &[Word]) -> Self {
        let counts = FxHashMap::default();
        let log = L::default();
        let mut s = Self { counts, log };
        for (idx, w) in words.iter().enumerate() {
            for i in 0..w.tokens.len() - 1 {
                s.record(idx, w, i, Mode::Incr);
            }
        }

        s
    }

    fn record(&mut self, word_idx: usize, word: &Word, i: usize, mode: Mode) {
        let ids = TokenPair::new(word, i);
        let Count { val, idxs } = self.counts.entry(ids).or_default();
        match mode {
            Mode::Incr => {
                *val += word.count;
                idxs.insert(word_idx);
            }
            Mode::Decr => {
                *val -= word.count;
                if *val == 0 {
                    self.counts.remove(&ids);
                }
            }
        }
        self.log.record(ids);
    }
}

pub(crate) struct Pairs<'a, L = ()> {
    words: &'a mut [Word],
    counts: PairCounts<L>,
}

impl<'a, L: Log> Deref for Pairs<'a, L> {
    type Target = CountMap;
    fn deref(&self) -> &Self::Target {
        &self.counts.counts
    }
}

impl<'a, L: Log> DerefMut for Pairs<'a, L> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.counts.counts
    }
}

enum Mode {
    Incr,
    Decr,
}

impl<'a> Pairs<'a, ()> {
    pub(crate) fn from_words(words: &'a mut [Word]) -> Self {
        let counts = PairCounts::from_words(&words);

        Self { counts, words }
    }

    pub(crate) fn to_logged(self) -> Pairs<'a, FxHashSet<TokenPair>> {
        Pairs {
            counts: self.counts.new_logger(),
            words: self.words,
        }
    }
}

impl<'a, L: Log> Pairs<'a, L> {
    pub(crate) fn merge(&mut self, merge: TokenPair, next_id: Token) {
        let Some(count) = self.counts.counts.get_mut(&merge) else {
            return;
        };

        let idxs = std::mem::take(&mut count.idxs);

        for w_idx in &idxs {
            let w = &mut self.words[*w_idx];
            let mut i = 0;

            while i < w.tokens.len() - 1 {
                let ids = TokenPair::new(w, i);
                if ids == merge {
                    let lo = (i).saturating_sub(1);
                    let hi = (i + 1).min(w.tokens.len() - 2);

                    for s in lo..=hi {
                        self.counts.record(*w_idx, w, s, Mode::Decr);
                    }

                    w.tokens[i] = next_id;
                    w.tokens.remove(i + 1);

                    for s in lo..hi {
                        self.counts.record(*w_idx, w, s, Mode::Incr);
                    }

                    continue;
                }

                i += 1;
            }
        }
    }

    pub(crate) fn drain_log<'b>(&'b mut self) -> impl IntoIterator<Item = TokenPair> + use<L> {
        self.counts.log.drain()
    }
}
