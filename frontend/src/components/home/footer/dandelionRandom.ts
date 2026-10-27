// Deterministic PRNG (splitmix32 — same algorithm already used by
// InterviewAvatar/Orb.tsx elsewhere in this codebase) so every value the
// footer scene needs — flower jitter, seed start position, drift timing —
// is a pure function of a fixed integer seed. Server and client compute the
// exact same numbers, so there is no SSR/hydration mismatch and no
// per-render Math.random() call.
export function splitmix32(seed: number) {
  let a = seed | 0;
  return function next() {
    a = (a + 0x9e3779b9) | 0;
    let t = a ^ (a >>> 16);
    t = Math.imul(t, 0x21f0aaad);
    t = t ^ (t >>> 15);
    t = Math.imul(t, 0x735a2d97);
    t = t ^ (t >>> 15);
    return (t >>> 0) / 4294967296;
  };
}

// [min, max) helper built on top of a splitmix32 generator function.
export function randRange(rand: () => number, min: number, max: number) {
  return min + rand() * (max - min);
}
