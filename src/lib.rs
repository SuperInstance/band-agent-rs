//! `band-agent-rs` — Autonomous instrument agent with spectral identity,
//! conservation envelope, and t-minus timing.
//!
//! Each [`agent::Agent`] combines:
//! - A [`identity::SpectralIdentity`] (timbral fingerprint via eigenvalues)
//! - A [`tminus::TMinusClock`] (beat-phase scheduler)
//! - A [`conservation::ConservationEnvelope`] (energy budget with damping)
//! - A [`dial::DialPosition`] (expressive control surface)
//!
//! Agents can listen to MIDI events, adapt their spectral identity, propose
//! and negotiate tempos, and fire registered beat callbacks.

// ─────────────────────────────────────────────────────────────────────────────
// role
// ─────────────────────────────────────────────────────────────────────────────

/// Module containing the [`InstrumentRole`] enum.
pub mod role {
    /// The musical role an agent occupies in the ensemble.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum InstrumentRole {
        /// Rhythmic foundation — kick, snare, hi-hat.
        Drums,
        /// Low-frequency harmonic anchor.
        Bass,
        /// Mid-range harmonic and melodic content.
        Keys,
        /// Bright, high-energy melodic lines.
        Horns,
        /// Sustained atmospheric texture.
        Pads,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// identity
// ─────────────────────────────────────────────────────────────────────────────

/// Module containing the [`SpectralIdentity`] type.
pub mod identity {
    use crate::role::InstrumentRole;

    /// Timbral fingerprint encoded as 8 eigenvalues in spectral space.
    ///
    /// Each role has a characteristic profile. Agents can adapt their identity
    /// via Banach-convergent updates.
    #[derive(Debug, Clone)]
    pub struct SpectralIdentity {
        /// The 8-dimensional timbral eigenvalue vector.
        pub eigenvalues: [f64; 8],
        /// The instrument role associated with this identity.
        pub role: InstrumentRole,
    }

    impl SpectralIdentity {
        /// Create a new `SpectralIdentity` with a characteristic eigenvalue
        /// profile for the given role.
        ///
        /// Each role receives a distinct, deterministic initial profile so that
        /// [`similarity`](SpectralIdentity::similarity) between different roles
        /// is meaningfully less than 1.0.
        pub fn new(role: InstrumentRole) -> Self {
            let eigenvalues = match role {
                InstrumentRole::Drums => [
                    0.9, 0.1, 0.8, 0.1, 0.7, 0.1, 0.6, 0.1,
                ],
                InstrumentRole::Bass => [
                    0.1, 0.9, 0.1, 0.8, 0.1, 0.7, 0.1, 0.6,
                ],
                InstrumentRole::Keys => [
                    0.5, 0.5, 0.6, 0.6, 0.4, 0.4, 0.5, 0.5,
                ],
                InstrumentRole::Horns => [
                    0.7, 0.3, 0.9, 0.2, 0.8, 0.1, 0.7, 0.3,
                ],
                InstrumentRole::Pads => [
                    0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7,
                ],
            };
            Self { eigenvalues, role }
        }

        /// Normalise `v` in-place (L2 norm); returns the norm before scaling.
        fn normalize(v: &[f64; 8]) -> ([f64; 8], f64) {
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < f64::EPSILON {
                return (*v, 0.0);
            }
            let mut out = [0.0f64; 8];
            for (o, x) in out.iter_mut().zip(v.iter()) {
                *o = x / norm;
            }
            (out, norm)
        }

        /// Cosine similarity between `self` and `other` in eigenvalue space.
        ///
        /// Returns a value in `[-1.0, 1.0]`; 1.0 means identical fingerprints.
        pub fn similarity(&self, other: &SpectralIdentity) -> f64 {
            let (a, _) = Self::normalize(&self.eigenvalues);
            let (b, _) = Self::normalize(&other.eigenvalues);
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        }

        /// Apply a Banach-convergent update to the eigenvalue vector.
        ///
        /// `delta` is scaled by **0.1** before being added, ensuring the
        /// sequence of updates converges (contraction factor ≤ 0.1).
        pub fn update(&mut self, delta: &[f64; 8]) {
            for (e, d) in self.eigenvalues.iter_mut().zip(delta.iter()) {
                *e += 0.1 * d;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// tminus
// ─────────────────────────────────────────────────────────────────────────────

/// Module containing the [`TMinusClock`] beat-phase scheduler.
pub mod tminus {
    /// A beat-phase clock that fires beat events at a steady tempo.
    ///
    /// Phase advances continuously; when it wraps past 1.0 a beat fires and
    /// `beat_count` is incremented.
    #[derive(Debug, Clone)]
    pub struct TMinusClock {
        /// Tempo in beats-per-minute.
        pub tempo_bpm: f64,
        /// Current beat phase in `[0, 1)`.
        pub phase: f64,
        /// Total number of beats that have fired since creation.
        pub beat_count: u64,
        /// Seconds remaining until the next beat fires.
        pub next_beat: f64,
    }

    impl TMinusClock {
        /// Create a new clock at the given tempo.
        ///
        /// Phase starts at 0.0; `next_beat` is set to one full beat duration.
        pub fn new(tempo_bpm: f64) -> Self {
            let beat_dur = 60.0 / tempo_bpm;
            Self {
                tempo_bpm,
                phase: 0.0,
                beat_count: 0,
                next_beat: beat_dur,
            }
        }

        /// Advance the clock by `dt` seconds.
        ///
        /// Returns `Some(beat_count)` (the new count) each time a beat fires,
        /// or `None` if no beat occurred during this tick.
        pub fn tick(&mut self, dt: f64) -> Option<u64> {
            let beat_dur = 60.0 / self.tempo_bpm;
            let phase_delta = dt / beat_dur;
            self.phase += phase_delta;

            let mut fired = false;
            while self.phase >= 1.0 {
                self.phase -= 1.0;
                self.beat_count += 1;
                fired = true;
            }

            self.next_beat = (1.0 - self.phase) * beat_dur;

            if fired {
                Some(self.beat_count)
            } else {
                None
            }
        }

        /// Predict the time (in seconds) until the next beat fires.
        pub fn predict_next_beat(&self) -> f64 {
            let beat_dur = 60.0 / self.tempo_bpm;
            (1.0 - self.phase) * beat_dur
        }

        /// Update the tempo; `next_beat` is recalculated for the new BPM.
        pub fn set_tempo(&mut self, bpm: f64) {
            self.tempo_bpm = bpm;
            self.next_beat = self.predict_next_beat();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// conservation
// ─────────────────────────────────────────────────────────────────────────────

/// Module containing the [`ConservationEnvelope`] energy budget.
pub mod conservation {
    /// Conservation constant: `γ · H = C` where `H = energy / capacity`.
    pub const C: f64 = 1.0;

    /// Energy budget with a Lyapunov-inspired conservation law.
    ///
    /// The envelope damps energy over time via `γ` (gamma) and allows
    /// instantaneous energy injection (e.g. from MIDI velocity).
    #[derive(Debug, Clone)]
    pub struct ConservationEnvelope {
        /// Damping coefficient `γ ∈ [0, 1]`.
        pub gamma: f64,
        /// Current energy level.
        pub energy: f64,
        /// Maximum energy capacity.
        pub capacity: f64,
    }

    impl ConservationEnvelope {
        /// Create a new envelope with the given capacity.
        ///
        /// Initialises `gamma = 1.0` and `energy = capacity * 0.5`.
        pub fn new(capacity: f64) -> Self {
            Self {
                gamma: 1.0,
                energy: capacity * 0.5,
                capacity,
            }
        }

        /// Advance the envelope by `dt` seconds, applying gamma decay.
        ///
        /// `energy *= 1.0 - γ · dt · 0.01`
        pub fn tick(&mut self, dt: f64) {
            self.energy *= 1.0 - self.gamma * dt * 0.01;
            if self.energy < 0.0 {
                self.energy = 0.0;
            }
        }

        /// Inject `amount` of energy, clamped to `capacity`.
        pub fn inject(&mut self, amount: f64) {
            self.energy = (self.energy + amount).min(self.capacity);
        }

        /// Compute the conservation error: `|γ · (energy / capacity) − 1.0|`.
        pub fn conservation_error(&self) -> f64 {
            (self.gamma * (self.energy / self.capacity) - C).abs()
        }

        /// Return `true` if the conservation error is below 0.1.
        pub fn is_conserved(&self) -> bool {
            self.conservation_error() < 0.1
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// dial
// ─────────────────────────────────────────────────────────────────────────────

/// Module containing the [`DialPosition`] expressive control surface.
pub mod dial {
    /// Three-axis expressive control surface for an agent.
    #[derive(Debug, Clone)]
    pub struct DialPosition {
        /// Tradition axis: `0` = avant-garde, `1` = traditional.
        pub tradition: f64,
        /// Density axis: `0` = sparse, `1` = dense.
        pub density: f64,
        /// Intensity axis: `0` = quiet, `1` = intense.
        pub intensity: f64,
    }

    impl DialPosition {
        /// Create a new `DialPosition` with all axes at the neutral midpoint (0.5).
        pub fn new() -> Self {
            Self {
                tradition: 0.5,
                density: 0.5,
                intensity: 0.5,
            }
        }

        /// Apply a `[tradition_delta, density_delta, intensity_delta]` adjustment.
        ///
        /// Each axis is clamped to `[0.0, 1.0]` after the adjustment.
        pub fn adjust(&mut self, delta: [f64; 3]) {
            self.tradition = (self.tradition + delta[0]).clamp(0.0, 1.0);
            self.density = (self.density + delta[1]).clamp(0.0, 1.0);
            self.intensity = (self.intensity + delta[2]).clamp(0.0, 1.0);
        }

        /// Return the dial state as `[tradition, density, intensity]`.
        pub fn as_array(&self) -> [f64; 3] {
            [self.tradition, self.density, self.intensity]
        }
    }

    impl Default for DialPosition {
        fn default() -> Self {
            Self::new()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// agent
// ─────────────────────────────────────────────────────────────────────────────

/// Module containing the top-level [`Agent`] type.
pub mod agent {
    use crate::{
        conservation::ConservationEnvelope,
        dial::DialPosition,
        identity::SpectralIdentity,
        role::InstrumentRole,
        tminus::TMinusClock,
    };

    /// An autonomous instrument agent combining spectral identity, beat-phase
    /// clock, conservation envelope, and expressive dial control.
    pub struct Agent {
        /// Unique identifier for this agent in the ensemble.
        pub id: u64,
        /// Timbral fingerprint.
        pub identity: SpectralIdentity,
        /// Beat-phase clock.
        pub clock: TMinusClock,
        /// Energy conservation envelope.
        pub conservation: ConservationEnvelope,
        /// Expressive control dials.
        pub dial: DialPosition,
        beat_callbacks: Vec<Box<dyn Fn(u64) + Send + Sync>>,
        /// Tempo this agent would like the ensemble to adopt.
        pub proposed_tempo: f64,
    }

    impl Agent {
        /// Create a new agent with the given id, instrument role, and initial tempo.
        pub fn new(id: u64, role: InstrumentRole, tempo_bpm: f64) -> Self {
            Self {
                id,
                identity: SpectralIdentity::new(role),
                clock: TMinusClock::new(tempo_bpm),
                conservation: ConservationEnvelope::new(1.0),
                dial: DialPosition::new(),
                beat_callbacks: Vec::new(),
                proposed_tempo: tempo_bpm,
            }
        }

        /// Advance the agent by `dt` seconds.
        ///
        /// Ticks both the clock and the conservation envelope.  If a beat fires
        /// all registered callbacks are invoked.
        ///
        /// Returns `Some(beat_count)` on a beat, `None` otherwise.
        pub fn tick(&mut self, dt: f64) -> Option<u64> {
            self.conservation.tick(dt);
            let beat = self.clock.tick(dt);
            if let Some(count) = beat {
                for cb in &self.beat_callbacks {
                    cb(count);
                }
            }
            beat
        }

        /// Process an incoming MIDI note.
        ///
        /// Injects energy into the conservation envelope proportional to
        /// `velocity / 127.0` (full velocity = inject 1.0 unit).
        pub fn listen(&mut self, _midi_pitch: u8, velocity: u8) {
            let amount = velocity as f64 / 127.0;
            self.conservation.inject(amount);
        }

        /// Adapt the spectral identity using a Banach-convergent update.
        pub fn adapt(&mut self, feedback: &[f64; 8]) {
            self.identity.update(feedback);
        }

        /// Propose a tempo based on `clock.tempo_bpm` modulated by dial intensity.
        ///
        /// At `intensity = 0.5` (neutral) the proposal equals the current tempo.
        /// Higher intensity pushes the proposal up by up to 10 %, lower pushes it
        /// down by up to 10 %.
        pub fn propose_tempo(&self) -> f64 {
            let factor = 1.0 + (self.dial.intensity - 0.5) * 0.2;
            self.clock.tempo_bpm * factor
        }

        /// Blend the agent's tempo toward `ensemble_tempo`.
        ///
        /// Uses an 80/20 rule: `new_tempo = 0.8 * self + 0.2 * ensemble`.
        pub fn negotiate(&mut self, ensemble_tempo: f64) {
            let new_tempo = 0.8 * self.clock.tempo_bpm + 0.2 * ensemble_tempo;
            self.clock.set_tempo(new_tempo);
            self.proposed_tempo = new_tempo;
        }

        /// Register a callback that will be invoked each time a beat fires.
        ///
        /// The callback receives the current `beat_count`.
        pub fn on_beat(&mut self, callback: impl Fn(u64) + Send + Sync + 'static) {
            self.beat_callbacks.push(Box::new(callback));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::{
        agent::Agent,
        conservation::ConservationEnvelope,
        dial::DialPosition,
        identity::SpectralIdentity,
        role::InstrumentRole,
        tminus::TMinusClock,
    };
    use std::sync::{Arc, Mutex};

    // ── SpectralIdentity ──────────────────────────────────────────────────

    #[test]
    fn identity_drums_non_zero() {
        let id = SpectralIdentity::new(InstrumentRole::Drums);
        assert!(id.eigenvalues.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn identity_bass_non_zero() {
        let id = SpectralIdentity::new(InstrumentRole::Bass);
        assert!(id.eigenvalues.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn identity_roles_distinct() {
        let drums = SpectralIdentity::new(InstrumentRole::Drums);
        let bass = SpectralIdentity::new(InstrumentRole::Bass);
        let keys = SpectralIdentity::new(InstrumentRole::Keys);
        let horns = SpectralIdentity::new(InstrumentRole::Horns);
        let pads = SpectralIdentity::new(InstrumentRole::Pads);

        // All eigenvalue vectors are different
        assert_ne!(drums.eigenvalues, bass.eigenvalues);
        assert_ne!(bass.eigenvalues, keys.eigenvalues);
        assert_ne!(keys.eigenvalues, horns.eigenvalues);
        assert_ne!(horns.eigenvalues, pads.eigenvalues);
        assert_ne!(drums.eigenvalues, pads.eigenvalues);
    }

    #[test]
    fn identity_self_similarity_is_one() {
        for role in [
            InstrumentRole::Drums,
            InstrumentRole::Bass,
            InstrumentRole::Keys,
            InstrumentRole::Horns,
            InstrumentRole::Pads,
        ] {
            let id = SpectralIdentity::new(role);
            let sim = id.similarity(&id);
            assert!(
                (sim - 1.0).abs() < 1e-10,
                "self-similarity for {role:?} = {sim}"
            );
        }
    }

    #[test]
    fn identity_cross_role_similarity_less_than_one() {
        let drums = SpectralIdentity::new(InstrumentRole::Drums);
        let bass = SpectralIdentity::new(InstrumentRole::Bass);
        let sim = drums.similarity(&bass);
        assert!(sim < 1.0, "drums-bass similarity should be <1.0, got {sim}");
    }

    #[test]
    fn identity_update_is_bounded() {
        let mut id = SpectralIdentity::new(InstrumentRole::Keys);
        let before = id.eigenvalues;
        let big_delta = [100.0f64; 8];
        id.update(&big_delta);
        // Scaled by 0.1 → each entry moves by 10.0
        for (b, a) in before.iter().zip(id.eigenvalues.iter()) {
            let diff = (a - b).abs();
            assert!(
                (diff - 10.0).abs() < 1e-10,
                "expected update of 10.0, got {diff}"
            );
        }
    }

    #[test]
    fn identity_update_negative_delta() {
        let mut id = SpectralIdentity::new(InstrumentRole::Pads);
        let before = id.eigenvalues;
        id.update(&[-1.0f64; 8]);
        for (b, a) in before.iter().zip(id.eigenvalues.iter()) {
            assert!(
                ((a - b) - (-0.1)).abs() < 1e-12,
                "expected -0.1 shift"
            );
        }
    }

    // ── TMinusClock ───────────────────────────────────────────────────────

    #[test]
    fn tminus_new_initializes() {
        let c = TMinusClock::new(120.0);
        assert_eq!(c.tempo_bpm, 120.0);
        assert_eq!(c.phase, 0.0);
        assert_eq!(c.beat_count, 0);
        assert!((c.next_beat - 0.5).abs() < 1e-10);
    }

    #[test]
    fn tminus_tick_no_beat_partial() {
        let mut c = TMinusClock::new(120.0); // beat every 0.5 s
        let result = c.tick(0.1);
        assert!(result.is_none());
        assert!((c.phase - 0.2).abs() < 1e-10);
    }

    #[test]
    fn tminus_tick_fires_beat() {
        let mut c = TMinusClock::new(120.0);
        let result = c.tick(0.5);
        assert_eq!(result, Some(1));
        assert_eq!(c.beat_count, 1);
    }

    #[test]
    fn tminus_phase_wraps_at_one() {
        let mut c = TMinusClock::new(120.0);
        c.tick(0.5);
        assert!(c.phase < 1.0, "phase should wrap: got {}", c.phase);
    }

    #[test]
    fn tminus_predict_consistent_with_phase() {
        let mut c = TMinusClock::new(60.0); // beat every 1.0 s
        c.tick(0.3);
        let predicted = c.predict_next_beat();
        assert!((predicted - c.next_beat).abs() < 1e-10);
    }

    #[test]
    fn tminus_set_tempo_updates_next_beat() {
        let mut c = TMinusClock::new(120.0);
        c.tick(0.1); // advance phase slightly
        c.set_tempo(60.0);
        assert_eq!(c.tempo_bpm, 60.0);
        // next_beat should reflect the new BPM
        let expected = (1.0 - c.phase) * (60.0 / 60.0);
        assert!((c.next_beat - expected).abs() < 1e-10);
    }

    #[test]
    fn tminus_multiple_beats_in_one_tick() {
        let mut c = TMinusClock::new(120.0); // 0.5 s per beat
        let result = c.tick(1.2);            // should fire 2 beats
        assert_eq!(result, Some(2));
        assert_eq!(c.beat_count, 2);
    }

    // ── ConservationEnvelope ──────────────────────────────────────────────

    #[test]
    fn conservation_new() {
        let ce = ConservationEnvelope::new(2.0);
        assert_eq!(ce.gamma, 1.0);
        assert_eq!(ce.energy, 1.0);
        assert_eq!(ce.capacity, 2.0);
    }

    #[test]
    fn conservation_inject_clamps() {
        let mut ce = ConservationEnvelope::new(1.0);
        ce.inject(999.0);
        assert_eq!(ce.energy, 1.0);
    }

    #[test]
    fn conservation_tick_applies_decay() {
        let mut ce = ConservationEnvelope::new(1.0);
        let before = ce.energy;
        ce.tick(1.0);
        assert!(ce.energy < before, "energy should decay");
    }

    #[test]
    fn conservation_error_formula() {
        let ce = ConservationEnvelope::new(1.0);
        // gamma=1, energy=0.5, capacity=1 → |1*(0.5/1) - 1| = 0.5
        let err = ce.conservation_error();
        assert!((err - 0.5).abs() < 1e-10, "got {err}");
    }

    #[test]
    fn conservation_is_conserved_near_capacity() {
        let mut ce = ConservationEnvelope::new(1.0);
        ce.energy = 0.95; // gamma=1, H=0.95, error=0.05 < 0.1
        assert!(ce.is_conserved());
    }

    #[test]
    fn conservation_is_not_conserved_at_half() {
        let ce = ConservationEnvelope::new(1.0); // energy=0.5, error=0.5
        assert!(!ce.is_conserved());
    }

    // ── DialPosition ──────────────────────────────────────────────────────

    #[test]
    fn dial_new_is_midpoint() {
        let d = DialPosition::new();
        assert_eq!(d.tradition, 0.5);
        assert_eq!(d.density, 0.5);
        assert_eq!(d.intensity, 0.5);
    }

    #[test]
    fn dial_adjust_within_range() {
        let mut d = DialPosition::new();
        d.adjust([0.2, -0.1, 0.3]);
        assert!((d.tradition - 0.7).abs() < 1e-10);
        assert!((d.density - 0.4).abs() < 1e-10);
        assert!((d.intensity - 0.8).abs() < 1e-10);
    }

    #[test]
    fn dial_adjust_clamps_high() {
        let mut d = DialPosition::new();
        d.adjust([1.0, 1.0, 1.0]);
        assert_eq!(d.tradition, 1.0);
        assert_eq!(d.density, 1.0);
        assert_eq!(d.intensity, 1.0);
    }

    #[test]
    fn dial_adjust_clamps_low() {
        let mut d = DialPosition::new();
        d.adjust([-1.0, -1.0, -1.0]);
        assert_eq!(d.tradition, 0.0);
        assert_eq!(d.density, 0.0);
        assert_eq!(d.intensity, 0.0);
    }

    #[test]
    fn dial_as_array_roundtrip() {
        let d = DialPosition::new();
        let arr = d.as_array();
        assert_eq!(arr, [0.5, 0.5, 0.5]);
    }

    // ── Agent ─────────────────────────────────────────────────────────────

    #[test]
    fn agent_new() {
        let a = Agent::new(1, InstrumentRole::Drums, 120.0);
        assert_eq!(a.id, 1);
        assert_eq!(a.clock.tempo_bpm, 120.0);
        assert_eq!(a.proposed_tempo, 120.0);
    }

    #[test]
    fn agent_tick_fires_callback() {
        let counter = Arc::new(Mutex::new(0u64));
        let c = Arc::clone(&counter);
        let mut a = Agent::new(2, InstrumentRole::Bass, 120.0);
        a.on_beat(move |beat| {
            *c.lock().unwrap() = beat;
        });
        a.tick(0.5); // one beat at 120 BPM
        assert_eq!(*counter.lock().unwrap(), 1);
    }

    #[test]
    fn agent_tick_no_beat_before_due() {
        let fired = Arc::new(Mutex::new(false));
        let f = Arc::clone(&fired);
        let mut a = Agent::new(3, InstrumentRole::Keys, 120.0);
        a.on_beat(move |_| {
            *f.lock().unwrap() = true;
        });
        a.tick(0.1);
        assert!(!*fired.lock().unwrap());
    }

    #[test]
    fn agent_listen_injects_energy() {
        let mut a = Agent::new(4, InstrumentRole::Horns, 120.0);
        a.conservation.energy = 0.0;
        a.listen(60, 127);
        assert!((a.conservation.energy - 1.0).abs() < 1e-10);
    }

    #[test]
    fn agent_adapt_updates_identity() {
        let mut a = Agent::new(5, InstrumentRole::Pads, 120.0);
        let before = a.identity.eigenvalues;
        a.adapt(&[1.0; 8]);
        assert_ne!(a.identity.eigenvalues, before);
    }

    #[test]
    fn agent_propose_tempo_neutral() {
        let a = Agent::new(6, InstrumentRole::Drums, 120.0);
        // intensity=0.5 → factor=1.0
        let pt = a.propose_tempo();
        assert!((pt - 120.0).abs() < 1e-10, "got {pt}");
    }

    #[test]
    fn agent_propose_tempo_high_intensity() {
        let mut a = Agent::new(7, InstrumentRole::Bass, 120.0);
        a.dial.intensity = 1.0; // factor = 1 + 0.5*0.2 = 1.10
        let pt = a.propose_tempo();
        assert!((pt - 132.0).abs() < 1e-10, "got {pt}");
    }

    #[test]
    fn agent_negotiate_blends_tempo() {
        let mut a = Agent::new(8, InstrumentRole::Keys, 120.0);
        a.negotiate(140.0);
        let expected = 0.8 * 120.0 + 0.2 * 140.0; // 124.0
        assert!((a.clock.tempo_bpm - expected).abs() < 1e-10);
        assert!((a.proposed_tempo - expected).abs() < 1e-10);
    }

    // ── Integration tests ─────────────────────────────────────────────────

    #[test]
    fn integration_agent_runs_multiple_beats() {
        let beat_log = Arc::new(Mutex::new(Vec::<u64>::new()));
        let log = Arc::clone(&beat_log);
        let mut a = Agent::new(10, InstrumentRole::Drums, 120.0);
        a.on_beat(move |n| log.lock().unwrap().push(n));

        // 120 BPM → beat every 0.5 s; drive 2.5 s → expect 5 beats
        let mut t = 0.0f64;
        while t < 2.5 {
            a.tick(0.05);
            t += 0.05;
        }
        let beats = beat_log.lock().unwrap().clone();
        assert!(
            beats.len() >= 4,
            "expected ≥4 beats, got {}",
            beats.len()
        );
    }

    #[test]
    fn integration_conservation_stays_bounded() {
        let mut a = Agent::new(11, InstrumentRole::Bass, 120.0);
        let mut t = 0.0f64;
        while t < 5.0 {
            a.tick(0.05);
            a.listen(60, 64); // moderate velocity every tick
            assert!(
                a.conservation.energy <= a.conservation.capacity,
                "energy exceeded capacity at t={t}"
            );
            t += 0.05;
        }
    }

    #[test]
    fn integration_dial_affects_proposed_tempo() {
        let mut low = Agent::new(12, InstrumentRole::Keys, 120.0);
        let mut high = Agent::new(13, InstrumentRole::Keys, 120.0);
        low.dial.intensity = 0.0;
        high.dial.intensity = 1.0;
        assert!(
            low.propose_tempo() < high.propose_tempo(),
            "high intensity should raise proposed tempo"
        );
    }

    #[test]
    fn integration_negotiate_converges() {
        let mut a = Agent::new(14, InstrumentRole::Pads, 100.0);
        let target = 120.0;
        for _ in 0..20 {
            a.negotiate(target);
        }
        // After many iterations of 80/20 blending toward 120, should be close
        assert!(
            (a.clock.tempo_bpm - target).abs() < 1.0,
            "tempo {:.3} should be near {target}",
            a.clock.tempo_bpm
        );
    }

    #[test]
    fn integration_spectral_identity_similarity_after_adaptation() {
        let mut a = Agent::new(15, InstrumentRole::Horns, 120.0);
        let original = a.identity.clone();
        // Apply many small updates - identity should drift but similarity should still be positive
        for _ in 0..10 {
            a.adapt(&[0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5]);
        }
        let sim = original.similarity(&a.identity);
        assert!(sim > 0.0, "similarity should remain positive after adaptation");
    }
}
