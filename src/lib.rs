#![forbid(unsafe_code)]

pub const SPECTRAL_BINS: usize = 8;

#[derive(Debug, Clone, PartialEq)]
pub struct SpectralIdentity {
    pub amplitudes: [f64; SPECTRAL_BINS],
}

impl SpectralIdentity {
    pub fn new(amplitudes: [f64; SPECTRAL_BINS]) -> Self {
        SpectralIdentity { amplitudes }
    }

    pub fn from_dominant_bin(bin: usize) -> Self {
        let mut a = [0.0f64; SPECTRAL_BINS];
        if bin < SPECTRAL_BINS { a[bin] = 1.0; }
        SpectralIdentity { amplitudes: a }
    }

    pub fn distance(&self, other: &Self) -> f64 {
        self.amplitudes.iter().zip(other.amplitudes.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt()
    }

    /// Amplitude-weighted mean bin, normalized to [0,1].
    pub fn centroid(&self) -> f64 {
        let total: f64 = self.amplitudes.iter().sum();
        if total == 0.0 { return 0.0; }
        let weighted: f64 = self.amplitudes.iter().enumerate().map(|(i, a)| i as f64 * a).sum();
        weighted / total / (SPECTRAL_BINS as f64 - 1.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TMinusClock {
    pub tick: i64,
    pub ticks_per_beat: u32,
}

impl TMinusClock {
    pub fn new(start_tick: i64, ticks_per_beat: u32) -> Self {
        TMinusClock { tick: start_tick, ticks_per_beat: ticks_per_beat.max(1) }
    }

    pub fn advance(&mut self) { self.tick += 1; }

    pub fn beat_phase(&self) -> f64 {
        let tpb = self.ticks_per_beat as i64;
        self.tick.rem_euclid(tpb) as f64 / tpb as f64
    }

    pub fn is_downbeat(&self) -> bool { self.tick % self.ticks_per_beat as i64 == 0 }
    pub fn is_before_zero(&self) -> bool { self.tick < 0 }
}

#[derive(Debug, Clone, Default)]
pub struct ConservationTracker {
    pub energy: f64,
    pub momentum: f64,
}

impl ConservationTracker {
    pub fn new() -> Self { ConservationTracker { energy: 0.0, momentum: 0.0 } }

    pub fn apply(&mut self, amplitude: f64, sign: f64) {
        self.energy   += amplitude * amplitude;
        self.momentum += sign * amplitude;
    }

    pub fn is_consistent(&self) -> bool { self.energy >= 0.0 }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AgentOutput {
    Note { pitch: u8, velocity: u8 },
    IdentityBroadcast { centroid: f64 },
    Idle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentState { Idle, Playing, Listening, Adapting }

pub struct Agent {
    pub id: u32,
    pub identity: SpectralIdentity,
    pub clock: TMinusClock,
    pub conservation: ConservationTracker,
    pub state: AgentState,
}

impl Agent {
    pub fn new(id: u32, dominant_bin: usize, start_tick: i64, ticks_per_beat: u32) -> Self {
        Agent {
            id,
            identity: SpectralIdentity::from_dominant_bin(dominant_bin),
            clock: TMinusClock::new(start_tick, ticks_per_beat),
            conservation: ConservationTracker::new(),
            state: AgentState::Idle,
        }
    }

    pub fn tick(&mut self) -> AgentOutput {
        self.clock.advance();
        match self.state {
            AgentState::Playing => {
                let pitch = (self.identity.centroid() * 127.0) as u8;
                self.conservation.apply(80.0 / 127.0, 1.0);
                AgentOutput::Note { pitch, velocity: 80 }
            }
            AgentState::Listening => AgentOutput::IdentityBroadcast { centroid: self.identity.centroid() },
            _ => AgentOutput::Idle,
        }
    }

    pub fn set_state(&mut self, state: AgentState) { self.state = state; }

    pub fn adapt_toward(&mut self, target: &SpectralIdentity, alpha: f64) {
        let alpha = alpha.clamp(0.0, 1.0);
        for (a, t) in self.identity.amplitudes.iter_mut().zip(target.amplitudes.iter()) {
            *a = *a * (1.0 - alpha) + t * alpha;
        }
    }

    pub fn spectral_distance(&self, other: &Agent) -> f64 {
        self.identity.distance(&other.identity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn spectral_distance_self_zero() {
        let id = SpectralIdentity::from_dominant_bin(3);
        assert!(id.distance(&id) < 1e-9);
    }
    #[test] fn spectral_distance_orthogonal() {
        let a = SpectralIdentity::from_dominant_bin(0);
        let b = SpectralIdentity::from_dominant_bin(1);
        assert!((a.distance(&b) - 2.0f64.sqrt()).abs() < 1e-9);
    }
    #[test] fn spectral_centroid_zero_amplitudes() {
        assert_eq!(SpectralIdentity::new([0.0; SPECTRAL_BINS]).centroid(), 0.0);
    }
    #[test] fn spectral_centroid_last_bin() {
        assert!((SpectralIdentity::from_dominant_bin(7).centroid() - 1.0).abs() < 1e-9);
    }
    #[test] fn spectral_centroid_first_bin() {
        assert!(SpectralIdentity::from_dominant_bin(0).centroid().abs() < 1e-9);
    }
    #[test] fn spectral_identity_eq() {
        let a = SpectralIdentity::new([0.5; SPECTRAL_BINS]);
        let b = SpectralIdentity::new([0.5; SPECTRAL_BINS]);
        assert_eq!(a, b);
    }
    #[test] fn clock_advance() {
        let mut clk = TMinusClock::new(-4, 4);
        clk.advance();
        assert_eq!(clk.tick, -3);
    }
    #[test] fn clock_downbeat_at_zero() { assert!(TMinusClock::new(0, 4).is_downbeat()); }
    #[test] fn clock_not_downbeat_off_beat() { assert!(!TMinusClock::new(1, 4).is_downbeat()); }
    #[test] fn clock_before_zero() { assert!(TMinusClock::new(-1, 4).is_before_zero()); }
    #[test] fn clock_not_before_zero() { assert!(!TMinusClock::new(0, 4).is_before_zero()); }
    #[test] fn clock_beat_phase_zero() { assert!(TMinusClock::new(0, 4).beat_phase().abs() < 1e-9); }
    #[test] fn clock_beat_phase_half() { assert!((TMinusClock::new(2, 4).beat_phase() - 0.5).abs() < 1e-9); }
    #[test] fn conservation_initial_zero() {
        let c = ConservationTracker::new();
        assert_eq!(c.energy, 0.0);
        assert_eq!(c.momentum, 0.0);
    }
    #[test] fn conservation_apply() {
        let mut c = ConservationTracker::new();
        c.apply(0.5, 1.0);
        assert!((c.energy - 0.25).abs() < 1e-9);
        assert!((c.momentum - 0.5).abs() < 1e-9);
    }
    #[test] fn conservation_is_consistent() { assert!(ConservationTracker::new().is_consistent()); }
    #[test] fn agent_new_idle() {
        let a = Agent::new(1, 3, -8, 4);
        assert_eq!(a.state, AgentState::Idle);
        assert_eq!(a.id, 1);
    }
    #[test] fn agent_tick_idle() {
        let mut a = Agent::new(1, 3, -8, 4);
        assert_eq!(a.tick(), AgentOutput::Idle);
    }
    #[test] fn agent_tick_playing_emits_note() {
        let mut a = Agent::new(1, 7, -8, 4);
        a.set_state(AgentState::Playing);
        assert!(matches!(a.tick(), AgentOutput::Note { .. }));
    }
    #[test] fn agent_tick_listening_broadcasts() {
        let mut a = Agent::new(2, 4, 0, 4);
        a.set_state(AgentState::Listening);
        assert!(matches!(a.tick(), AgentOutput::IdentityBroadcast { .. }));
    }
    #[test] fn agent_adapt_toward_full() {
        let mut a = Agent::new(1, 0, 0, 4);
        let target = SpectralIdentity::from_dominant_bin(7);
        a.adapt_toward(&target, 1.0);
        assert!((a.identity.amplitudes[7] - 1.0).abs() < 1e-9);
    }
    #[test] fn agent_adapt_toward_zero() {
        let mut a = Agent::new(1, 7, 0, 4);
        let original = a.identity.clone();
        a.adapt_toward(&SpectralIdentity::from_dominant_bin(0), 0.0);
        assert_eq!(a.identity, original);
    }
    #[test] fn agent_spectral_distance_self_zero() {
        let a = Agent::new(1, 3, 0, 4);
        assert!(a.spectral_distance(&a) < 1e-9);
    }
    #[test] fn agent_conservation_accumulates_on_play() {
        let mut a = Agent::new(1, 7, -4, 4);
        a.set_state(AgentState::Playing);
        a.tick(); a.tick();
        assert!(a.conservation.energy > 0.0);
    }
}
