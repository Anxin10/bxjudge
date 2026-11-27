// Synthetic /judge/ data in the shape bxjudge.judge_view.build_view writes
// (index.json + data/<key>.json). No run output, frames or video are used.

const FPS = 120;
const W = 1280;
const H = 720;
// No video is attached, so the page draws at currentTime 0. A negative media offset puts
// that at t = 1.0 s (frame 120), where the fixture has boxes, trails and the zone reference.
const MEDIA_OFFSET = -1.0;
const SAMPLE_FRAME = 120;

const STAGES = [
  ['detect', '偵測'], ['track', '兩顆追蹤'], ['geometry', '場地幾何'],
  ['identity', '身分'], ['finish', '終結判定'], ['other', '其他'],
];
// Mirrors judge_view.stage_of for the codes used here.
function stageOf(reason) {
  const code = String(reason).split('@', 1)[0];
  const exact = { launch_release_unobserved: 'detect', id_swap_risk: 'identity', position_unknown: 'track' };
  if (exact[code]) return exact[code];
  const prefix = [['after_finish:', 'finish'], ['two_bey_', 'track'], ['track_', 'track'],
    ['calibration_', 'geometry'], ['identity_', 'identity'], ['finish_', 'finish']];
  for (const [p, s] of prefix) if (code.startsWith(p)) return s;
  return code.includes('_stop_') ? 'finish' : 'other';
}
function stages(reasons) {
  const flagged = {};
  for (const r of reasons) (flagged[stageOf(r)] ||= []).push(r);
  return STAGES.filter(([id]) => id !== 'other' || flagged.other)
    .map(([id, label]) => ({ id, label, status: flagged[id] ? 'flag' : 'ok', reasons: flagged[id] || [] }));
}

function ellipse(cx, cy, rx, ry, n = 32) {
  return Array.from({ length: n }, (_, i) => {
    const a = (i / n) * 2 * Math.PI;
    return [Math.round(cx + rx * Math.cos(a)), Math.round(cy + ry * Math.sin(a))];
  });
}
const ZONES = {
  stadium: ellipse(640, 380, 430, 300),
  battle_area: ellipse(640, 380, 300, 205),
  xtreme: [[600, 676], [680, 676], [690, 712], [590, 712]],
  over_left: [[214, 300], [250, 260], [262, 420], [220, 440]],
  over_right: [[1066, 300], [1030, 260], [1018, 420], [1060, 440]],
};

/** Points [t, x, y, score, track_id, zone, frame, segment] orbiting the battle area. */
function orbit(entity, phase, radius, gapFrames = []) {
  const points = [];
  let segment = 1;
  for (let frame = 0; frame <= SAMPLE_FRAME; frame += 2) {
    if (gapFrames.includes(frame)) { segment += 1; continue; }
    const a = phase + frame / 22;
    points.push([+(frame / FPS).toFixed(4), +(640 + radius * Math.cos(a)).toFixed(1),
      +(380 + radius * 0.68 * Math.sin(a)).toFixed(1), 0.9, entity, 'battle_area', frame, segment]);
  }
  return points;
}
function boxAt(points, label, score) {
  const p = points[points.length - 1];
  return [p[1] - 28, p[2] - 28, p[1] + 28, p[2] + 28, score, 'primary', label];
}

function round({ video, number, state, finish = null, winner = null, points = 0, confidence, reasons,
  reviewed = false, geometry = true, calibrationScore = 0.91, tLaunch = 0.42, tEnd = 18.6, identityKnown = true }) {
  const x = orbit(1, 0, 170);
  const b = orbit(2, Math.PI, 120, [60, 62]);
  const key = `${video.replace(/\.mp4$/, '')}_r${number}`;
  const frameBoxes = [boxAt(x, identityKnown ? 'X' : '#1 · 身分未知', 0.93), boxAt(b, identityKnown ? 'B' : '#2 · 身分未知', 0.81),
    [890, 180, 930, 220, 0.34, 'candidate', 'low_score_hand']];
  const labels = { spin: '耐久', burst: '爆裂', over: '超出', xtreme: '極限' };
  return {
    video, round: number, state, finish, finish_label: finish ? labels[finish] : '未判定', winner, points,
    confidence, threshold: 0.8, above_threshold: confidence >= 0.8, reasons, reviewed,
    t_launch: tLaunch, t_end: tEnd, key, src: null, fps: FPS, media_offset: MEDIA_OFFSET,
    coordinate_size: [W, H], stages: stages(reasons),
    geometry: geometry
      ? { status: 'launch_reference', zones: ZONES, sample_frame: SAMPLE_FRAME, sample_status: 'temporal_verified' }
      : { status: 'unavailable', zones: {} },
    calibration: { id: 'synthetic-calib', kind: 'autocalib', score: calibrationScore },
    tracks: { X: x, B: b },
    boxes: [[1.0, SAMPLE_FRAME, frameBoxes]],
    detector: { kind: 'yolo', weights: 'synthetic.pt' },
    vlm_note: null,
  };
}

export const rounds = {
  synthetic_match_a_r1: round({ video: 'synthetic_match_a.mp4', number: 1, state: 'judged', finish: 'over',
    winner: 'X', points: 2, confidence: 0.93, reasons: [] }),
  synthetic_match_a_r2: round({ video: 'synthetic_match_a.mp4', number: 2, state: 'review', confidence: 0.41,
    reasons: ['launch_release_unobserved@B', 'id_swap_risk@X:2300', 'X_stop_reversed', 'after_finish:track_lost', 'unmapped_example_code'],
    identityKnown: false }),
  synthetic_match_b_r1: round({ video: 'synthetic_match_b.mp4', number: 1, state: 'review', confidence: 0.66,
    reasons: ['calibration_low_score', 'track_evidence_gap@B:880'], geometry: false, calibrationScore: 0.38 }),
  synthetic_match_b_r2: round({ video: 'synthetic_match_b.mp4', number: 2, state: 'judged', finish: 'spin',
    winner: 'B', points: 1, confidence: 0.88, reasons: [], reviewed: true }),
};

export function index(keys = Object.keys(rounds)) {
  const list = keys.map((k) => rounds[k]).map((r) => ({
    video: r.video, round: r.round, state: r.state, finish_label: r.finish_label, winner: r.winner,
    confidence: r.confidence, reviewed: r.reviewed, key: r.key, reason_count: r.reasons.length,
    stages: r.stages.filter((s) => s.status === 'flag').map((s) => s.id),
  })).sort((a, b) => (a.state !== 'review') - (b.state !== 'review') || b.reason_count - a.reason_count || a.key.localeCompare(b.key));
  return {
    run: 'runs/synthetic-storybook', round_count: list.length,
    review_count: list.filter((r) => r.state === 'review').length, threshold: 0.8, top_reasons: [],
    stores: { 'reid-review': 1, 'identity-audit': 2 }, rounds: list,
  };
}
