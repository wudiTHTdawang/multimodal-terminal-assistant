let data;
let pendingMessage;
let currentTrack;
let activeMode;
let switchArmed = false;
let cameraStream;
let faceLandmarker;
let faceDetectionFrame;
let lastFaceDetectionAt = 0;
let handGestureRecognizer;
let handGesturesEnabled = false;
let lastHandDetectionAt = 0;
let handGestureCandidate;
let lastHandGestureAt = 0;
let handMotionHistory = [];
let handOpenSeenUntil = 0;
let handGestureSuppressHeadUntil = 0;
let stablePalmPose;
let stablePalmPoseAt = 0;
let lastFacePresence;
let latestEyeFeatures;
let gazeMapper;
let previousGazeMapper;
let gazeTargetElement;
let gazeTargetSince = 0;
let gazeTargetLastMatchedAt = 0;
let gazeTargetLocked = false;
let gazeCandidateScores = [];
let gazeCandidateZone;
let gazeCandidateReliabilityUpdatedAt = 0;
let calibrationActive = false;
let calibrationTimer;
let calibrationSamples = [];
let eyeFeatureHistory = [];
let lastLockedGaze;
let gazeSuggestionCooldownUntil = 0;
let selectedContactId;
let selectedContactSource;
let pendingGazeSuggestion;
let headMotionHistory = [];
let lastHeadGestureAt = 0;
let messageDecisionInProgress = false;
let musicGazeTrackId;
let musicGestureReadyAt = 0;
// 选中的是“正在播放”交互卡片，而不是某一首具体歌曲；切歌后仍持续有效。
let musicCardSelected = false;
let pendingMusicGazeSuggestion;
// 歌曲卡片锁定后，头部动作会暂时改变眼部特征。短暂保留锁定，避免点头/摇头
// 被下一帧的视线估计误认为“离开了卡片”。
let musicGazeGestureWindowUntil = 0;
// 当前歌曲已收到明确的“喜欢”反馈后，直到换歌前不再用头部动作重复判断。
let musicFeedbackLockedTrackId;
let currentRecommendationReason = '';
let currentPreferencePlaylist = [];
let playbackTimer;
let playbackDeadline = 0;
let playbackTrackId;
let playbackFeedbackRecorded = false;
let playbackPaused = false;
let pausedPlaybackRemainingMs = 0;
let memoAuthorizationMode = 'merge';
let authorizedSources = [];

const DEMO_TRACK_DURATION_SECONDS = 18;
const MUSIC_GAZE_GESTURE_WINDOW_MS = 3200;
// 联系人卡片以黄色候选状态保持此时长后，立即询问用户，不再等待另一套锁定条件。
const CONTACT_GAZE_PROMPT_DWELL_MS = 650;
// 摄像头每帧的视线落点会有轻微抖动。短暂保留上一候选，既避免黄色框闪烁，
// 也避免联系人确认的累计时间反复归零。
const GAZE_CANDIDATE_HOLD_MS = 420;
const MUSIC_GAZE_MINIMUM_SCORE = 0.54;
// 只在持续候选超过一个较长的间隔后才学习一次，避免把每帧检测噪声写入校准记录。
const GAZE_CANDIDATE_REWARD_INTERVAL_MS = 600;
const GAZE_CANDIDATE_ABANDON_MIN_MS = 260;
const HAND_GESTURE_INTERVAL_MS = 100;
const HAND_GESTURE_DWELL_MS = 300;
const HAND_GESTURE_CANDIDATE_GAP_MS = 300;
const HAND_WAVE_WINDOW_MS = 900;
const HAND_WAVE_MIN_HORIZONTAL_SPAN = 0.065;
const HAND_WAVE_MIN_TOTAL_SPAN = 0.12;
const HAND_WAVE_MAX_VERTICAL_SPAN = 0.34;
const PALM_TOGGLE_TRANSITION_WINDOW_MS = 1600;
const HAND_GESTURE_COOLDOWN_MS = 450;
const HAND_GESTURE_HEAD_SUPPRESS_MS = 900;
const MUSIC_HEAD_GESTURE_WARMUP_MS = 420;

const FACE_TASK_VERSION = '1.0.1';
// 模型与主脚本随项目发布，避免 Google Storage 或 CDN 被网络策略拦截后导致功能失效。
const FACE_MODEL_URL = '/models/face_landmarker.task';
const HAND_GESTURE_MODEL_URL = '/models/gesture_recognizer.task';
const FACE_BUNDLE_URL = '/vendor/mediapipe/vision_bundle.mjs';
// WASM 体积较大，保留两个等价来源；一个无法访问时会自动尝试另一个。
const FACE_WASM_URLS = [
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${FACE_TASK_VERSION}/wasm`,
  `https://unpkg.com/@mediapipe/tasks-vision@${FACE_TASK_VERSION}/wasm`,
];
const GAZE_CALIBRATION_STORAGE_KEY = 'zhiji.gaze-calibration.v4';
const EYE_LANDMARKS = [33, 133, 159, 145, 160, 158, 153, 144, 362, 263, 386, 374, 385, 387, 373, 380, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477];
const CALIBRATION_POINTS = [
  { x: 0.16, y: 0.24 }, { x: 0.50, y: 0.24 }, { x: 0.84, y: 0.24 },
  { x: 0.16, y: 0.52 }, { x: 0.50, y: 0.52 }, { x: 0.84, y: 0.52 },
  { x: 0.16, y: 0.80 }, { x: 0.50, y: 0.80 }, { x: 0.84, y: 0.80 },
];

const $ = (selector) => document.querySelector(selector);

async function api(action, extra = {}) {
  const response = await fetch('/api/action', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, ...extra }),
  });
  const payload = await response.json();
  if (!payload.ok) throw new Error(payload.error);
  return payload.result;
}

function toast(message) {
  const node = $('#toast');
  node.textContent = message;
  node.classList.add('show');
  window.setTimeout(() => node.classList.remove('show'), 2400);
}

function setCameraStatus(message, status = 'idle') {
  const node = $('#camera-status');
  node.textContent = message;
  node.dataset.status = status;
}

function updateCameraControls(active) {
  $('#start-camera').disabled = active;
  $('#calibrate-gaze').disabled = !active;
  $('#clear-gaze-calibration').disabled = !gazeMapper;
  $('#stop-camera').disabled = !active;
  $('#camera-panel').hidden = !active;
}

function setCameraContext(label) {
  $('#camera-context').textContent = `上下文：${label}`;
}

function loadGazeCalibration() {
  try {
    const saved = JSON.parse(localStorage.getItem(GAZE_CALIBRATION_STORAGE_KEY));
    if (saved?.version === 4 && saved.prototypes && saved.zone_stats && saved.linear_x && saved.linear_y) return saved;
  } catch { /* 本地校准损坏时忽略并要求重新校准。 */ }
  return undefined;
}

function saveGazeCalibration() {
  if (gazeMapper) localStorage.setItem(GAZE_CALIBRATION_STORAGE_KEY, JSON.stringify(gazeMapper));
  updateCameraControls(Boolean(cameraStream));
}

function clearSavedGazeCalibration() {
  localStorage.removeItem(GAZE_CALIBRATION_STORAGE_KEY);
  gazeMapper = undefined;
  previousGazeMapper = undefined;
  clearGazeTarget();
  updateCameraControls(Boolean(cameraStream));
  $('#gaze-feedback').textContent = '已清除本机视线校准记录；下次使用请重新校准。';
}

function clearFaceOverlay() {
  const canvas = $('#face-overlay');
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

function drawFaceLandmarks(landmarks) {
  const video = $('#camera-preview');
  const canvas = $('#face-overlay');
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);

  const xValues = landmarks.map((point) => point.x * canvas.width);
  const yValues = landmarks.map((point) => point.y * canvas.height);
  const left = Math.min(...xValues);
  const top = Math.min(...yValues);
  const width = Math.max(...xValues) - left;
  const height = Math.max(...yValues) - top;
  context.strokeStyle = '#62e0a1';
  context.lineWidth = Math.max(2, canvas.width / 250);
  context.strokeRect(left, top, width, height);

  context.fillStyle = '#bfffe0';
  const radius = Math.max(1.8, canvas.width / 280);
  EYE_LANDMARKS.forEach((index) => {
    const point = landmarks[index];
    if (!point) return;
    context.beginPath();
    context.arc(point.x * canvas.width, point.y * canvas.height, radius, 0, Math.PI * 2);
    context.fill();
  });
}

function meanLandmark(landmarks, indexes) {
  const points = indexes.map((index) => landmarks[index]).filter(Boolean);
  if (!points.length) return undefined;
  return {
    x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
    y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
  };
}

function normalizedIrisPosition(landmarks, irisIndexes, eyeIndexes) {
  const iris = meanLandmark(landmarks, irisIndexes);
  const eye = eyeIndexes.map((index) => landmarks[index]).filter(Boolean);
  if (!iris || eye.length !== eyeIndexes.length) return undefined;
  const xValues = eye.map((point) => point.x);
  const yValues = eye.map((point) => point.y);
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);
  if (maxX - minX < 0.002 || maxY - minY < 0.002) return undefined;
  return [(iris.x - minX) / (maxX - minX), (iris.y - minY) / (maxY - minY)];
}

function extractEyeFeatures(landmarks) {
  const left = normalizedIrisPosition(landmarks, [468, 469, 470, 471, 472], [33, 133, 159, 145]);
  const right = normalizedIrisPosition(landmarks, [473, 474, 475, 476, 477], [362, 263, 386, 374]);
  const nose = landmarks[1];
  const xValues = landmarks.map((point) => point.x);
  const yValues = landmarks.map((point) => point.y);
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);
  if (!left || !right || !nose || maxX - minX < 0.01 || maxY - minY < 0.01) return undefined;
  return [
    ...left,
    ...right,
    (nose.x - minX) / (maxX - minX),
    (nose.y - minY) / (maxY - minY),
  ];
}

function solveLinearSystem(matrix, vector) {
  const size = vector.length;
  const augmented = matrix.map((row, index) => [...row, vector[index]]);
  for (let pivot = 0; pivot < size; pivot += 1) {
    let bestRow = pivot;
    for (let row = pivot + 1; row < size; row += 1) {
      if (Math.abs(augmented[row][pivot]) > Math.abs(augmented[bestRow][pivot])) bestRow = row;
    }
    if (Math.abs(augmented[bestRow][pivot]) < 1e-8) throw new Error('校准数据差异不足，请保持正对屏幕后重新校准。');
    [augmented[pivot], augmented[bestRow]] = [augmented[bestRow], augmented[pivot]];
    const divisor = augmented[pivot][pivot];
    for (let column = pivot; column <= size; column += 1) augmented[pivot][column] /= divisor;
    for (let row = 0; row < size; row += 1) {
      if (row === pivot) continue;
      const factor = augmented[row][pivot];
      for (let column = pivot; column <= size; column += 1) augmented[row][column] -= factor * augmented[pivot][column];
    }
  }
  return augmented.map((row) => row[size]);
}

function fitLinearGazeAxis(samples, target) {
  const dimension = samples[0].features.length + 1;
  const matrix = Array.from({ length: dimension }, () => Array(dimension).fill(0));
  const vector = Array(dimension).fill(0);
  samples.forEach((sample) => {
    const row = [1, ...sample.features];
    const value = target(sample);
    row.forEach((left, i) => {
      vector[i] += left * value;
      row.forEach((right, j) => { matrix[i][j] += left * right; });
    });
  });
  // 轻微正则化：避免九个样本的眼部特征接近时矩阵不可逆。
  matrix.forEach((row, index) => { row[index] += 0.001; });
  return solveLinearSystem(matrix, vector);
}

function fitGazeMapper(samples) {
  const prototypes = {};
  samples.forEach((sample) => {
    prototypes[gazeZone({ x: sample.screenX, y: sample.screenY })] = sample.features;
  });
  if (Object.keys(prototypes).length !== CALIBRATION_POINTS.length) {
    throw new Error('校准区域不完整，请重新完成 9 点校准。');
  }
  return {
    version: 4,
    prototypes,
    linear_x: fitLinearGazeAxis(samples, (sample) => sample.screenX / window.innerWidth),
    linear_y: fitLinearGazeAxis(samples, (sample) => sample.screenY / window.innerHeight),
    zone_stats: Object.fromEntries(Object.keys(prototypes).map((zone) => [zone, { reliability: 1, success: 0, cancel: 0 }])),
    saved_at_ms: Date.now(),
  };
}

function predictGazePoint(features) {
  if (!gazeMapper) return undefined;
  const ranked = Object.entries(gazeMapper.prototypes).map(([zone, prototype]) => ({
    zone,
    distance: Math.hypot(...features.map((value, index) => value - prototype[index])),
  })).sort((left, right) => left.distance - right.distance);
  const [best, second] = ranked;
  if (!best || !second) return undefined;
  const margin = Math.max(0, Math.min(1, (second.distance - best.distance) / Math.max(second.distance, 0.0001)));
  const reliability = gazeMapper.zone_stats[best.zone]?.reliability ?? 1;
  const maxDistance = Math.max(...ranked.map((item) => item.distance), 0.0001);
  const zoneScores = Object.fromEntries(ranked.map((item) => [
    item.zone,
    Math.max(0.05, (1 - item.distance / maxDistance) * (gazeMapper.zone_stats[item.zone]?.reliability ?? 1)),
  ]));
  // 以九点校准拟合出的线性映射得到连续屏幕坐标。这个坐标用于卡片几何匹配；
  // confidence 仅保留为“样本分离程度”，不再单独决定音乐卡片是否锁定。
  const row = [1, ...features];
  const point = {
    x: gazeMapper.linear_x.reduce((sum, coefficient, index) => sum + coefficient * row[index], 0),
    y: gazeMapper.linear_y.reduce((sum, coefficient, index) => sum + coefficient * row[index], 0),
  };
  return { zone: best.zone, point, confidence: Math.max(0, Math.min(1, (0.4 + margin * 0.6) * reliability)), zoneScores };
}

function clearGazeTarget() {
  document.querySelectorAll('.gaze-candidate, .gaze-focused').forEach((node) => {
    node.classList.remove('gaze-candidate', 'gaze-focused');
  });
  gazeTargetElement = undefined;
  gazeTargetLocked = false;
  gazeTargetSince = 0;
  gazeTargetLastMatchedAt = 0;
  gazeCandidateScores = [];
  gazeCandidateZone = undefined;
  gazeCandidateReliabilityUpdatedAt = 0;
  musicGazeGestureWindowUntil = 0;
  musicGestureReadyAt = 0;
}

function clearMusicTrackSelection(message) {
  document.querySelectorAll('.music-track-card.music-selected').forEach((node) => node.classList.remove('music-selected'));
  document.querySelector('.music-gaze-prompt')?.remove();
  musicCardSelected = false;
  pendingMusicGazeSuggestion = undefined;
  musicGazeTrackId = undefined;
  musicGestureReadyAt = 0;
  clearGazeTarget();
  if (message) $('#gaze-feedback').textContent = message;
}

function clearGazeSuggestion() {
  document.querySelector('.contact-gaze-prompt')?.remove();
  pendingGazeSuggestion = undefined;
  headMotionHistory = [];
}

function adjustGazeReliability(outcome) {
  if (!gazeMapper || !lastLockedGaze?.zone) return;
  const stats = gazeMapper.zone_stats[lastLockedGaze.zone];
  if (!stats) return;
  if (outcome === 'success') {
    stats.success += 1;
    stats.reliability = Math.min(1.2, +(stats.reliability + 0.03).toFixed(3));
    $('#gaze-feedback').textContent = `已记录本次视线选择成功，${lastLockedGaze.zone} 区域可靠度提升至 ${stats.reliability.toFixed(2)}。`;
  } else {
    stats.cancel += 1;
    // 取消不一定由视线误判造成，因此只小幅降低可靠度。
    stats.reliability = Math.max(0.7, +(stats.reliability - 0.02).toFixed(3));
    $('#gaze-feedback').textContent = `已记录本次取消，${lastLockedGaze.zone} 区域可靠度微调至 ${stats.reliability.toFixed(2)}。`;
  }
  saveGazeCalibration();
}

function tuneCandidateReliability(zone, delta, field) {
  if (!gazeMapper || !zone) return;
  const stats = gazeMapper.zone_stats[zone];
  if (!stats) return;
  stats[field] = (stats[field] || 0) + 1;
  // 候选停留只是弱反馈，调整幅度远小于“实际发送成功/取消”的明确反馈。
  stats.reliability = Math.max(0.72, Math.min(1.15, +(stats.reliability + delta).toFixed(3)));
  saveGazeCalibration();
}

function rewardStableGazeCandidate(now) {
  if (!gazeTargetElement || !gazeCandidateZone) return;
  if (now - gazeCandidateReliabilityUpdatedAt < GAZE_CANDIDATE_REWARD_INTERVAL_MS) return;
  tuneCandidateReliability(gazeCandidateZone, 0.008, 'stable_candidate');
  gazeCandidateReliabilityUpdatedAt = now;
}

function penalizeAbandonedGazeCandidate(now) {
  if (!gazeTargetElement || !gazeCandidateZone) return;
  // 仅对已经形成可见黄色候选、随后又离开的情况做一次极小下调。
  if (now - gazeTargetSince < GAZE_CANDIDATE_ABANDON_MIN_MS) return;
  tuneCandidateReliability(gazeCandidateZone, -0.006, 'abandoned_candidate');
}

function eligibleGazeElements() {
  const activePage = document.querySelector('.page.active')?.id;
  const selector = activePage === 'message-page' ? '.contact' : activePage === 'music-page' ? '.music-track-card' : '.schedule-item';
  return [...document.querySelectorAll(selector)].filter((node) => {
    const rect = node.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0 && !node.closest('.muted');
  });
}

function gazeZone(point) {
  const column = point.x < window.innerWidth / 3 ? 'left' : point.x < window.innerWidth * 2 / 3 ? 'center' : 'right';
  const row = point.y < window.innerHeight / 3 ? 'top' : point.y < window.innerHeight * 2 / 3 ? 'middle' : 'bottom';
  return `${row}-${column}`;
}

function elementZone(node) {
  const rect = node.getBoundingClientRect();
  return gazeZone({ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 });
}

function zoneCenter(zone) {
  const [row, column] = zone.split('-');
  const coordinate = { left: 1 / 6, center: 0.5, right: 5 / 6, top: 1 / 6, middle: 0.5, bottom: 5 / 6 };
  return { x: coordinate[column] ?? 0.5, y: coordinate[row] ?? 0.5 };
}

function gazeTargetScore(node, prediction) {
  const rect = node.getBoundingClientRect();
  const target = { x: (rect.left + rect.width / 2) / window.innerWidth, y: (rect.top + rect.height / 2) / window.innerHeight };
  const predicted = zoneCenter(prediction.zone);
  const distance = Math.hypot(target.x - predicted.x, target.y - predicted.y);
  const proximity = Math.max(0, 1 - distance / 0.9);
  // 中心位置只是小幅先验，不会覆盖摄像头实际预测到的区域。
  const centerPrior = Math.max(0, 1 - Math.hypot(target.x - 0.5, target.y - 0.5) / 0.71);
  const zoneScore = prediction.zoneScores[elementZone(node)] || 0;
  return zoneScore * 0.62 + proximity * 0.28 + centerPrior * 0.10;
}

function musicCardGazeScore(node, prediction) {
  if (!prediction.point) return 0;
  const rect = node.getBoundingClientRect();
  const x = prediction.point.x * window.innerWidth;
  const y = prediction.point.y * window.innerHeight;
  const point = { x, y };
  const distance = distanceToRect(point, rect);
  // 整张播放卡片均为有效区域，并允许约一个文本行高度的校准误差。
  const tolerance = Math.max(48, Math.min(120, Math.min(rect.width, rect.height) * 0.60));
  if (distance > tolerance) return 0;
  const spatialMatch = 1 - distance / tolerance;
  return 0.62 + spatialMatch * 0.30 + prediction.confidence * 0.08;
}

function targetMetadata(node) {
  const page = document.querySelector('.page.active')?.id.replace('-page', '') || 'unknown';
  const isSchedule = node.classList.contains('schedule-item');
  const isMusicTrack = node.classList.contains('music-track-card');
  const label = node.querySelector('strong')?.textContent || node.querySelector('.schedule-title')?.textContent || '当前项目';
  return {
    page,
    target_type: isSchedule ? 'schedule_item' : isMusicTrack ? 'music_track' : 'contact',
    target_id: node.dataset.id || node.dataset.trackId || node.querySelector('[data-event-key]')?.dataset.eventKey || label,
    label,
  };
}

function validateBrowserEvent(event) {
  if (!['gaze', 'screen_context', 'speech_text', 'head_gesture', 'hand_gesture'].includes(event?.modality)) {
    throw new Error('浏览器事件模态不合法。');
  }
  if (!Number.isInteger(event.timestamp_ms) || event.timestamp_ms <= 0) {
    throw new Error('浏览器事件缺少毫秒时间戳。');
  }
  if (typeof event.confidence !== 'number' || event.confidence < 0 || event.confidence > 1) {
    throw new Error('浏览器事件置信度不在 0 到 1 之间。');
  }
  if (!event.payload || typeof event.payload !== 'object') {
    throw new Error('浏览器事件缺少 payload。');
  }
  if (['head_gesture', 'hand_gesture'].includes(event.modality)
    && (!['message', 'music'].includes(event.payload.page) || !['confirm', 'reject', 'toggle_playback', 'skip_track'].includes(event.payload.decision))) {
    throw new Error('视觉确认事件缺少页面或支持的操作结果。');
  }
}

async function recordBrowserEvent(event) {
  try {
    validateBrowserEvent(event);
    const result = await api('record_multimodal_event', { event });
    // 面板默认折叠，只有用户主动展开时才额外刷新，避免为调试展示增加日常请求。
    if ($('#multimodal-inspector')?.open) void refreshMultimodalInspector();
    return result;
  } catch (error) {
    // 结构化事件失败不阻断本地交互，但会让开发者在控制台看到明确原因。
    console.warn('多模态事件未记录：', error.message);
  }
}

function describeMultimodalEvent(event) {
  const payload = event.payload || {};
  if (event.modality === 'gaze') return `注视 ${payload.target_type === 'contact' ? '联系人' : '页面对象'}：${payload.target_id || '未命名对象'}（停留 ${payload.dwell_ms || 0}ms）`;
  if (event.modality === 'screen_context') return `页面上下文：${payload.page || 'unknown'}，可见 ${payload.visible_targets?.length || 0} 个对象`;
  if (event.modality === 'speech_text') return `模拟语音：${payload.text || '空文本'}`;
  if (event.modality === 'head_gesture') return `头部动作：${payload.decision === 'confirm' ? '点头确认' : '摇头拒绝'}`;
  if (event.modality === 'hand_gesture') return `手势输入：${payload.gesture || payload.decision}`;
  return '未知结构化事件';
}

function renderMultimodalInspector(events) {
  const target = $('#multimodal-event-list');
  if (!target) return;
  if (!events?.length) {
    target.textContent = '最近 10 秒暂未记录事件。';
    return;
  }
  target.innerHTML = [...events].reverse().map((event) => {
    const time = new Date(event.timestamp_ms).toLocaleTimeString('zh-CN', { hour12: false });
    return `<article class="multimodal-event"><strong>${escapeHtml(event.modality)}</strong><span>${escapeHtml(describeMultimodalEvent(event))}</span><time>${escapeHtml(time)}</time></article>`;
  }).join('');
}

async function refreshMultimodalInspector() {
  try {
    const result = await api('get_recent_multimodal_events');
    renderMultimodalInspector(result.events);
  } catch (error) {
    const target = $('#multimodal-event-list');
    if (target) target.textContent = `读取本地事件失败：${error.message}`;
  }
}

function recordScreenContext() {
  const page = document.querySelector('.page.active')?.id.replace('-page', '') || 'unknown';
  const visibleTargets = eligibleGazeElements().map((node) => ({
    ...targetMetadata(node),
    zone: elementZone(node),
  }));
  return recordBrowserEvent({
    modality: 'screen_context',
    timestamp_ms: Date.now(),
    confidence: 1,
    payload: { page, visible_targets: visibleTargets },
  });
}

async function recordHeadDecision(decision, purpose, page = 'message') {
  await recordBrowserEvent({
    modality: 'head_gesture',
    timestamp_ms: Date.now(),
    confidence: 0.72,
    payload: { page, decision, gesture: decision === 'confirm' ? 'nod' : 'shake', purpose },
  });
}

async function recordHandGesture(decision, gesture, purpose, page = 'message', confidence = 0.75) {
  await recordBrowserEvent({
    modality: 'hand_gesture',
    timestamp_ms: Date.now(),
    confidence,
    payload: { page, decision, gesture, purpose },
  });
}

async function finishMessageDecision(action, outcome, source = 'button') {
  if (!pendingMessage || messageDecisionInProgress) return;
  messageDecisionInProgress = true;
  try {
    if (source === 'head') await recordHeadDecision(outcome === 'success' ? 'confirm' : 'reject', 'message_confirmation');
    if (source === 'hand') await recordHandGesture(outcome === 'success' ? 'confirm' : 'reject', outcome === 'success' ? 'Thumb_Up' : 'Thumb_Down', 'message_confirmation');
    const result = await api(action, pendingMessage);
    adjustGazeReliability(outcome);
    resetMessageForm(result.message);
  } catch (error) {
    toast(error.message);
  } finally {
    messageDecisionInProgress = false;
  }
}

function renderMessageUnderstanding(result) {
  const explanation = result.explanation?.length
    ? `<ul>${result.explanation.map((item) => `<li>${item}</li>`).join('')}</ul>` : '';
  const actions = result.pending
    ? '<p><button class="primary" id="confirm-send">确认发送</button><button class="secondary" id="cancel-send">取消</button></p><small class="decision-hint">也可在模拟语音框中说“是 / 确认”或“不用 / 取消”，或点头 / 摇头。</small>' : '';
  $('#message-result').innerHTML = `<div class="result-box"><strong>${result.message}</strong>${explanation}${actions}</div>`;
  if (result.pending) {
    pendingMessage = result.pending;
    $('#confirm-send').onclick = () => finishMessageDecision('confirm_send', 'success');
    $('#cancel-send').onclick = () => finishMessageDecision('cancel_message', 'cancel');
  }
}

async function submitSimulatedSpeech() {
  const text = $('#message-content').value.trim();
  if (!text) {
    toast('请输入或选择一条模拟语音指令。');
    return;
  }

  // 联系人候选出现时，优先将简短的“是 / 不用”视为对候选的回答，
  // 而不是一条新的消息发送指令。
  if (pendingGazeSuggestion) {
    const decision = parseContactSuggestionSpeech(text);
    if (decision === 'confirm') {
      const suggestion = pendingGazeSuggestion;
      clearGazeSuggestion();
      $('#message-content').value = '';
      await lockGazeTarget(suggestion.node, suggestion.zone, suggestion.confidence);
      return;
    }
    if (decision === 'reject') {
      rejectGazeSuggestion();
      $('#message-content').value = '';
      return;
    }
    toast('可以说“是”或“不用”，也可以点头或摇头。');
    return;
  }

  const timestamp = Date.now();
  const isMessageDecision = Boolean(pendingMessage);
  if (isMessageDecision) messageDecisionInProgress = true;
  try {
    await recordBrowserEvent({
      modality: 'speech_text',
      timestamp_ms: timestamp,
      confidence: 1,
      payload: { text, page: 'message', source: 'simulated' },
    });
    const result = await api('understand_multimodal_command', {
      speech_timestamp_ms: timestamp,
      preferred_contact_id: selectedContactSource === 'manual' ? selectedContactId : undefined,
    });
    if (result.clear_message_form) {
      if (result.intent === 'confirm') adjustGazeReliability('success');
      if (result.intent === 'cancel') adjustGazeReliability('cancel');
      resetMessageForm(result.message);
      return;
    }
    renderMessageUnderstanding(result);
  } catch (error) {
    toast(error.message);
  } finally {
    if (isMessageDecision) messageDecisionInProgress = false;
  }
}

async function submitSimulatedMusicCommand() {
  const text = $('#music-command').value.trim();
  if (!text) {
    toast('请输入或选择一条音乐模拟语音指令。');
    return;
  }
  const timestamp = Date.now();
  try {
    await recordBrowserEvent({
      modality: 'speech_text',
      timestamp_ms: timestamp,
      confidence: 1,
      payload: { text, page: 'music', source: 'simulated' },
    });
    const result = await api('understand_multimodal_command', { speech_timestamp_ms: timestamp });
    $('#music-command').value = '';
    if (result.intent === 'cancel_music_selection') {
      clearMusicTrackSelection(result.message);
      toast(result.message);
      return;
    }
    if (result.intent === 'next_track') {
      await nextTrack();
      return;
    }
    toast(result.message);
  } catch (error) { toast(error.message); }
}

function distanceToRect(point, rect) {
  const dx = Math.max(rect.left - point.x, 0, point.x - rect.right);
  const dy = Math.max(rect.top - point.y, 0, point.y - rect.bottom);
  return Math.hypot(dx, dy);
}

async function lockGazeTarget(node, zone, confidence) {
  clearGazeSuggestion();
  node.classList.add('gaze-focused');
  const metadata = targetMetadata(node);
  const label = metadata.label;
  const isContact = node.classList.contains('contact');
  if (isContact) {
    // 先锁定本轮联系人，避免等待事件写入或接口响应时又弹出新的候选提示。
    selectedContactId = node.dataset.id;
    selectedContactSource = 'gaze';
  }
  $('#gaze-feedback').textContent = `已持续注视：${label}`;
  await recordBrowserEvent({
    modality: 'gaze',
    timestamp_ms: Date.now(),
    confidence,
    payload: {
      page: metadata.page,
      target_type: metadata.target_type,
      target_id: metadata.target_id,
      zone,
      dwell_ms: Math.round(performance.now() - gazeTargetSince),
      calibration: '9-point-multi-frame-v2',
    },
  });
  lastLockedGaze = { zone, target_id: metadata.target_id };
  if (isContact) {
    try {
      await api('select_contact', { contact_id: node.dataset.id });
      document.querySelectorAll('.contact').forEach((item) => item.classList.remove('selected'));
      node.classList.add('selected');
      $('#message-result').innerHTML = '';
      $('#gaze-feedback').textContent = `已确认选择：${label}。现在可提交模拟语音指令。`;
      toast(`已通过摄像头注视选择：${label}`);
    } catch (error) {
      selectedContactId = undefined;
      selectedContactSource = undefined;
      toast(error.message);
    }
  } else if (node.classList.contains('music-track-card')) {
    musicGazeTrackId = node.dataset.trackId;
    // 进入明确的“等待反馈”窗口：这段时间只识别点头/摇头，不再用每一帧的
    // 眼部特征重新竞争卡片，避免头部动作本身冲掉已经确认的注视锁定。
    musicGazeGestureWindowUntil = performance.now() + MUSIC_GAZE_GESTURE_WINDOW_MS;
    // 锁定视线后先留出短暂基线期，让用户自然调整坐姿的微动不被误认为点头/摇头。
    musicGestureReadyAt = performance.now() + MUSIC_HEAD_GESTURE_WARMUP_MS;
    // 只从“确认注视歌曲卡片”这一刻开始采集头部运动，避免把此前的自然晃动误判成偏好。
    headMotionHistory = [];
    $('#gaze-feedback').textContent = `已确认注视：${label}。请在 3 秒内点头表示喜欢，摇头表示不喜欢并换歌。`;
  }
}

function parseContactSuggestionSpeech(text) {
  const normalized = text.replace(/[，。！？、\s]/g, '').toLowerCase();
  if (/^(是|是的|好的|好|确认|选中|选择|帮我选|可以)$/.test(normalized)) return 'confirm';
  if (/^(不|不是|不用|暂不|取消|取消发送|继续识别|不要|不要发送|不发送)$/.test(normalized)) return 'reject';
  return undefined;
}

function positionGazePrompt(prompt, node) {
  const rect = node.getBoundingClientRect();
  const width = 250;
  const left = Math.max(12, Math.min(rect.left, window.innerWidth - width - 12));
  const top = Math.min(rect.bottom + 8, window.innerHeight - 132);
  prompt.style.left = `${left}px`;
  prompt.style.top = `${Math.max(12, top)}px`;
}

function rejectGazeSuggestion() {
  gazeSuggestionCooldownUntil = Date.now() + 3000;
  clearGazeSuggestion();
  clearGazeTarget();
  $('#gaze-feedback').textContent = '好的，我会继续留意你的注视。';
}

function showContactGazeSuggestion(node, zone, confidence) {
  if (selectedContactId || pendingMessage || pendingGazeSuggestion) return;
  const metadata = targetMetadata(node);
  const prompt = document.createElement('section');
  prompt.className = 'contact-gaze-prompt';
  prompt.innerHTML = `<strong>看起来你想联系 ${metadata.label}</strong><p>需要我帮你选中这位联系人吗？</p><div><button class="primary" data-decision="confirm">选中</button><button class="secondary" data-decision="reject">暂不</button></div><small>也可说“是”或“不用”，或点头 / 摇头。</small>`;
  document.body.append(prompt);
  positionGazePrompt(prompt, node);
  pendingGazeSuggestion = { node, zone, confidence };
  headMotionHistory = [];
  prompt.querySelector('[data-decision="confirm"]').onclick = async () => {
    const suggestion = pendingGazeSuggestion;
    clearGazeSuggestion();
    if (suggestion) await lockGazeTarget(suggestion.node, suggestion.zone, suggestion.confidence);
  };
  prompt.querySelector('[data-decision="reject"]').onclick = rejectGazeSuggestion;
}

function dismissMusicGazeSuggestion(message) {
  document.querySelector('.music-gaze-prompt')?.remove();
  pendingMusicGazeSuggestion = undefined;
  clearGazeTarget();
  if (message) $('#gaze-feedback').textContent = message;
}

async function selectMusicTrackForInteraction(node, zone, confidence) {
  document.querySelector('.music-gaze-prompt')?.remove();
  pendingMusicGazeSuggestion = undefined;
  if (!currentTrack || node.dataset.trackId !== currentTrack.id) return;
  await lockGazeTarget(node, zone, confidence);
  musicCardSelected = true;
  node.classList.add('music-selected');
  // 卡片选中后不再把自然转头解释成音乐偏好，只保留语音和手势操作。
  musicGazeTrackId = undefined;
  musicGazeGestureWindowUntil = 0;
  clearGazeTarget();
  headMotionHistory = [];
  $('#gaze-feedback').textContent = '已选中正在播放卡片。现在无论切换到哪首歌，都可使用手势操作；可说“取消当前歌曲选择”解除。';
}

function showMusicGazeSuggestion(node, zone, confidence) {
  if (musicCardSelected || pendingMusicGazeSuggestion || !currentTrack || node.dataset.trackId !== currentTrack.id) return;
  const prompt = document.createElement('section');
  prompt.className = 'music-gaze-prompt';
  prompt.innerHTML = `<strong>似乎想操作正在播放的音乐</strong><p>要选中这个播放卡片吗？</p><div><button class="primary" data-decision="confirm">选中</button><button class="secondary" data-decision="reject">暂不</button></div><small>选中后会持续保留；说“取消当前歌曲选择”即可解除。</small>`;
  document.body.append(prompt);
  positionGazePrompt(prompt, node);
  pendingMusicGazeSuggestion = { node, zone, confidence };
  headMotionHistory = [];
  prompt.querySelector('[data-decision="confirm"]').onclick = async () => {
    const suggestion = pendingMusicGazeSuggestion;
    if (suggestion) await selectMusicTrackForInteraction(suggestion.node, suggestion.zone, suggestion.confidence);
  };
  prompt.querySelector('[data-decision="reject"]').onclick = () => dismissMusicGazeSuggestion('好的，暂不选中这首歌。');
}

function updateGazeTarget(prediction) {
  const activePage = document.querySelector('.page.active')?.id;
  // 在联系人页面已选联系人或已有待发送消息时，不能让新的注视结果改变本轮消息对象。
  // 切换到音乐、日程页面后，它们仍可独立使用视线输入。
  if (activePage === 'message-page' && (selectedContactId || pendingMessage)) {
    clearGazeTarget();
    return;
  }
  // 明确选中播放卡片后，暂停视线竞争，后续仅接收语音和手势，直到用户取消选择或停止模式。
  if (activePage === 'music-page' && musicCardSelected) return;
  if (pendingMusicGazeSuggestion) return;
  // 喜欢当前歌曲后，保留播放状态，但不再重新锁定同一张歌曲卡片。
  // 下一首歌出现时会清除此标记，届时才会再次允许“注视 + 点头/摇头”。
  if (activePage === 'music-page' && currentTrack?.id === musicFeedbackLockedTrackId) {
    clearGazeTarget();
    return;
  }
  const musicGestureWindowOpen = activePage === 'music-page'
    && gazeTargetLocked
    && musicGazeTrackId
    && currentTrack?.id === musicGazeTrackId
    && performance.now() < musicGazeGestureWindowUntil;
  // 已锁定歌曲卡片时，给用户一个短暂且明确的反馈窗口。点头或摇头会明显改变
  // 眼睛、鼻子的相对位置，因此窗口内不再重算注视目标；超时后恢复正常注视判断。
  if (musicGestureWindowOpen) return;
  if (activePage === 'music-page'
    && gazeTargetLocked
    && musicGazeTrackId
    && currentTrack?.id === musicGazeTrackId
    && musicGazeGestureWindowUntil > 0) {
    clearGazeTarget();
    musicGazeTrackId = undefined;
  }
  if (!prediction || calibrationActive || pendingGazeSuggestion) return;
  const { zone, confidence, zoneScores } = prediction;
  const strictMusicGaze = activePage === 'music-page';
  const rankedCandidates = eligibleGazeElements().map((node) => ({
    node,
    zone: elementZone(node),
    score: strictMusicGaze ? musicCardGazeScore(node, prediction) : gazeTargetScore(node, prediction),
  })).sort((left, right) => right.score - left.score);
  const bestCandidate = rankedCandidates[0];
  const secondCandidate = rankedCandidates[1];
  const minimumScore = strictMusicGaze ? MUSIC_GAZE_MINIMUM_SCORE : 0.16;
  const now = performance.now();
  if (!bestCandidate || bestCandidate.score < minimumScore) {
    // 单帧落点短暂跑出卡片范围时保留当前黄色框，而不是立刻清除并重新计时。
    if (gazeTargetElement && now - gazeTargetLastMatchedAt <= GAZE_CANDIDATE_HOLD_MS) return;
    penalizeAbandonedGazeCandidate(now);
    clearGazeTarget();
    $('#gaze-feedback').textContent = '正在估计注视候选，请保持正对屏幕并看向一个卡片。';
    return;
  }
  const { node: closest } = bestCandidate;
  const candidateMargin = bestCandidate.score - (secondCandidate?.score || 0);
  // 候选对象在相邻检测帧之间偶尔切换时，优先保留先前的候选一小段时间。
  // 这样用户持续看同一张卡片时，黄框不会频繁闪烁或把停留时间清零。
  if (gazeTargetElement && gazeTargetElement !== closest
    && now - gazeTargetLastMatchedAt <= GAZE_CANDIDATE_HOLD_MS) return;
  if (gazeTargetElement !== closest) {
    penalizeAbandonedGazeCandidate(now);
    clearGazeTarget();
    gazeTargetElement = closest;
    gazeTargetSince = now;
    gazeTargetLastMatchedAt = now;
    gazeCandidateZone = bestCandidate.zone;
    gazeCandidateReliabilityUpdatedAt = now;
    gazeCandidateScores = [{ time: gazeTargetSince, score: bestCandidate.score }];
    closest.classList.add('gaze-candidate');
    const label = closest.querySelector('strong')?.textContent || closest.querySelector('.schedule-title')?.textContent || '当前项目';
    $('#gaze-feedback').textContent = `正在留意：${label}`;
    return;
  }
  gazeTargetLastMatchedAt = now;
  gazeCandidateScores.push({ time: now, score: bestCandidate.score });
  gazeCandidateScores = gazeCandidateScores.filter((sample) => sample.time >= gazeTargetSince);
  rewardStableGazeCandidate(now);
  if (closest.classList.contains('contact')
    && now - gazeTargetSince >= CONTACT_GAZE_PROMPT_DWELL_MS
    && !selectedContactId
    && !pendingMessage
    && !pendingGazeSuggestion) {
    if (Date.now() >= gazeSuggestionCooldownUntil) {
      gazeTargetLocked = true;
      showContactGazeSuggestion(closest, bestCandidate.zone, Math.max(confidence, bestCandidate.score));
      return;
    }
  }
  const requiredDwellMs = strictMusicGaze ? 550 : 700;
  if (!gazeTargetLocked && performance.now() - gazeTargetSince >= requiredDwellMs) {
    const averageScore = gazeCandidateScores.reduce((sum, sample) => sum + sample.score, 0) / gazeCandidateScores.length;
    if (strictMusicGaze && (gazeCandidateScores.length < 3 || averageScore < 0.60)) return;
    gazeTargetLocked = true;
    if (strictMusicGaze) {
      showMusicGazeSuggestion(closest, bestCandidate.zone, averageScore);
      return;
    }
    if (closest.classList.contains('contact')) {
      // 联系人由上方的黄色候选累计时间直接触发提示，避免依赖本分支。
      return;
    } else if (candidateMargin >= 0.06) {
      // 音乐页以 1.2 秒平均空间匹配分锁定最高分卡片，而不是单帧原型置信度。
      lockGazeTarget(closest, bestCandidate.zone, strictMusicGaze ? averageScore : Math.max(confidence, candidateMargin));
    } else {
      $('#gaze-feedback').textContent = `系统已找到多个接近候选，当前优先显示：${closest.querySelector('strong')?.textContent || '该对象'}。`;
    }
  }
}

function observeHeadGesture(landmarks) {
  // 音乐反馈仅在歌曲卡片锁定后的短暂窗口内接受。窗口内暂停重新判断视线，
  // 使点头/摇头不会因为自身改变了眼部特征而丢失锁定。
  // 抬手、挥手会带动上半身与脸部关键点；手势识别期间不把这类变化误当成摇头。
  if (performance.now() < handGestureSuppressHeadUntil) return;
  const canAnswerMusic = canAnswerMusicFeedback();
  if ((!pendingGazeSuggestion && !pendingMusicGazeSuggestion && (!pendingMessage || messageDecisionInProgress) && !canAnswerMusic) || Date.now() - lastHeadGestureAt < 1500) return;
  const leftEye = landmarks[33];
  const rightEye = landmarks[263];
  const nose = landmarks[1];
  if (!leftEye || !rightEye || !nose) return;
  const eyeCenterX = (leftEye.x + rightEye.x) / 2;
  const eyeCenterY = (leftEye.y + rightEye.y) / 2;
  const eyeDistance = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y);
  if (eyeDistance < 0.01) return;
  headMotionHistory.push({
    time: performance.now(),
    yaw: (nose.x - eyeCenterX) / eyeDistance,
    pitch: (nose.y - eyeCenterY) / eyeDistance,
  });
  const cutoff = performance.now() - 1200;
  headMotionHistory = headMotionHistory.filter((sample) => sample.time >= cutoff);
  if (headMotionHistory.length < 8) return;

  const horizontalRange = Math.max(...headMotionHistory.map((sample) => sample.yaw)) - Math.min(...headMotionHistory.map((sample) => sample.yaw));
  const verticalRange = Math.max(...headMotionHistory.map((sample) => sample.pitch)) - Math.min(...headMotionHistory.map((sample) => sample.pitch));
  // 用“前半段到后半段的方向性变化”识别动作，而非只看抖动范围。
  // 摄像头静止噪声会有小范围波动，但不会持续朝一个方向移动。
  const half = Math.floor(headMotionHistory.length / 2);
  const firstHalf = headMotionHistory.slice(0, half);
  const secondHalf = headMotionHistory.slice(half);
  const average = (samples, field) => samples.reduce((sum, sample) => sum + sample[field], 0) / samples.length;
  const directionalYaw = average(secondHalf, 'yaw') - average(firstHalf, 'yaw');
  const directionalPitch = average(secondHalf, 'pitch') - average(firstHalf, 'pitch');
  const gentleMusicGesture = canAnswerMusic && !pendingGazeSuggestion && !pendingMessage;
  // 音乐卡片锁定后不应把自然坐姿微调当成偏好。音乐反馈使用更大的幅度、
  // 更强的主方向要求；消息页仍保持原有确认手势灵敏度。
  const verticalThreshold = gentleMusicGesture ? 0.075 : 0.04;
  const horizontalThreshold = gentleMusicGesture ? 0.10 : 0.06;
  const dominance = gentleMusicGesture ? 1.5 : 1.3;
  const rangeMultiplier = gentleMusicGesture ? 1.7 : 1.45;
  const verticalGesture = Math.abs(directionalPitch) > verticalThreshold
    && verticalRange > verticalThreshold * rangeMultiplier
    && Math.abs(directionalPitch) > Math.abs(directionalYaw) * dominance;
  const horizontalGesture = Math.abs(directionalYaw) > horizontalThreshold
    && horizontalRange > horizontalThreshold * rangeMultiplier
    && Math.abs(directionalYaw) > Math.abs(directionalPitch) * dominance;
  if (verticalGesture) {
    lastHeadGestureAt = Date.now();
    if (pendingGazeSuggestion) {
      const suggestion = pendingGazeSuggestion;
      clearGazeSuggestion();
      void (async () => {
        await recordHeadDecision('confirm', 'contact_selection');
        await lockGazeTarget(suggestion.node, suggestion.zone, suggestion.confidence);
      })();
    } else if (pendingMusicGazeSuggestion) {
      const suggestion = pendingMusicGazeSuggestion;
      void selectMusicTrackForInteraction(suggestion.node, suggestion.zone, suggestion.confidence);
    } else if (pendingMessage) {
      void finishMessageDecision('confirm_send', 'success', 'head');
    } else {
      void likeCurrentTrack('head');
    }
  } else if (horizontalGesture) {
    lastHeadGestureAt = Date.now();
    if (pendingGazeSuggestion) {
      void recordHeadDecision('reject', 'contact_selection');
      rejectGazeSuggestion();
    } else if (pendingMusicGazeSuggestion) {
      dismissMusicGazeSuggestion('好的，暂不选中这首歌。');
    } else if (pendingMessage) {
      void finishMessageDecision('cancel_message', 'cancel', 'head');
    } else {
      void dislikeCurrentTrack('head');
    }
  }
}

function updateCalibrationPoint(point) {
  const target = $('#gaze-calibration-point');
  target.style.left = `${point.x * 100}vw`;
  target.style.top = `${point.y * 100}vh`;
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function medianEyeFeatures() {
  if (eyeFeatureHistory.length < 6) return undefined;
  return eyeFeatureHistory[0].map((_, index) => median(eyeFeatureHistory.map((sample) => sample[index])));
}

function finishCalibration(success, message) {
  clearTimeout(calibrationTimer);
  calibrationActive = false;
  $('#gaze-calibration').hidden = true;
  if (success) {
    saveGazeCalibration();
    $('#gaze-feedback').textContent = '9 点校准完成。请注视页面中的卡片，系统会以约 1.2 秒的稳定落点锁定最高匹配对象。';
    setCameraStatus('视线校准已完成，可识别当前页面注视对象', 'active');
  } else {
    gazeMapper = previousGazeMapper;
    $('#gaze-feedback').textContent = message;
  }
}

function runCalibrationStep(index) {
  if (!calibrationActive) return;
  if (index >= CALIBRATION_POINTS.length) {
    try {
      gazeMapper = fitGazeMapper(calibrationSamples);
      finishCalibration(true);
    } catch (error) {
      finishCalibration(false, error.message);
    }
    return;
  }
  const point = CALIBRATION_POINTS[index];
  eyeFeatureHistory = [];
  updateCalibrationPoint(point);
  $('#calibration-instruction').textContent = `请持续注视蓝点。正在采集第 ${index + 1} / ${CALIBRATION_POINTS.length} 个位置…`;
  calibrationTimer = window.setTimeout(() => {
    const features = medianEyeFeatures();
    if (!features) {
      $('#calibration-instruction').textContent = '没有检测到眼部关键点，请正对摄像头后保持注视。将重新采集此位置…';
      calibrationTimer = window.setTimeout(() => runCalibrationStep(index), 1300);
      return;
    }
    calibrationSamples.push({
      features,
      screenX: window.innerWidth * point.x,
      screenY: window.innerHeight * point.y,
    });
    calibrationTimer = window.setTimeout(() => runCalibrationStep(index + 1), 650);
  }, 2600);
}

function startGazeCalibration() {
  if (!cameraStream || !latestEyeFeatures) {
    toast('请先开启摄像头，并确认已检测到人脸和眼部关键点。');
    return;
  }
  previousGazeMapper = gazeMapper;
  calibrationSamples = [];
  calibrationActive = true;
  clearGazeTarget();
  $('#gaze-calibration').hidden = false;
  runCalibrationStep(0);
}

async function initializeFaceLandmarker() {
  if (faceLandmarker) return;
  const { FaceLandmarker, FilesetResolver } = await import(FACE_BUNDLE_URL);
  const modelResponse = await fetch(FACE_MODEL_URL, { cache: 'no-store' });
  if (!modelResponse.ok) throw new Error(`本地人脸模型文件不可用（HTTP ${modelResponse.status}）。`);
  let lastError;
  for (const wasmUrl of FACE_WASM_URLS) {
    try {
      const vision = await FilesetResolver.forVisionTasks(wasmUrl);
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        // 当前 MediaPipe 版本要求由运行时自行读取模型路径；直接传 ArrayBuffer
        // 会使内部资源读取器收到错误的对象，进而报 “read is not a function”。
        baseOptions: { modelAssetPath: FACE_MODEL_URL },
        runningMode: 'VIDEO',
        numFaces: 1,
        minFaceDetectionConfidence: 0.55,
        minFacePresenceConfidence: 0.55,
        minTrackingConfidence: 0.55,
      });
      return;
    } catch (error) {
      lastError = error;
    }
  }
  throw new Error(`视觉运行时无法加载，请检查网络是否可访问 jsDelivr 或 unpkg。${lastError?.message ? ` 原因：${lastError.message}` : ''}`);
}

function setHandGestureStatus(message) {
  const node = $('#hand-gesture-status');
  if (node) node.textContent = message;
}

async function initializeHandGestureRecognizer() {
  if (handGestureRecognizer) return;
  const { GestureRecognizer, FilesetResolver } = await import(FACE_BUNDLE_URL);
  const modelResponse = await fetch(HAND_GESTURE_MODEL_URL, { cache: 'no-store' });
  if (!modelResponse.ok) throw new Error(`本地手势模型文件不可用（HTTP ${modelResponse.status}）。`);
  let lastError;
  for (const wasmUrl of FACE_WASM_URLS) {
    try {
      const vision = await FilesetResolver.forVisionTasks(wasmUrl);
      handGestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
        baseOptions: { modelAssetPath: HAND_GESTURE_MODEL_URL },
        runningMode: 'VIDEO',
        numHands: 1,
        minHandDetectionConfidence: 0.62,
        minHandPresenceConfidence: 0.62,
        minTrackingConfidence: 0.58,
      });
      return;
    } catch (error) {
      lastError = error;
    }
  }
  throw new Error(`手势运行时无法加载。${lastError?.message ? ` 原因：${lastError.message}` : ''}`);
}

async function enableHandGestures() {
  if (!cameraStream) {
    return;
  }
  if (handGesturesEnabled) return;
  try {
    setHandGestureStatus('正在加载本机手势模型…');
    await initializeHandGestureRecognizer();
    handGesturesEnabled = true;
    lastHandDetectionAt = 0;
    handGestureCandidate = undefined;
    handOpenSeenUntil = 0;
    handGestureSuppressHeadUntil = 0;
    stablePalmPose = undefined;
    stablePalmPoseAt = 0;
    setHandGestureStatus('手势已开启：点赞确认 / 喜欢，踩拒绝 / 不喜欢，手掌开合暂停 / 继续，横向或斜向挥动张开的手掌切下一首。');
  } catch (error) {
    handGesturesEnabled = false;
    setHandGestureStatus(`手势模型加载失败：${error.message || '未知错误'}`);
  }
}

function canAnswerMusicFeedback() {
  // 已明确选中播放卡片时，头部动作不再控制偏好，避免与他人交流时误触发。
  if (musicCardSelected) return false;
  return Boolean(
    musicGazeTrackId
    && currentTrack?.id === musicGazeTrackId
    && gazeTargetLocked
    && gazeTargetElement?.classList.contains('music-track-card')
    && gazeTargetElement.dataset.trackId === currentTrack.id
    && activeMode
    && performance.now() >= musicGestureReadyAt
    && performance.now() < musicGazeGestureWindowUntil
  );
}

function canProvideMusicPreference() {
  return Boolean(
    activeMode
    && currentTrack
    && currentTrack.id !== musicFeedbackLockedTrackId
    && (musicCardSelected || canAnswerMusicFeedback())
  );
}

function canControlSelectedMusicTrack() {
  return Boolean(
    activeMode
    && currentTrack
    && (musicCardSelected || canAnswerMusicFeedback())
  );
}

async function applyHandGesture(gesture, confidence) {
  if (gesture === 'Thumb_Up') {
    if (pendingGazeSuggestion) {
      const suggestion = pendingGazeSuggestion;
      clearGazeSuggestion();
      await recordHandGesture('confirm', 'Thumb_Up', 'contact_selection', 'message', confidence);
      await lockGazeTarget(suggestion.node, suggestion.zone, suggestion.confidence);
      return true;
    }
    if (pendingMusicGazeSuggestion) {
      const suggestion = pendingMusicGazeSuggestion;
      await selectMusicTrackForInteraction(suggestion.node, suggestion.zone, suggestion.confidence);
      return true;
    }
    if (pendingMessage) {
      await finishMessageDecision('confirm_send', 'success', 'hand');
      return true;
    }
    if (canProvideMusicPreference()) {
      await likeCurrentTrack('hand');
      return true;
    }
  }
  if (gesture === 'Thumb_Down') {
    if (pendingGazeSuggestion) {
      await recordHandGesture('reject', 'Thumb_Down', 'contact_selection', 'message', confidence);
      rejectGazeSuggestion();
      return true;
    }
    if (pendingMusicGazeSuggestion) {
      dismissMusicGazeSuggestion('好的，暂不选中这首歌。');
      return true;
    }
    if (pendingMessage) {
      await finishMessageDecision('cancel_message', 'cancel', 'hand');
      return true;
    }
    if (canProvideMusicPreference()) {
      await dislikeCurrentTrack('hand');
      return true;
    }
  }
  if (gesture === 'Palm_Toggle' && activeMode && currentTrack) {
    await recordHandGesture('toggle_playback', 'Palm_Toggle', 'music_playback', 'music', confidence);
    toggleDemoPlayback();
    return true;
  }
  if (gesture === 'Palm_Wave' && canControlSelectedMusicTrack()) {
    await recordHandGesture('skip_track', 'Palm_Wave', 'music_skip', 'music', confidence);
    await nextTrack();
    return true;
  }
  return false;
}

function pointDistance(left, right) {
  return Math.hypot(left.x - right.x, left.y - right.y);
}

function isLikelyOpenPalm(landmarks) {
  if (!landmarks?.[0] || !landmarks?.[5] || !landmarks?.[17]) return false;
  const wrist = landmarks[0];
  const extendedFingerPairs = [[8, 6], [12, 10], [16, 14], [20, 18]];
  const extendedCount = extendedFingerPairs.filter(([tip, joint]) => (
    pointDistance(wrist, landmarks[tip]) > pointDistance(wrist, landmarks[joint]) * 1.18
  )).length;
  const palmWidth = pointDistance(landmarks[5], landmarks[17]);
  const fingertipSpread = pointDistance(landmarks[8], landmarks[20]);
  return extendedCount >= 3 && palmWidth > 0.02 && fingertipSpread > palmWidth * 1.05;
}

function detectPalmWave(landmarks, now, waveEligible) {
  if (!waveEligible || !landmarks?.[0] || !landmarks?.[5] || !landmarks?.[17]) {
    handMotionHistory = [];
    return false;
  }
  const center = {
    x: (landmarks[0].x + landmarks[5].x + landmarks[17].x) / 3,
    y: (landmarks[0].y + landmarks[5].y + landmarks[17].y) / 3,
    timestamp: now,
  };
  handMotionHistory.push(center);
  handMotionHistory = handMotionHistory.filter((sample) => now - sample.timestamp <= HAND_WAVE_WINDOW_MS);
  if (handMotionHistory.length < 3) return false;
  const xs = handMotionHistory.map((sample) => sample.x);
  const ys = handMotionHistory.map((sample) => sample.y);
  const horizontalSpan = Math.max(...xs) - Math.min(...xs);
  const verticalSpan = Math.max(...ys) - Math.min(...ys);
  const totalSpan = Math.hypot(horizontalSpan, verticalSpan);
  // 允许斜向挥手：需含有一定横向分量并形成足够的整体移动，不再要求近似纯水平。
  return horizontalSpan >= HAND_WAVE_MIN_HORIZONTAL_SPAN
    && totalSpan >= HAND_WAVE_MIN_TOTAL_SPAN
    && verticalSpan <= HAND_WAVE_MAX_VERTICAL_SPAN;
}

function updatePalmToggleState(palmState, now) {
  if (!palmState) {
    // 开合动作中分类器常会短暂给出“无手势”；在有限窗口内保留前一个端点。
    if (stablePalmPose && now - stablePalmPoseAt > PALM_TOGGLE_TRANSITION_WINDOW_MS) {
      stablePalmPose = undefined;
      stablePalmPoseAt = 0;
    }
    return false;
  }
  if (!stablePalmPose) {
    stablePalmPose = palmState;
    stablePalmPoseAt = now;
    return false;
  }
  if (stablePalmPose === palmState) {
    stablePalmPoseAt = now;
    return false;
  }
  const isTransition = now - stablePalmPoseAt <= PALM_TOGGLE_TRANSITION_WINDOW_MS;
  stablePalmPose = palmState;
  stablePalmPoseAt = now;
  return isTransition;
}

function observeHandGesture(result, now) {
  const landmarks = result.handLandmarks?.[0];
  const category = result.gestures?.[0]?.[0];
  let gesture = category?.categoryName;
  const confidence = category?.score || 0;
  const openPalm = gesture === 'Open_Palm' || isLikelyOpenPalm(landmarks);
  if (openPalm) handOpenSeenUntil = now + 950;
  const palmState = openPalm ? 'open' : gesture === 'Closed_Fist' && confidence >= 0.50 ? 'closed' : undefined;
  const palmToggle = updatePalmToggleState(palmState, now);
  // 挥动时分类器偶尔会短暂丢失“张开手掌”，仍在最近识别到张开的短窗口内追踪其轨迹。
  const waveEligible = Boolean(landmarks && now <= handOpenSeenUntil);
  if (palmToggle) {
    handGestureCandidate = undefined;
    handMotionHistory = [];
    handGestureSuppressHeadUntil = Math.max(handGestureSuppressHeadUntil, now + HAND_GESTURE_HEAD_SUPPRESS_MS);
    if (now - lastHandGestureAt < HAND_GESTURE_COOLDOWN_MS) return;
    lastHandGestureAt = now;
    void applyHandGesture('Palm_Toggle', 0.8).then((handled) => {
      setHandGestureStatus(handled
        ? '已通过手掌开合处理。'
        : '已识别手掌开合；当前没有可控制的音乐。');
    });
    return;
  } else if (detectPalmWave(landmarks, now, waveEligible)) {
    handGestureCandidate = undefined;
    handMotionHistory = [];
    handGestureSuppressHeadUntil = Math.max(handGestureSuppressHeadUntil, now + HAND_GESTURE_HEAD_SUPPRESS_MS);
    if (now - lastHandGestureAt < HAND_GESTURE_COOLDOWN_MS) return;
    lastHandGestureAt = now;
    void applyHandGesture('Palm_Wave', 0.8).then((handled) => {
      setHandGestureStatus(handled
        ? '已通过斜向 / 横向挥手切换下一首。'
        : '已识别挥手；请先选中音乐播放卡片。');
    });
    return;
  } else if (palmState) {
    handGestureCandidate = undefined;
    handGestureSuppressHeadUntil = Math.max(handGestureSuppressHeadUntil, now + HAND_GESTURE_HEAD_SUPPRESS_MS);
    setHandGestureStatus(palmState === 'open'
      ? '已识别张开手掌：原位握拳可暂停 / 继续；横向或斜向挥动可切下一首。'
      : '已识别握拳：原位张开手掌可暂停 / 继续。');
    return;
  }
  if (!['Thumb_Up', 'Thumb_Down'].includes(gesture) || confidence < 0.72) {
    if (handGestureCandidate && now - handGestureCandidate.lastSeen <= HAND_GESTURE_CANDIDATE_GAP_MS) return;
    handGestureCandidate = undefined;
    return;
  }
  handGestureSuppressHeadUntil = Math.max(handGestureSuppressHeadUntil, now + HAND_GESTURE_HEAD_SUPPRESS_MS);
  if (!handGestureCandidate || handGestureCandidate.gesture !== gesture) {
    handGestureCandidate = { gesture, confidence, since: now, lastSeen: now };
    const label = gesture === 'Thumb_Up' ? '点赞' : '踩';
    setHandGestureStatus(`检测到${label}，请保持片刻确认。`);
    return;
  }
  handGestureCandidate.confidence = Math.min(handGestureCandidate.confidence, confidence);
  handGestureCandidate.lastSeen = now;
  if (now - handGestureCandidate.since < HAND_GESTURE_DWELL_MS || now - lastHandGestureAt < HAND_GESTURE_COOLDOWN_MS) return;
  const stableGesture = handGestureCandidate;
  handGestureCandidate = undefined;
  lastHandGestureAt = now;
  void applyHandGesture(stableGesture.gesture, stableGesture.confidence).then((handled) => {
    setHandGestureStatus(handled
      ? `已通过手势处理：${stableGesture.gesture}。`
      : '已识别手势；当前没有可确认的操作。');
  });
}

function runHandGestureDetection(video, now) {
  if (!handGesturesEnabled || !handGestureRecognizer || now - lastHandDetectionAt < HAND_GESTURE_INTERVAL_MS) return;
  lastHandDetectionAt = now;
  try {
    observeHandGesture(handGestureRecognizer.recognizeForVideo(video, now), now);
  } catch (error) {
    handGesturesEnabled = false;
    updateCameraControls(Boolean(cameraStream));
    setHandGestureStatus(`手势检测已暂停：${error.message || '未知错误'}`);
  }
}

function updateFacePresence(hasFace) {
  if (lastFacePresence === hasFace) return;
  lastFacePresence = hasFace;
  if (hasFace) {
    setCameraStatus('已检测到人脸与眼部关键点（本机处理）', 'active');
  } else {
    setCameraStatus('未检测到人脸，请正对屏幕并保持光线充足', 'waiting');
  }
}

function runFaceDetection() {
  const video = $('#camera-preview');
  if (!cameraStream || !faceLandmarker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  const now = performance.now();
  // 先处理手势并写入短暂的头部动作抑制标记，再处理人脸，避免挥手同帧被判成摇头。
  runHandGestureDetection(video, now);
  if (now - lastFaceDetectionAt >= 120) {
    lastFaceDetectionAt = now;
    try {
      const result = faceLandmarker.detectForVideo(video, now);
      const landmarks = result.faceLandmarks?.[0];
      if (landmarks) {
        drawFaceLandmarks(landmarks);
        observeHeadGesture(landmarks);
        latestEyeFeatures = extractEyeFeatures(landmarks);
        if (latestEyeFeatures) {
          eyeFeatureHistory.push([...latestEyeFeatures]);
          eyeFeatureHistory = eyeFeatureHistory.slice(-24);
          if (gazeMapper) updateGazeTarget(predictGazePoint(latestEyeFeatures));
        } else clearGazeTarget();
      } else {
        latestEyeFeatures = undefined;
        eyeFeatureHistory = [];
        clearFaceOverlay();
        clearGazeTarget();
      }
      updateFacePresence(Boolean(landmarks));
    } catch (error) {
      setCameraStatus(`人脸关键点检测暂时不可用：${error.message || '未知错误'}`, 'error');
      return;
    }
  }
  faceDetectionFrame = requestAnimationFrame(runFaceDetection);
}

async function startFaceDetection() {
  try {
    setCameraStatus('摄像头已开启，正在加载本机人脸关键点模型…', 'waiting');
    await initializeFaceLandmarker();
    if (!cameraStream) return;
    lastFaceDetectionAt = 0;
    lastFacePresence = undefined;
    cancelAnimationFrame(faceDetectionFrame);
    faceDetectionFrame = requestAnimationFrame(runFaceDetection);
  } catch (error) {
    setCameraStatus(`摄像头预览正常，但人脸模型加载失败：${error.message || '请检查网络后重试'}`, 'error');
  }
}

function cameraErrorMessage(error) {
  if (error.name === 'NotAllowedError' || error.name === 'SecurityError') {
    return '未获得摄像头权限。请在浏览器地址栏的权限设置中允许摄像头后重试。';
  }
  if (error.name === 'NotFoundError') return '未检测到可用摄像头。请检查设备是否已连接。';
  if (error.name === 'NotReadableError') return '摄像头正被其他应用占用。请关闭会议软件或相机应用后重试。';
  return `无法开启摄像头：${error.message || '发生未知错误。'}`;
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setCameraStatus('当前浏览器不支持摄像头访问。请使用最新版 Chrome、Edge 或 Firefox。', 'error');
    return;
  }
  try {
    setCameraStatus('正在请求本机摄像头权限…', 'waiting');
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
    });
    cameraStream = stream;
    const video = $('#camera-preview');
    video.srcObject = stream;
    await video.play();
    updateCameraControls(true);
    startFaceDetection();
    void enableHandGestures();
    const [track] = stream.getVideoTracks();
    track?.addEventListener('ended', () => {
      if (cameraStream === stream) stopCamera(false);
    }, { once: true });
  } catch (error) {
    cameraStream = undefined;
    updateCameraControls(false);
    setCameraStatus(cameraErrorMessage(error), 'error');
  }
}

function stopCamera(showMessage = true) {
  if (calibrationActive) finishCalibration(false, '摄像头已关闭，已取消本次视线校准。');
  cancelAnimationFrame(faceDetectionFrame);
  faceDetectionFrame = undefined;
  lastFacePresence = undefined;
  handGesturesEnabled = false;
  handGestureCandidate = undefined;
  handMotionHistory = [];
  handOpenSeenUntil = 0;
  handGestureSuppressHeadUntil = 0;
  stablePalmPose = undefined;
  stablePalmPoseAt = 0;
  lastHandDetectionAt = 0;
  latestEyeFeatures = undefined;
  eyeFeatureHistory = [];
  clearGazeTarget();
  clearFaceOverlay();
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = undefined;
  const video = $('#camera-preview');
  video.srcObject = null;
  updateCameraControls(false);
  setHandGestureStatus('手势识别未开启。');
  setCameraStatus('摄像头已关闭。', 'idle');
  if (showMessage) toast('摄像头已关闭，本机画面已停止。');
}

function renderContacts() {
  $('#contacts').innerHTML = data.contacts.map((contact) => `
    <button class="card contact" data-id="${contact.id}">
      <strong>${contact.name}</strong><small>${contact.relationship}${contact.frequent ? ' · 常用联系人' : ''}</small>
    </button>`).join('');
  document.querySelectorAll('.contact').forEach((node) => node.addEventListener('click', async () => {
    clearGazeSuggestion();
    clearGazeTarget();
    lastLockedGaze = undefined;
    selectedContactId = node.dataset.id;
    selectedContactSource = 'manual';
    try {
      await api('select_contact', { contact_id: node.dataset.id });
      document.querySelectorAll('.contact').forEach((item) => item.classList.remove('selected'));
      node.classList.add('selected');
      toast(`已模拟注视：${node.querySelector('strong').textContent}`);
    } catch (error) {
      selectedContactId = undefined;
      selectedContactSource = undefined;
      toast(error.message);
    }
  }));
}

const MODE_LABELS = { general: '热门推荐', focus: '专注模式', driving: '开车模式', entertainment: '娱乐模式' };

function renderModes() {
  const modes = [
    ['focus', '专注模式', '学习、阅读和需要低干扰的任务'],
    ['driving', '开车模式', '通勤、驾车或需要保持精神的旅程'],
    ['entertainment', '娱乐模式', '休闲、聚会和放松时刻'],
  ];
  $('#modes').innerHTML = modes.map(([id, name, note]) => `
    <button class="card mode" data-mode="${id}"><strong>${name}</strong><small>${note}</small></button>`).join('');
  document.querySelectorAll('.mode').forEach((node) => node.addEventListener('click', () => requestMode(node.dataset.mode)));
  updateModeStatus();
}

function updateModeStatus() {
  document.querySelectorAll('.mode').forEach((node) => node.classList.toggle('selected', node.dataset.mode === activeMode));
  if (!activeMode) {
    $('#mode-status').textContent = '当前未启用音乐模式，将为你推荐近期热门歌曲。';
    return;
  }
  $('#mode-status').innerHTML = `当前正在使用<strong>${MODE_LABELS[activeMode]}</strong>。<button class="secondary" id="stop-mode">停止当前模式</button><button class="secondary" id="switch-mode">切换模式</button>`;
  $('#stop-mode').onclick = stopMode;
  $('#switch-mode').onclick = () => {
    switchArmed = true;
    $('#mode-decision').textContent = '请从上方选择想切换到的模式。';
  };
}

async function requestMode(nextMode) {
  if (activeMode && activeMode !== nextMode) {
    // 从未指定状态的热门推荐进入某一模式，是一次明确选择，不需要再确认。
    if (activeMode === 'general') {
      await activateMode(nextMode);
      return;
    }
    if (switchArmed) {
      await activateMode(nextMode);
      return;
    }
    $('#mode-decision').innerHTML = `当前为${MODE_LABELS[activeMode]}。你可以先停止，或确认切换至${MODE_LABELS[nextMode]}。<p><button class="secondary" id="decision-stop">停止当前模式</button><button class="primary" id="decision-switch">确认切换</button></p>`;
    $('#decision-stop').onclick = stopMode;
    $('#decision-switch').onclick = () => activateMode(nextMode);
    return;
  }
  if (!activeMode) await activateMode(nextMode);
}

async function activateMode(mode) {
  try {
    stopDemoPlayback();
    clearMusicTrackSelection();
    const result = await api('start_mode', { mode, mode_label: MODE_LABELS[mode] });
    activeMode = mode;
    currentTrack = result.track;
    currentRecommendationReason = result.recommendation_reason;
    currentPreferencePlaylist = result.preference_playlist || [];
    musicGazeTrackId = undefined;
    musicGazeGestureWindowUntil = 0;
    musicFeedbackLockedTrackId = undefined;
    switchArmed = false;
    $('#mode-decision').innerHTML = '';
    updateModeStatus();
    renderNowPlaying(result.message);
    toast(result.message);
  } catch (error) { toast(error.message); }
}

async function activateGeneralMusic() {
  try {
    stopDemoPlayback();
    clearMusicTrackSelection();
    const result = await api('start_general_music');
    activeMode = 'general';
    currentTrack = result.track;
    currentRecommendationReason = result.recommendation_reason;
    currentPreferencePlaylist = result.preference_playlist || [];
    musicGazeTrackId = undefined;
    musicGazeGestureWindowUntil = 0;
    musicFeedbackLockedTrackId = undefined;
    switchArmed = false;
    $('#mode-decision').innerHTML = '';
    updateModeStatus();
    renderNowPlaying(result.message);
    toast(result.message);
  } catch (error) { toast(error.message); }
}

async function stopMode() {
  try {
    stopDemoPlayback();
    clearMusicTrackSelection();
    const result = await api('stop_mode');
    activeMode = undefined;
    musicGazeTrackId = undefined;
    musicGazeGestureWindowUntil = 0;
    musicFeedbackLockedTrackId = undefined;
    switchArmed = false;
    $('#mode-decision').innerHTML = '';
    updateModeStatus();
    toast(result.message);
  } catch (error) { toast(error.message); }
}

function renderNowPlaying(message) {
  if (!currentTrack) return;
  const playlist = currentPreferencePlaylist.length
    ? currentPreferencePlaylist.map((track) => track.title).join('、')
    : '尚未形成偏好歌单；会同时参考当前状态下的平台大众常听。';
  const playlistLabel = activeMode === 'general' ? '通用偏好歌单' : '当前模式偏好歌单';
  const feedbackHint = currentTrack.id === musicFeedbackLockedTrackId
    ? '已记录你喜欢这首歌；播放结束或切到下一首后再询问你的偏好。'
    : '持续注视歌曲播放卡片约 0.55 秒后，系统会询问是否选中；确认后，即使切歌也保持选中，直到说“取消当前歌曲选择”。开启手势后：原位开合手掌可暂停 / 继续，横向或斜向挥动张开的手掌可切下一首。';
  $('#music-result').innerHTML = `<div class="result-box music-result-box"><article class="music-track-card${musicCardSelected ? ' music-selected' : ''}" data-track-id="${currentTrack.id}"><span>正在播放</span><strong>${currentTrack.title}</strong><small>推荐依据：${currentRecommendationReason || '与当前模式匹配'}</small></article><p>${message}</p><p class="playback-progress" id="playback-progress">正在启动本地演示播放…</p><p class="music-hint">${feedbackHint}</p><button class="secondary" id="toggle-playback">${playbackPaused ? '继续播放' : '暂停播放'}</button><button class="secondary" id="like-track" ${currentTrack.id === musicFeedbackLockedTrackId ? 'disabled' : ''}>我喜欢这首</button><button class="secondary" id="dislike-track">不喜欢这首</button><button class="secondary" id="skip-track">下一首</button><p class="playlist-summary"><strong>${playlistLabel}：</strong>${playlist}</p></div>`;
  $('#toggle-playback').onclick = toggleDemoPlayback;
  $('#like-track').onclick = likeCurrentTrack;
  $('#dislike-track').onclick = dislikeCurrentTrack;
  $('#skip-track').onclick = nextTrack;
  startDemoPlayback();
  void recordScreenContext();
}

function stopDemoPlayback() {
  clearInterval(playbackTimer);
  playbackTimer = undefined;
  playbackDeadline = 0;
  playbackTrackId = undefined;
  playbackPaused = false;
  pausedPlaybackRemainingMs = 0;
}

function updatePlaybackProgress() {
  const node = $('#playback-progress');
  if (!node || (!playbackDeadline && !playbackPaused)) return;
  const remaining = playbackPaused ? pausedPlaybackRemainingMs : playbackDeadline - Date.now();
  const seconds = Math.max(0, Math.ceil(remaining / 1000));
  if (playbackPaused) {
    node.textContent = `本地演示已暂停：剩余 ${seconds} 秒；继续播放后才会自动记录完整收听。`;
    return;
  }
  node.textContent = `本地演示播放中：剩余 ${seconds} 秒；播放结束后系统会自动记录完整收听。`;
}

function toggleDemoPlayback() {
  if (!currentTrack || !activeMode) return;
  if (playbackPaused) {
    playbackPaused = false;
    playbackDeadline = Date.now() + pausedPlaybackRemainingMs;
    pausedPlaybackRemainingMs = 0;
    $('#toggle-playback').textContent = '暂停播放';
    startDemoPlayback();
    toast('已继续播放。');
    return;
  }
  pausedPlaybackRemainingMs = Math.max(0, playbackDeadline - Date.now());
  clearInterval(playbackTimer);
  playbackTimer = undefined;
  playbackPaused = true;
  $('#toggle-playback').textContent = '继续播放';
  updatePlaybackProgress();
  toast('已暂停播放。');
}

function startDemoPlayback() {
  if (!currentTrack || !activeMode) return;
  if (playbackTrackId !== currentTrack.id) {
    stopDemoPlayback();
    playbackTrackId = currentTrack.id;
    playbackFeedbackRecorded = false;
    playbackDeadline = Date.now() + DEMO_TRACK_DURATION_SECONDS * 1000;
  }
  updatePlaybackProgress();
  if (playbackPaused) return;
  if (playbackTimer) return;
  playbackTimer = window.setInterval(() => {
    if (!currentTrack || playbackTrackId !== currentTrack.id) return;
    updatePlaybackProgress();
    if (Date.now() < playbackDeadline) return;
    clearInterval(playbackTimer);
    playbackTimer = undefined;
    if (playbackFeedbackRecorded) {
      void advanceAfterPlayback();
    } else {
      void completeCurrentTrack();
    }
  }, 500);
}

async function likeCurrentTrack(source = 'button') {
  try {
    if (!currentTrack) return;
    if (source === 'head') await recordHeadDecision('confirm', 'music_feedback', 'music');
    if (source === 'hand') await recordHandGesture('confirm', 'Thumb_Up', 'music_feedback', 'music');
    const result = await api('like_track', { track_id: currentTrack.id });
    playbackFeedbackRecorded = true;
    currentPreferencePlaylist = result.preference_playlist || currentPreferencePlaylist;
    musicGazeTrackId = undefined;
    musicFeedbackLockedTrackId = currentTrack.id;
    clearGazeTarget();
    renderNowPlaying(result.message);
    toast(result.message);
  } catch (error) { toast(error.message); }
}

async function completeCurrentTrack() {
  try {
    if (!currentTrack) return;
    stopDemoPlayback();
    const result = await api('complete_track', { track_id: currentTrack.id });
    currentTrack = result.track;
    currentRecommendationReason = result.recommendation_reason;
    currentPreferencePlaylist = result.preference_playlist || currentPreferencePlaylist;
    musicGazeTrackId = undefined;
    musicFeedbackLockedTrackId = undefined;
    clearGazeTarget();
    renderNowPlaying(result.message);
    toast(result.message);
  } catch (error) { toast(error.message); }
}

async function advanceAfterPlayback() {
  try {
    if (!currentTrack) return;
    const result = await api('advance_track', { current_track_id: currentTrack.id });
    currentTrack = result.track;
    currentRecommendationReason = result.recommendation_reason;
    currentPreferencePlaylist = result.preference_playlist || currentPreferencePlaylist;
    musicGazeTrackId = undefined;
    musicFeedbackLockedTrackId = undefined;
    clearGazeTarget();
    renderNowPlaying(result.message);
  } catch (error) { toast(error.message); }
}

async function dislikeCurrentTrack(source = 'button') {
  try {
    if (!currentTrack) return;
    stopDemoPlayback();
    if (source === 'head') await recordHeadDecision('reject', 'music_feedback', 'music');
    if (source === 'hand') await recordHandGesture('reject', 'Thumb_Down', 'music_feedback', 'music');
    const result = await api('dislike_track', { current_track_id: currentTrack.id });
    currentTrack = result.track;
    currentRecommendationReason = result.recommendation_reason;
    currentPreferencePlaylist = result.preference_playlist || currentPreferencePlaylist;
    musicGazeTrackId = undefined;
    musicFeedbackLockedTrackId = undefined;
    clearGazeTarget();
    renderNowPlaying(result.message);
    toast(result.message);
  } catch (error) { toast(error.message); }
}

async function nextTrack() {
  try {
    if (!currentTrack) return;
    stopDemoPlayback();
    const result = await api('next_track', { current_track_id: currentTrack.id });
    currentTrack = result.track;
    currentRecommendationReason = result.recommendation_reason;
    currentPreferencePlaylist = result.preference_playlist || currentPreferencePlaylist;
    musicGazeTrackId = undefined;
    musicFeedbackLockedTrackId = undefined;
    clearGazeTarget();
    renderNowPlaying(result.message);
    toast(result.message);
  } catch (error) { toast(error.message); }
}

function showSchedule(result) {
  if (!result.items.length) {
    $('#memo-result').innerHTML = '<div class="result-box">已授权文件中没有可识别的日程事项。</div>';
    return;
  }
  const items = result.items.map((item) => `
    <article class="schedule-item${item.is_past ? ' past' : ''}${item.is_completed ? ' completed' : ''}">
      <label class="schedule-check">
        <input type="checkbox" data-event-key="${item.event_key}" ${item.is_completed ? 'checked' : ''} />
        <span class="schedule-time">${item.date} ${item.time}</span>
        <span class="schedule-title">${item.title}</span>
        <span class="schedule-meta">${item.location} · ${item.priority} 优先级${item.is_past ? ' · 已过时间' : ''}</span>
        <small>${item.content || '无补充说明'}</small>
      </label>
    </article>`).join('');
  $('#memo-result').innerHTML = `<div class="result-box schedule-box"><div class="schedule-summary"><strong>${result.title || '全部日程'}</strong><span>共 ${result.total} 项 · 已完成 ${result.completed} 项 · 已过时间 ${result.past} 项</span></div>${items}</div>`;
  document.querySelectorAll('[data-event-key]').forEach((checkbox) => {
    checkbox.addEventListener('change', async () => {
      try {
        await api('toggle_event_completion', { event_key: checkbox.dataset.eventKey, completed: checkbox.checked });
        showSchedule(await api('query_schedule'));
      } catch (error) {
        toast(error.message);
        checkbox.checked = !checkbox.checked;
      }
    });
  });
  void recordScreenContext();
}

function readTextFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const bytes = new Uint8Array(reader.result);
      let content;
      try {
        content = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
      } catch {
        content = new TextDecoder('gb18030').decode(bytes);
      }
      resolve({ file_name: file.name, file_content: content });
    };
    reader.onerror = () => reject(new Error(`无法读取文件：${file.name}`));
    reader.readAsArrayBuffer(file);
  });
}

async function submitScheduleSimulatedSpeech() {
  const text = $('#schedule-content').value.trim();
  if (!text) {
    toast('请输入或选择一条日程查询语句。');
    return;
  }
  const timestamp = Date.now();
  await recordBrowserEvent({
    modality: 'speech_text',
    timestamp_ms: timestamp,
    confidence: 1,
    payload: { text, page: 'memo', source: 'simulated' },
  });
  try {
    const result = await api('understand_multimodal_command', { speech_timestamp_ms: timestamp });
    if (!result.items) {
      $('#memo-result').innerHTML = `<div class="result-box"><strong>${result.message}</strong></div>`;
      return;
    }
    showSchedule(result);
  } catch (error) { toast(error.message); }
}

async function authorizeSelectedMemos() {
  const input = $('#memo-file');
  const files = [...input.files];
  if (!files.length) return;
  try {
    const selectedFiles = await Promise.all(files.map(readTextFile));
    const result = await api('authorize_memo_files', { files: selectedFiles, authorization_mode: memoAuthorizationMode });
    renderAuthorizedSources(result.sources, memoAuthorizationMode === 'merge' ? '已合并并更新本次选择的文件。' : '已替换授权文件列表。');
    input.value = '';
    $('#memo-result').innerHTML = '';
    toast('备忘录授权成功。');
  } catch (error) {
    input.value = '';
    toast(error.message);
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[character]));
}

function renderAuthorizedSources(sources, note = '') {
  authorizedSources = sources || [];
  if (!authorizedSources.length) {
    $('#authorization-status').textContent = '尚未选择授权备忘录文件。';
    $('#authorized-files').innerHTML = '';
    return;
  }
  const names = authorizedSources.map((source) => `${source.display_name}（${source.item_count ?? '已读取'} 项）`).join('、');
  $('#authorization-status').textContent = `当前已授权 ${authorizedSources.length} 个文件：${names}（仅保存本地授权副本）${note ? ` ${note}` : ''}`;
  $('#authorized-files').innerHTML = authorizedSources.map((source) => `<div class="authorized-file"><span><strong>${escapeHtml(source.display_name)}</strong> <small>${source.item_count ?? '—'} 项</small></span><button class="secondary revoke-memo" data-stored-name="${escapeHtml(source.stored_name)}">取消授权</button></div>`).join('');
  document.querySelectorAll('.revoke-memo').forEach((button) => {
    button.onclick = async () => {
      try {
        const result = await api('revoke_memo_file', { stored_name: button.dataset.storedName });
        renderAuthorizedSources(result.sources, result.message);
        $('#memo-result').innerHTML = '';
        toast(result.message);
      } catch (error) { toast(error.message); }
    };
  });
}

function openMemoPicker(mode) {
  memoAuthorizationMode = mode;
  $('#memo-file').click();
}

function bindEvents() {
  document.querySelectorAll('.nav-button').forEach((node) => node.addEventListener('click', async () => {
    if (node.dataset.page !== 'music' && activeMode) await stopMode();
    document.querySelectorAll('.nav-button, .page').forEach((item) => item.classList.remove('active'));
    node.classList.add('active');
    $(`#${node.dataset.page}-page`).classList.add('active');
    setCameraContext(node.textContent.trim());
    void recordScreenContext();
    if (node.dataset.page === 'music' && !activeMode) await activateGeneralMusic();
  }));

  $('#prepare-message').onclick = submitSimulatedSpeech;
  $('#refresh-multimodal-events').onclick = refreshMultimodalInspector;
  $('#multimodal-inspector').addEventListener('toggle', (event) => {
    if (event.currentTarget.open) void refreshMultimodalInspector();
  });
  document.querySelectorAll('.speech-preset').forEach((node) => node.onclick = () => {
    $('#message-content').value = node.dataset.text;
    $('#message-content').focus();
  });
  $('#prepare-music-command').onclick = submitSimulatedMusicCommand;
  document.querySelectorAll('.music-speech-preset').forEach((node) => node.onclick = () => {
    $('#music-command').value = node.dataset.text;
    $('#music-command').focus();
  });
  $('#start-camera').onclick = startCamera;
  $('#calibrate-gaze').onclick = startGazeCalibration;
  $('#clear-gaze-calibration').onclick = clearSavedGazeCalibration;
  $('#stop-camera').onclick = () => stopCamera();
  $('#cancel-calibration').onclick = () => finishCalibration(false, '已取消视线校准。');

  $('#choose-memo').onclick = () => openMemoPicker('merge');
  $('#sync-memo').onclick = () => openMemoPicker('merge');
  $('#memo-file').onchange = authorizeSelectedMemos;
  $('#prepare-schedule').onclick = submitScheduleSimulatedSpeech;
  document.querySelectorAll('.schedule-speech-preset').forEach((node) => node.onclick = () => {
    $('#schedule-content').value = node.dataset.text;
    $('#schedule-content').focus();
  });
  $('#query-schedule').onclick = async () => {
    try { showSchedule(await api('query_schedule')); }
    catch (error) { toast(error.message); }
  };

  // ----- 消息页面麦克风 -----
  const voiceMessageBtn = document.getElementById('voice-message-btn');
  if (voiceMessageBtn) {
    voiceMessageBtn.addEventListener('click', () => {
      const input = document.getElementById('message-content');
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) {
        toast('当前浏览器不支持语音识别，请使用 Chrome 或 Edge。');
        return;
      }
      if (input.dataset.listening === 'true') {
        toast('正在录音，请稍候…');
        return;
      }

      const recognition = new SpeechRecognition();
      recognition.lang = 'zh-CN';
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;

      const micBtn = voiceMessageBtn;
      input.dataset.listening = 'true';
      micBtn.classList.add('listening');
      const originalPlaceholder = input.placeholder;
      input.placeholder = '🎤 正在倾听（语音直接后台处理）…';
      toast('🎤 请说话…');

      recognition.onresult = async (event) => {
        // 只取最终识别结果
        let finalTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          }
        }
        if (finalTranscript) {
          //识别到文字后，不显示在输入框，直接走后端处理
          const timestamp = Date.now();
          try {
            // 1. 先记录多模态事件（与模拟语音共用一套事件系统）
            await recordBrowserEvent({
              modality: 'speech_text',
              timestamp_ms: timestamp,
              confidence: 1,
              payload: { text: finalTranscript.trim(), page: 'message', source: 'realtime_mic' }
            });
            // 2. 直接调用后端理解接口
            const result = await api('understand_multimodal_command', {
              speech_timestamp_ms: timestamp,
              preferred_contact_id: selectedContactSource === 'manual' ? selectedContactId : undefined
            });
            // 3. 处理返回结果（复用现有渲染逻辑）
            if (result.clear_message_form) {
              if (result.intent === 'confirm') adjustGazeReliability('success');
              if (result.intent === 'cancel') adjustGazeReliability('cancel');
              resetMessageForm(result.message);
            } else {
              renderMessageUnderstanding(result);
            }
          } catch (error) {
            toast(error.message);
          } finally {
            recognition.stop();
          }
        }
      };

      recognition.onend = () => {
        input.dataset.listening = 'false';
        micBtn.classList.remove('listening');
        input.placeholder = originalPlaceholder;
        // 如果没有任何识别结果，且没有报错，给一个温和提示
        if (!input.dataset._hasResult) {
          // 防止重复提示，但这里不做额外处理，因为可能用户只是取消了。
        }
        input.dataset._hasResult = false;
      };

      recognition.onerror = (event) => {
        console.warn('语音识别错误：', event.error);
        let msg = '语音识别失败';
        if (event.error === 'not-allowed') msg = '请允许浏览器使用麦克风权限。';
        else if (event.error === 'no-speech') msg = '未检测到语音，请对着麦克风说话。';
        else if (event.error === 'audio-capture') msg = '无法访问麦克风，请检查设备连接。';
        toast(msg);
        recognition.stop();
      };

      try {
        recognition.start();
        input.dataset._hasResult = false; // 重置标记
      } catch (error) {
        input.dataset.listening = 'false';
        micBtn.classList.remove('listening');
        toast('语音识别启动失败，请刷新页面重试。');
      }
    });
  }

  // ----- 音乐页面麦克风 -----
  const voiceMusicBtn = document.getElementById('voice-music-btn');
  if (voiceMusicBtn) {
    voiceMusicBtn.addEventListener('click', () => {
      const input = document.getElementById('music-command');
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) {
        toast('当前浏览器不支持语音识别，请使用 Chrome 或 Edge。');
        return;
      }
      if (input.dataset.listening === 'true') {
        toast('正在录音，请稍候…');
        return;
      }

      const recognition = new SpeechRecognition();
      recognition.lang = 'zh-CN';
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;

      const micBtn = voiceMusicBtn;
      input.dataset.listening = 'true';
      micBtn.classList.add('listening');
      const originalPlaceholder = input.placeholder;
      input.placeholder = '🎤 正在倾听（语音直接后台处理）…';
      toast('🎤 请说话…');

      recognition.onresult = async (event) => {
        let finalTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          }
        }
        if (finalTranscript) {
          //不显示在输入框，直接走后端
          const timestamp = Date.now();
          try {
            await recordBrowserEvent({
              modality: 'speech_text',
              timestamp_ms: timestamp,
              confidence: 1,
              payload: { text: finalTranscript.trim(), page: 'music', source: 'realtime_mic' }
            });
            const result = await api('understand_multimodal_command', { speech_timestamp_ms: timestamp });
            // 处理音乐指令结果（复用现有逻辑）
            if (result.intent === 'cancel_music_selection') {
              clearMusicTrackSelection(result.message);
              toast(result.message);
            } else if (result.intent === 'next_track') {
              await nextTrack();
            } else {
              toast(result.message);
            }
          } catch (error) {
            toast(error.message);
          } finally {
            recognition.stop();
          }
        }
      };

      recognition.onend = () => {
        input.dataset.listening = 'false';
        micBtn.classList.remove('listening');
        input.placeholder = originalPlaceholder;
      };

      recognition.onerror = (event) => {
        console.warn('语音识别错误：', event.error);
        let msg = '语音识别失败';
        if (event.error === 'not-allowed') msg = '请允许浏览器使用麦克风权限。';
        else if (event.error === 'no-speech') msg = '未检测到语音，请对着麦克风说话。';
        else if (event.error === 'audio-capture') msg = '无法访问麦克风，请检查设备连接。';
        toast(msg);
        recognition.stop();
      };

      try {
        recognition.start();
      } catch (error) {
        input.dataset.listening = 'false';
        micBtn.classList.remove('listening');
        toast('语音识别启动失败，请刷新页面重试。');
      }
    });
  }
}

function resetMessageForm(message) {
  pendingMessage = undefined;
  messageDecisionInProgress = false;
  selectedContactId = undefined;
  selectedContactSource = undefined;
  clearGazeSuggestion();
  clearGazeTarget();
  $('#message-content').value = '';
  $('#message-result').innerHTML = '';
  document.querySelectorAll('.contact').forEach((item) => item.classList.remove('selected'));
  toast(message);
}

async function init() {
  data = await (await fetch('/api/bootstrap')).json();
  gazeMapper = loadGazeCalibration();
  $('#profile').innerHTML = data.profiles.map((profile) => `<option value="${profile.id}">${profile.display_name}</option>`).join('');
  if (data.state.authorized_sources?.length) {
    renderAuthorizedSources(data.state.authorized_sources, '已从本机保存的授权记录恢复。文件若已改动，请选择更新后的同名文件并同步。');
  }
  renderContacts();
  activeMode = data.state.active_mode;
  renderModes();
  bindEvents();
  updateCameraControls(false);
  if (gazeMapper) $('#gaze-feedback').textContent = '已加载本机视线校准记录；如更换坐姿、摄像头或屏幕，请重新校准。';
  void recordScreenContext();
}

init().catch((error) => toast(`初始化失败：${error.message}`));

window.addEventListener('pagehide', () => {
  stopCamera(false);
  if (!activeMode) return;
  navigator.sendBeacon('/api/action', JSON.stringify({ action: 'stop_mode_silent' }));
});
