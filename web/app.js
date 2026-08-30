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
// 因离开音乐页而暂停的标志：仅此类暂停在回到音乐页时自动恢复。
let navPausedPlayback = false;
let memoAuthorizationMode = 'merge';
let authorizedSources = [];
let activeProfileId;
let localLatencySamples = [];
// 后端手势画像中的最小触发阈值；每次头部确认/撤销后更新，用于过滤低于该幅度的自然晃动。
let adaptiveHeadMinStrength = 0;

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
// 视线触发操作后的冷却：同一卡片操作后的一段时间内不允许再次锁定（避免误触）。
const GAZE_ACTION_COOLDOWN_MS = 2500;
let gazeActionCooldownKey = null;
let gazeActionCooldownUntil = 0;
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
// 上指/挥手切歌的间隔：避免手指持续上举或连续挥手导致误跳多首。
const POINTING_SKIP_COOLDOWN_MS = 2500;
let lastPointingSkipAt = 0;

const FACE_TASK_VERSION = '1.0.1';
// 模型与主脚本随项目发布，避免 Google Storage 或 CDN 被网络策略拦截后导致功能失效。
const FACE_MODEL_URL = '/models/face_landmarker.task';
const HAND_GESTURE_MODEL_URL = '/models/gesture_recognizer.task';
const FACE_BUNDLE_URL = '/vendor/mediapipe/vision_bundle.mjs';
// 视觉运行时、模型和 WASM 全部随项目本地发布；断网时也可完成端侧推理。
const FACE_WASM_URLS = ['/vendor/mediapipe/wasm'];
const GAZE_CALIBRATION_STORAGE_KEY = 'zhiji.gaze-calibration.v4';
const EYE_LANDMARKS = [33, 133, 159, 145, 160, 158, 153, 144, 362, 263, 386, 374, 385, 387, 373, 380, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477];
const CALIBRATION_POINTS = [
  { x: 0.16, y: 0.24 }, { x: 0.50, y: 0.24 }, { x: 0.84, y: 0.24 },
  { x: 0.16, y: 0.52 }, { x: 0.50, y: 0.52 }, { x: 0.84, y: 0.52 },
  { x: 0.16, y: 0.80 }, { x: 0.50, y: 0.80 }, { x: 0.84, y: 0.80 },
];

const $ = (selector) => document.querySelector(selector);

async function api(action, extra = {}) {
  const startedAt = performance.now();
  const response = await fetch('/api/action', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, ...extra }),
  });
  const payload = await response.json();
  const elapsedMs = performance.now() - startedAt;
  if (action !== 'record_multimodal_event' && action !== 'get_recent_multimodal_events') {
    localLatencySamples.push({ action, elapsedMs });
    localLatencySamples = localLatencySamples.slice(-100);
    updateLatencySummary();
  }
  if (!payload.ok) throw new Error(payload.error);
  return payload.result;
}

function updateLatencySummary() {
  const node = $('#latency-summary');
  if (!node || !localLatencySamples.length) return;
  const values = localLatencySamples.map((item) => item.elapsedMs).sort((left, right) => left - right);
  const percentile = (ratio) => values[Math.min(values.length - 1, Math.ceil(values.length * ratio) - 1)];
  node.textContent = `本页面本轮本地接口延迟：${values.length} 次，P50 ${percentile(0.50).toFixed(0)} ms，P95 ${percentile(0.95).toFixed(0)} ms。该指标不含摄像头采样等待和人工输入时间。`;
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

// ---- 离线语音识别（浏览器端 Whisper 量化模型；音频不出本机）----
let asrPipeline = null;
let asrLoading = null;
let asrRecorder = null; // { stream, context, source, processor, chunks, targetId }

function setAsrStatus(message) {
  const node = $('#asr-status');
  if (node) node.textContent = message;
}

async function getAsrPipeline() {
  if (asrPipeline) return asrPipeline;
  if (asrLoading) return asrLoading;
  asrLoading = (async () => {
    setAsrStatus('正在加载本地语音识别模型（约 45MB，首次约需数十秒）…');
    if (!window.__asr?.pipeline) {
      throw new Error('本地语音识别运行时未加载。请用 Chrome/Edge 并硬刷新（Ctrl+F5）；若仍失败请查看浏览器控制台。');
    }
    const { pipeline, env } = window.__asr;
    env.allowRemoteModels = false;                                    // 禁止访问远端
    env.allowLocalModels = true;                                      // 显式允许本地模型（v3 必需）
    env.localModelPath = '/models/';                                  // 模型随项目本地发布
    env.backends.onnx = env.backends.onnx || {};
    env.backends.onnx.wasm = env.backends.onnx.wasm || {};
    env.backends.onnx.wasm.wasmPaths = '/vendor/transformers/';       // WASM 运行时本地发布
    asrPipeline = await pipeline('automatic-speech-recognition', 'whisper-tiny', { quantized: true });
    setAsrStatus('本地语音识别已就绪（Whisper 量化模型，音频在浏览器内处理，不上传）。');
    return asrPipeline;
  })();
  try {
    return await asrLoading;
  } catch (error) {
    asrLoading = null;
    setAsrStatus(`语音模型加载失败：${error.message || error}`);
    throw error;
  }
}

function setVoiceButtonsRecording(recording) {
  document.querySelectorAll('.voice-input').forEach((button) => {
    button.classList.toggle('recording', recording);
    button.textContent = recording ? '■ 停止录音' : '🎤 开始录音';
  });
  const status = $('#asr-status');
  if (status) status.classList.toggle('recording', recording);
}

// ---- 语音输入：优先浏览器内置 Web Speech（Chrome/Edge 即开即用，参考 Hzp_back_03 设计），
//      不可用或出错时回退本地 Whisper（离线、音频不出本机）。----
let webSpeechSession = null; // { recognizer, targetId, done }

function webSpeechAvailable() {
  return Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);
}

function submitRecognizedText(targetId, text) {
  // 语音识别结果写入消息框：若已有内容则接在后面，不覆盖（便于连续补充再统一确认）。
  const box = $('#voice-result');
  if (box) {
    box.value = box.value.trim() ? `${box.value.trim()}，${text}` : text;
    box.focus();
  }
  setAsrStatus(`已识别：${text}。请确认后点击「确认发送」。`);
  toast(`已识别：${text}`);
}

function confirmVoiceText() {
  const box = $('#voice-result');
  const text = (box?.value || '').trim();
  if (!text) {
    toast('请先语音输入或填写内容。');
    return;
  }
  const activePage = document.querySelector('.page.active')?.id;
  const page = activePage === 'music-page' ? 'music' : activePage === 'memo-page' ? 'memo' : 'message';
  const submit = {
    'message': () => submitSimulatedSpeech('mic', text),
    'music': () => submitSimulatedMusicCommand('mic', text),
    'memo': () => submitScheduleSimulatedSpeech('mic', text),
  }[page];
  if (box) box.value = '';
  setAsrStatus('已发送，正在由本地大模型理解并执行…');
  submit();
}

let webSpeechSeq = 0;
let webSpeechRestartAttempts = 0;

function startWebSpeechRecording(targetId) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognizer = new SpeechRecognition();
  recognizer.lang = 'zh-CN';
  recognizer.continuous = true;              // 持续聆听，直到用户点击「■ 停止录音」
  recognizer.interimResults = true;          // 说话时实时显示中间结果
  recognizer.maxAlternatives = 1;
  const session = { recognizer, targetId, seq: ++webSpeechSeq, accumulated: '', processed: 0 };
  webSpeechSession = session;
  setVoiceButtonsRecording(true);
  setAsrStatus('正在聆听…（说话后点击「■ 停止录音」结束并识别）');
  recognizer.onresult = (event) => {
    if (webSpeechSession !== session) return;
    // 持续聆听：把已确认(final)的段落累加，中间结果仅作实时预览；停止时才一次性提交。
    let interim = '';
    for (let i = session.processed; i < event.results.length; i++) {
      const entry = event.results[i];
      const part = (entry?.[0]?.transcript || '').trim();
      if (entry?.isFinal) {
        session.accumulated = session.accumulated ? `${session.accumulated}，${part}` : part;
        session.processed = i + 1;
      } else {
        interim = part;
      }
    }
    if (interim) {
      setAsrStatus(`正在聆听…「${session.accumulated ? `${session.accumulated}，` : ''}${interim}」`);
    } else if (session.accumulated) {
      setAsrStatus(`已记录：${session.accumulated}。继续说或点「■ 停止录音」。`);
    }
  };
  recognizer.onerror = (event) => {
    if (webSpeechSession !== session) return;
    const error = event.error || 'unknown';
    // 'aborted' 多为停止/快速重开时的良性中断：自动重试一次。
    if (error === 'aborted' && webSpeechRestartAttempts < 2) {
      webSpeechRestartAttempts += 1;
      webSpeechSession = null;
      setVoiceButtonsRecording(false);
      setAsrStatus('语音识别被中断，正在自动重试…');
      window.setTimeout(() => startWebSpeechRecording(session.targetId), 500);
      return;
    }
    webSpeechRestartAttempts = 0;
    webSpeechSession = null;
    setVoiceButtonsRecording(false);
    if (['network', 'service-not-allowed', 'language-not-supported'].includes(error)) {
      setAsrStatus(`浏览器语音识别不可用（${error}），回退本地 Whisper（离线）。`);
      void startAsrRecording(session.targetId);
    } else {
      setAsrStatus(`语音识别未成功：${error}。请再试一次。`);
    }
  };
  recognizer.onend = () => {
    if (webSpeechSession !== session) return;
    webSpeechSession = null;
    setVoiceButtonsRecording(false);
    const text = (session.accumulated || '').trim();
    if (text) {
      setAsrStatus(`已识别：${text}。请确认后点击「确认发送」。`);
      submitRecognizedText(session.targetId, text);
    } else {
      setAsrStatus('未检测到语音。请点击「🎤」后说话，说完点「■ 停止录音」。');
    }
  };
  try {
    recognizer.start();
    webSpeechRestartAttempts = 0;
  } catch (error) {
    if (webSpeechSession !== session) return;
    webSpeechSession = null;
    setVoiceButtonsRecording(false);
    setAsrStatus(`浏览器语音识别不可用：${error.message}；回退本地 Whisper。`);
    void startAsrRecording(targetId);
  }
}

function stopWebSpeech() {
  // 触发 onresult 的最终结果或 onend；不在这里置空会话，交给回调收尾。
  if (!webSpeechSession) return;
  try { webSpeechSession.recognizer.stop(); } catch { /* 已停止 */ }
}

async function startAsrRecording(targetId) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const context = new AudioContext({ sampleRate: 16000 }); // 采样率统一为 16k
  await context.resume();
  const source = context.createMediaStreamSource(stream);
  const processor = context.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  processor.onaudioprocess = (event) => {
    chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  };
  source.connect(processor);
  processor.connect(context.destination);
  asrRecorder = { stream, context, source, processor, chunks, targetId };
  setVoiceButtonsRecording(true);
  setAsrStatus('正在聆听…点击「■ 停止录音」结束并识别。');
}

async function stopAsrRecording() {
  if (!asrRecorder) return;
  const { stream, context, source, processor, chunks, targetId } = asrRecorder;
  asrRecorder = null;
  setVoiceButtonsRecording(false);
  try { source.disconnect(); processor.disconnect(); } catch { /* 已断开 */ }
  stream.getTracks().forEach((track) => track.stop());
  await context.close();
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  if (length < 16000 * 0.3) {
    setAsrStatus('录音太短，未识别。请点击「🎤 语音输入」后至少说 1 秒。');
    return;
  }
  const audio = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    audio.set(chunk, offset);
    offset += chunk.length;
  }
  setAsrStatus('正在本地识别…');
  try {
    const pipe = await getAsrPipeline();
    const output = await pipe(audio, { language: 'zh', task: 'transcribe' });
    const text = (output.text || '').trim();
    if (!text) {
      setAsrStatus('未识别到语音，请再试一次。');
      return;
    }
    setAsrStatus(`已识别：${text}。将提交给本地大模型理解后执行。`);
    toast(`已识别：${text}`);
    submitRecognizedText(targetId, text);
  } catch (error) {
    setAsrStatus(`识别失败：${error.message || error}（可刷新页面重试，或查看浏览器控制台）`);
    toast(`识别失败：${error.message || error}`);
  }
}

async function toggleVoiceInput(targetId) {
  try {
    if (webSpeechSession) { stopWebSpeech(); return; }
    if (asrRecorder) { await stopAsrRecording(); return; }
    if (webSpeechAvailable()) { startWebSpeechRecording(targetId); return; }
    await startAsrRecording(targetId);
  } catch (error) {
    setAsrStatus(`麦克风不可用：${error.message || error}（请允许麦克风权限后重试）`);
    toast(`无法开启麦克风：${error.message || error}`);
  }
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
  // 日程页不参与视线目标：避免黄色锁定框干扰浏览，提醒通过右侧按钮或语音设置。
  const selector = activePage === 'message-page' ? '.contact' : activePage === 'music-page' ? '.music-track-card' : null;
  if (!selector) return [];
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

function scheduleItemGazeScore(node, prediction) {
  // 日程行较高且信息密集：以注视点与行矩形的距离精确打分，
  // 容差约为行高，避免相邻行互相抢锁。
  if (!prediction.point) return 0;
  const rect = node.getBoundingClientRect();
  const x = prediction.point.x * window.innerWidth;
  const y = prediction.point.y * window.innerHeight;
  const distance = distanceToRect({ x, y }, rect);
  const tolerance = Math.max(22, Math.min(56, rect.height * 0.85));
  if (distance > tolerance) return 0;
  const spatialMatch = 1 - distance / tolerance;
  return 0.55 + spatialMatch * 0.40 + prediction.confidence * 0.05;
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
    return result;
  } catch (error) {
    // 结构化事件失败不阻断本地交互，但会让开发者在控制台看到明确原因。
    console.warn('多模态事件未记录：', error.message);
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

async function recordHeadDecision(decision, purpose, page = 'message', motionStrength) {
  const payload = { page, decision, gesture: decision === 'confirm' ? 'nod' : 'shake', purpose };
  // 把归一化的点头/摇头幅度一并上报，供后端手势自适应画像学习。
  if (typeof motionStrength === 'number' && motionStrength > 0 && motionStrength < 1) {
    payload.motion_strength = Number(motionStrength.toFixed(4));
  }
  const result = await recordBrowserEvent({
    modality: 'head_gesture',
    timestamp_ms: Date.now(),
    confidence: 0.72,
    payload,
  });
  // 后端返回的最新画像（含自适应阈值）驱动检测下限，形成“学习→调整→再检测”闭环。
  const updatedThreshold = result?.gesture_profile?.head_min_strength;
  if (typeof updatedThreshold === 'number') adaptiveHeadMinStrength = updatedThreshold;
}

async function recordHandGesture(decision, gesture, purpose, page = 'message', confidence = 0.75) {
  await recordBrowserEvent({
    modality: 'hand_gesture',
    timestamp_ms: Date.now(),
    confidence,
    payload: { page, decision, gesture, purpose },
  });
}

async function finishMessageDecision(action, outcome, source = 'button', motionStrength) {
  if (!pendingMessage || messageDecisionInProgress) return;
  messageDecisionInProgress = true;
  try {
    if (source === 'head') await recordHeadDecision(outcome === 'success' ? 'confirm' : 'reject', 'message_confirmation', 'message', motionStrength);
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
    ? '<p><button class="primary" id="confirm-send">确认发送</button><button class="secondary" id="cancel-send">取消</button></p><small class="decision-hint">确认后将仅在本地模拟消息应用中展示发送成功；也可说“是 / 确认”或“不用 / 取消”，或点头 / 摇头。</small>' : '';
  $('#message-result').innerHTML = `<div class="result-box"><strong>${result.message}</strong>${explanation}${actions}</div>`;
  if (result.pending) {
    pendingMessage = result.pending;
    $('#confirm-send').onclick = () => finishMessageDecision('confirm_send', 'success');
    $('#cancel-send').onclick = () => finishMessageDecision('cancel_message', 'cancel');
  }
}

function fusionSummary(result) {
  return result?.fusion?.summary ? `多模态综合判断：${result.fusion.summary}` : '';
}

async function submitSimulatedSpeech(source = 'mic', providedText) {
  const text = (providedText || '').trim();
  if (!text) {
    toast('未识别到语音内容，请重试。');
    return;
  }

  // 联系人候选出现时，优先将简短的“是 / 不用”视为对候选的回答，
  // 而不是一条新的消息发送指令。
  if (pendingGazeSuggestion) {
    const decision = parseContactSuggestionSpeech(text);
    if (decision === 'confirm') {
      const suggestion = pendingGazeSuggestion;
      clearGazeSuggestion();
      await lockGazeTarget(suggestion.node, suggestion.zone, suggestion.confidence);
      return;
    }
    if (decision === 'reject') {
      rejectGazeSuggestion();
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
      payload: { text, page: 'message', source },
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
    // 未理解/跨页面的指令：仅提示，不覆盖已有的确认框或结果。
    if (result.ignored || result.intent === 'unknown') {
      toast(result.message);
      return;
    }
    renderMessageUnderstanding(result);
  } catch (error) {
    toast(error.message);
  } finally {
    if (isMessageDecision) messageDecisionInProgress = false;
  }
}

async function submitSimulatedMusicCommand(source = 'mic', providedText) {
  const text = (providedText || '').trim();
  if (!text) {
    toast('未识别到语音内容，请重试。');
    return;
  }
  const timestamp = Date.now();
  try {
    await recordBrowserEvent({
      modality: 'speech_text',
      timestamp_ms: timestamp,
      confidence: 1,
      payload: { text, page: 'music', source },
    });
    const result = await api('understand_multimodal_command', {
      speech_timestamp_ms: timestamp,
      current_track_id: currentTrack?.id,
    });
    if (fusionSummary(result)) $('#gaze-feedback').textContent = fusionSummary(result);
    // 未理解/跨页面的指令：仅提示，不覆盖正在播放的卡片。
    if (result.ignored || result.intent === 'unknown') {
      toast(result.message);
      return;
    }
    if (result.intent === 'cancel_music_selection') {
      clearMusicTrackSelection(result.message);
      toast(result.message);
      return;
    }
    if (result.intent === 'next_track') {
      await nextTrack();
      return;
    }
    if (result.intent === 'toggle_playback') {
      toggleDemoPlayback();
      toast(playbackPaused ? '已暂停播放。' : '已继续播放。');
      return;
    }
    // 后端已切换模式/播放/切歌：应用返回的曲目、模式与偏好歌单。
    if (result.intent === 'start_focus' || result.intent === 'play_music' || result.intent === 'dislike_track') {
      if (result.mode) {
        activeMode = result.mode;
        updateModeStatus();
      }
      if (result.track) {
        stopDemoPlayback();
        currentTrack = result.track;
        currentRecommendationReason = result.recommendation_reason || '';
        currentPreferencePlaylist = result.preference_playlist || [];
        musicGazeTrackId = undefined;
        musicFeedbackLockedTrackId = undefined;
        musicCardSelected = false;
        switchArmed = false;
        $('#mode-decision').innerHTML = '';
        renderNowPlaying(result.message);
      } else {
        toast(result.message);
      }
      return;
    }
    if (result.intent === 'like_track') {
      if (result.preference_playlist) currentPreferencePlaylist = result.preference_playlist;
      if (currentTrack) {
        musicFeedbackLockedTrackId = currentTrack.id;
        musicGazeTrackId = undefined;
        renderNowPlaying(result.message);
      } else {
        toast(result.message);
      }
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
      await api('select_contact', { contact_id: node.dataset.id, input_modality: 'gaze' });
      document.querySelectorAll('.contact').forEach((item) => item.classList.remove('selected'));
      node.classList.add('selected');
      $('#message-result').innerHTML = '';
      $('#gaze-feedback').textContent = `已确认选择：${label}。现在可语音发消息。`;
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
  } else if (node.classList.contains('schedule-item')) {
    $('#gaze-feedback').textContent = `已持续注视：${label}。可在事项右侧点击「⏰ 设置提醒」。`;
  }
  // 操作冷却：同一目标在本次视线操作后的一段时间内不再允许被立即再次锁定。
  gazeActionCooldownKey = metadata.target_id;
  gazeActionCooldownUntil = performance.now() + GAZE_ACTION_COOLDOWN_MS;
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
  const schedulePage = activePage === 'memo-page';
  // 日程页改用“注视点与行矩形的实际距离”评分，代替按区域粗评，提升逐行选择精度。
  const rankedCandidates = eligibleGazeElements().map((node) => ({
    node,
    zone: elementZone(node),
    score: strictMusicGaze ? musicCardGazeScore(node, prediction)
      : schedulePage ? scheduleItemGazeScore(node, prediction) : gazeTargetScore(node, prediction),
  })).sort((left, right) => right.score - left.score);
  const bestCandidate = rankedCandidates[0];
  const secondCandidate = rankedCandidates[1];
  const minimumScore = strictMusicGaze ? MUSIC_GAZE_MINIMUM_SCORE : schedulePage ? 0.5 : 0.16;
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
  // 操作冷却期内，同一目标不允许被再次锁定（避免刚操作完又立即触发）。
  const candidateKey = closest.dataset.id || closest.dataset.trackId || closest.querySelector('[data-event-key]')?.dataset.eventKey;
  if (candidateKey && candidateKey === gazeActionCooldownKey && now < gazeActionCooldownUntil) return;
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

function headGestureHasTarget() {
  // 存在任何一个“是/否”交互时，点头/摇头才生效：联系人/歌曲候选、待确认消息、
  // 音乐卡片反馈、模式切换确认。
  return Boolean(
    pendingGazeSuggestion
    || pendingMusicGazeSuggestion
    || pendingMessage
    || canAnswerMusicFeedback()
    || modeDecisionActive()
  );
}

function modeDecisionActive() {
  return Boolean($('#decision-switch') && $('#decision-switch').isConnected);
}

async function confirmModeSwitch(motionStrength) {
  await recordHeadDecision('confirm', 'mode_switch', 'music', motionStrength);
  $('#decision-switch')?.click();
}

async function rejectModeSwitch(motionStrength) {
  await recordHeadDecision('reject', 'mode_switch', 'music', motionStrength);
  switchArmed = false;
  $('#mode-decision').innerHTML = '';
  toast('已取消切换模式。');
}

function observeHeadGesture(landmarks) {
  // 音乐反馈仅在歌曲卡片锁定后的短暂窗口内接受。窗口内暂停重新判断视线，
  // 使点头/摇头不会因为自身改变了眼部特征而丢失锁定。
  // 抬手、挥手会带动上半身与脸部关键点；手势识别期间不把这类变化误当成摇头。
  if (performance.now() < handGestureSuppressHeadUntil) return;
  const canAnswerMusic = canAnswerMusicFeedback();
  if ((!headGestureHasTarget() || (pendingMessage && messageDecisionInProgress)) || Date.now() - lastHeadGestureAt < 1500) return;
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
  // 归一化点头/摇头幅度：范围已除以双眼间距，映射到 (0,1) 供后端手势自适应画像学习。
  const motionStrength = Math.min(0.95, Math.max(0.05, Math.max(horizontalRange, verticalRange) * 2.0));
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
  // 自适应下限：撤销过某幅度后，后续动作需比它更明显才触发，避免把自然晃动当成指令。
  const adaptiveFloor = adaptiveHeadMinStrength || 0;
  const verticalGesture = Math.abs(directionalPitch) > verticalThreshold
    && verticalRange > Math.max(verticalThreshold * rangeMultiplier, adaptiveFloor)
    && Math.abs(directionalPitch) > Math.abs(directionalYaw) * dominance;
  const horizontalGesture = Math.abs(directionalYaw) > horizontalThreshold
    && horizontalRange > Math.max(horizontalThreshold * rangeMultiplier, adaptiveFloor)
    && Math.abs(directionalYaw) > Math.abs(directionalPitch) * dominance;
  if (verticalGesture) {
    lastHeadGestureAt = Date.now();
    if (pendingGazeSuggestion) {
      const suggestion = pendingGazeSuggestion;
      clearGazeSuggestion();
      void (async () => {
        await recordHeadDecision('confirm', 'contact_selection', 'message', motionStrength);
        await lockGazeTarget(suggestion.node, suggestion.zone, suggestion.confidence);
      })();
    } else if (pendingMusicGazeSuggestion) {
      const suggestion = pendingMusicGazeSuggestion;
      void (async () => {
        await recordHeadDecision('confirm', 'music_selection', 'music', motionStrength);
        await selectMusicTrackForInteraction(suggestion.node, suggestion.zone, suggestion.confidence);
      })();
    } else if (pendingMessage) {
      void finishMessageDecision('confirm_send', 'success', 'head', motionStrength);
    } else if (modeDecisionActive()) {
      void confirmModeSwitch(motionStrength);
    } else {
      void likeCurrentTrack('head', motionStrength);
    }
  } else if (horizontalGesture) {
    lastHeadGestureAt = Date.now();
    if (pendingGazeSuggestion) {
      void recordHeadDecision('reject', 'contact_selection', 'message', motionStrength);
      rejectGazeSuggestion();
    } else if (pendingMusicGazeSuggestion) {
      void recordHeadDecision('reject', 'music_selection', 'music', motionStrength);
      dismissMusicGazeSuggestion('好的，暂不选中这首歌。');
    } else if (pendingMessage) {
      void finishMessageDecision('cancel_message', 'cancel', 'head', motionStrength);
    } else if (modeDecisionActive()) {
      void rejectModeSwitch(motionStrength);
    } else {
      void dislikeCurrentTrack('head', motionStrength);
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
  throw new Error(`本地视觉运行时无法加载，请检查 web/vendor/mediapipe/wasm 是否完整。${lastError?.message ? ` 原因：${lastError.message}` : ''}`);
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
    setHandGestureStatus('手势已开启：点赞确认 / 喜欢，踩拒绝 / 不喜欢，食指上指或横向挥手切下一首，手掌开合暂停 / 继续（切歌无需先选中播放卡片）。');
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
    // 音乐手势必须以“已确认选中整个播放卡片”为前提，不能仅凭短暂注视触发。
    && musicCardSelected
  );
}

function canControlSelectedMusicTrack() {
  return Boolean(
    activeMode
    && currentTrack
    && musicCardSelected
  );
}

function canWaveSkip() {
  // 挥手切歌：大幅横向/斜向运动本身就是明确意图，只需处于音乐模式且有当前歌曲即可触发，
  // 不再要求先注视选中播放卡片，符合“听歌时挥手切歌”的真实使用场景。
  return Boolean(activeMode && currentTrack);
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
  if (gesture === 'Palm_Toggle' && canControlSelectedMusicTrack()) {
    await recordHandGesture('toggle_playback', 'Palm_Toggle', 'music_playback', 'music', confidence);
    toggleDemoPlayback();
    return true;
  }
  if (gesture === 'Palm_Wave' && canWaveSkip()) {
    await recordHandGesture('skip_track', 'Palm_Wave', 'music_skip', 'music', confidence);
    await nextTrack('hand');
    return true;
  }
  // 食指上指：静态手势，分类器识别稳定，作为切歌的可靠备用方案（无需大幅挥手）。
  if (gesture === 'Pointing_Up' && canWaveSkip()) {
    await recordHandGesture('skip_track', 'Pointing_Up', 'music_skip', 'music', confidence);
    await nextTrack('hand');
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
  // 只要有手部关键点就跟踪轨迹（不再强求先识别到“张开手掌”，分类器在挥手时
  // 经常短暂丢帧，只靠开手条件会让挥手几乎无法触发）。
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

function isWaveInProgress() {
  // 手势轨迹已出现明显横向分量时视为“挥手进行中”，此时抑制开合切换，避免误判。
  if (handMotionHistory.length < 2) return false;
  const xs = handMotionHistory.map((sample) => sample.x);
  return Math.max(...xs) - Math.min(...xs) >= HAND_WAVE_MIN_HORIZONTAL_SPAN * 0.6;
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
  // 只要手部可见即可跟踪挥动轨迹（挥手时“张开手掌”分类常丢帧，不再作为前置条件）。
  const waveEligible = Boolean(landmarks);
  // 先判挥手（大幅横向/斜向运动）再判开合（原地切换）：
  // 挥手起手时的张手动作不应被误判成暂停 / 继续。
  if (detectPalmWave(landmarks, now, waveEligible)) {
    handGestureCandidate = undefined;
    handMotionHistory = [];
    handGestureSuppressHeadUntil = Math.max(handGestureSuppressHeadUntil, now + HAND_GESTURE_HEAD_SUPPRESS_MS);
    if (now - lastHandGestureAt < HAND_GESTURE_COOLDOWN_MS) return;
    lastHandGestureAt = now;
    void applyHandGesture('Palm_Wave', 0.8).then((handled) => {
      setHandGestureStatus(handled
        ? '已通过斜向 / 横向挥手切换下一首。'
        : '已识别挥手；请先开启音乐模式并播放歌曲。');
    });
    return;
  }
  if (palmToggle && !isWaveInProgress()) {
    handGestureCandidate = undefined;
    handMotionHistory = [];
    handGestureSuppressHeadUntil = Math.max(handGestureSuppressHeadUntil, now + HAND_GESTURE_HEAD_SUPPRESS_MS);
    if (now - lastHandGestureAt < HAND_GESTURE_COOLDOWN_MS) return;
    lastHandGestureAt = now;
    void applyHandGesture('Palm_Toggle', 0.8).then((handled) => {
      setHandGestureStatus(handled
        ? '已通过手掌开合处理。'
        : '已识别手掌开合；请先选中音乐播放卡片。');
    });
    return;
  }
  if (palmState) {
    handGestureCandidate = undefined;
    handGestureSuppressHeadUntil = Math.max(handGestureSuppressHeadUntil, now + HAND_GESTURE_HEAD_SUPPRESS_MS);
    setHandGestureStatus(palmState === 'open'
      ? '已识别张开手掌：食指上指或横向挥动可切下一首；原位握拳可暂停 / 继续（需先选中播放卡片）。'
      : '已识别握拳：原位张开手掌可暂停 / 继续。');
    return;
  }
  if (!['Thumb_Up', 'Thumb_Down', 'Pointing_Up'].includes(gesture) || confidence < 0.72) {
    if (handGestureCandidate && now - handGestureCandidate.lastSeen <= HAND_GESTURE_CANDIDATE_GAP_MS) return;
    handGestureCandidate = undefined;
    return;
  }
  handGestureSuppressHeadUntil = Math.max(handGestureSuppressHeadUntil, now + HAND_GESTURE_HEAD_SUPPRESS_MS);
  if (!handGestureCandidate || handGestureCandidate.gesture !== gesture) {
    handGestureCandidate = { gesture, confidence, since: now, lastSeen: now };
    const label = gesture === 'Thumb_Up' ? '点赞' : gesture === 'Pointing_Up' ? '上指' : '踩';
    setHandGestureStatus(`检测到${label}，请保持片刻确认。`);
    return;
  }
  handGestureCandidate.confidence = Math.min(handGestureCandidate.confidence, confidence);
  handGestureCandidate.lastSeen = now;
  if (now - handGestureCandidate.since < HAND_GESTURE_DWELL_MS || now - lastHandGestureAt < HAND_GESTURE_COOLDOWN_MS) return;
  const stableGesture = handGestureCandidate;
  handGestureCandidate = undefined;
  lastHandGestureAt = now;
  if (stableGesture.gesture === 'Pointing_Up' && now - lastPointingSkipAt < POINTING_SKIP_COOLDOWN_MS) {
    setHandGestureStatus(`切歌间隔未到，请 ${Math.ceil((POINTING_SKIP_COOLDOWN_MS - (now - lastPointingSkipAt)) / 1000)} 秒后再上指切歌。`);
    return;
  }
  if (stableGesture.gesture === 'Pointing_Up') lastPointingSkipAt = now;
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
  const profile = data.profiles.find((item) => item.id === activeProfileId);
  const frequentContacts = new Set(profile?.frequent_contacts || []);
  $('#contacts').innerHTML = data.contacts.map((contact) => `
    <button class="card contact" data-id="${contact.id}">
      <strong>${contact.name}</strong><small>${contact.relationship}${frequentContacts.has(contact.id) ? ' · 常用联系人' : ''}</small>
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
    : '持续注视歌曲播放卡片约 0.55 秒后，系统会询问是否选中；确认后，即使切歌也保持选中，直到说“取消当前歌曲选择”。开启手势后：食指上指或横向挥动张开的手掌可直接切下一首，原位开合手掌可暂停 / 继续（开合需先选中播放卡片）。';
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
  navPausedPlayback = false;
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
  navPausedPlayback = false;
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

function pauseDemoPlayback() {
  // 离开音乐页时暂停本地演示播放；回到音乐页且无其他操作时自动恢复。
  if (!currentTrack || !activeMode || playbackPaused || !playbackTimer) return;
  navPausedPlayback = true;
  pausedPlaybackRemainingMs = Math.max(0, playbackDeadline - Date.now());
  clearInterval(playbackTimer);
  playbackTimer = undefined;
  playbackPaused = true;
  $('#toggle-playback').textContent = '继续播放';
  updatePlaybackProgress();
}

function resumeDemoPlayback() {
  if (!navPausedPlayback || !currentTrack || !activeMode || !playbackPaused) return;
  navPausedPlayback = false;
  playbackPaused = false;
  playbackDeadline = Date.now() + pausedPlaybackRemainingMs;
  pausedPlaybackRemainingMs = 0;
  $('#toggle-playback').textContent = '暂停播放';
  startDemoPlayback();
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

async function likeCurrentTrack(source = 'button', motionStrength) {
  try {
    if (!currentTrack) return;
    if (source === 'head') await recordHeadDecision('confirm', 'music_feedback', 'music', motionStrength);
    if (source === 'hand') await recordHandGesture('confirm', 'Thumb_Up', 'music_feedback', 'music');
    const body = { track_id: currentTrack.id };
    // 头部反馈的幅度同时写入撤销栈，撤销时用于“未通过样本”自适应。
    if (source === 'head' && typeof motionStrength === 'number' && motionStrength > 0 && motionStrength < 1) {
      body.motion_strength = Number(motionStrength.toFixed(4));
    }
    const result = await api('like_track', body);
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

async function dislikeCurrentTrack(source = 'button', motionStrength) {
  try {
    if (!currentTrack) return;
    stopDemoPlayback();
    if (source === 'head') await recordHeadDecision('reject', 'music_feedback', 'music', motionStrength);
    if (source === 'hand') await recordHandGesture('reject', 'Thumb_Down', 'music_feedback', 'music');
    const body = { current_track_id: currentTrack.id };
    if (source === 'head' && typeof motionStrength === 'number' && motionStrength > 0 && motionStrength < 1) {
      body.motion_strength = Number(motionStrength.toFixed(4));
    }
    const result = await api('dislike_track', body);
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

async function nextTrack(source = 'button') {
  try {
    if (!currentTrack) return;
    stopDemoPlayback();
    const body = { current_track_id: currentTrack.id };
    // 挥手触发的切歌写入交互历史，便于可解释展示。
    if (source === 'hand') body.input_modality = 'hand_gesture';
    const result = await api('next_track', body);
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
      ${item.is_past ? '' : `<button class="secondary reminder-btn" data-event-key="${item.event_key}" data-reminder-time="${item.reminder_time || ''}">${item.reminder_time ? `已设置 ${item.reminder_time.slice(11)}` : '⏰ 设置提醒'}</button>`}
    </article>`).join('');
  const fusionNote = fusionSummary(result);
  const llmMessage = result.llm?.used && result.message
    ? `<p class="assistant-answer"><span>个性化建议</span>${escapeHtml(result.message)}</p>` : '';
  const llmReasons = (result.explanation || [])
    .filter((item) => item.startsWith('本地大模型'))
    .map((item) => `<small class="fusion-note">${escapeHtml(item)}</small>`).join('');
  $('#memo-result').innerHTML = `<div class="result-box schedule-box"><div class="schedule-summary"><strong>${result.title || '全部日程'}</strong><span>共 ${result.total} 项 · 已完成 ${result.completed} 项 · 已过时间 ${result.past} 项</span></div>${llmMessage}${llmReasons}${fusionNote ? `<small class="fusion-note">${escapeHtml(fusionNote)}</small>` : ''}${items}</div>`;
  document.querySelectorAll('[data-event-key]').forEach((checkbox) => {
    checkbox.addEventListener('change', async () => {
      try {
        await api('toggle_event_completion', { event_key: checkbox.dataset.eventKey, completed: checkbox.checked });
        showSchedule(await api('query_schedule', { record_history: false }));
      } catch (error) {
        toast(error.message);
        checkbox.checked = !checkbox.checked;
      }
    });
  });
  // 每个日程项的“设置提醒 / 已设置”按钮：由用户自行选择哪些事项需要提醒（不再自动弹出建议）。
  document.querySelectorAll('.reminder-btn').forEach((button) => {
    button.addEventListener('click', async () => {
      const eventKey = button.dataset.eventKey;
      try {
        if (button.dataset.reminderTime) {
          await api('remove_reminder', { event_key: eventKey });
        } else {
          await api('create_reminder', { event_key: eventKey });
        }
        showSchedule(await api('query_schedule', { record_history: false }));
      } catch (error) { toast(error.message); }
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

async function submitScheduleSimulatedSpeech(source = 'mic', providedText) {
  const text = (providedText || '').trim();
  if (!text) {
    toast('未识别到语音内容，请重试。');
    return;
  }
  const timestamp = Date.now();
  await recordBrowserEvent({
    modality: 'speech_text',
    timestamp_ms: timestamp,
    confidence: 1,
    payload: { text, page: 'memo', source },
  });
  try {
    const result = await api('understand_multimodal_command', { speech_timestamp_ms: timestamp });
    // 未理解/跨页面的指令：仅提示，不覆盖已显示的日程。
    if (result.ignored || result.intent === 'unknown') {
      toast(result.message);
      return;
    }
    if (result.items) {
      showSchedule(result);
      return;
    }
    // 无事项的指令（设置/取消提醒、拒绝等）：仅提示；若日程可见，则刷新其状态而非覆盖。
    toast(result.message);
    if (document.querySelector('#memo-result .schedule-box')) {
      showSchedule(await api('query_schedule', { record_history: false }));
    }
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

const HISTORY_PAGE_LABELS = { message: '消息', music: '音乐', memo: '日程' };
const HISTORY_ACTION_LABELS = {
  open_page: '打开页面',
  select_contact: '选择联系人',
  undo: '撤销操作',
  like_track: '喜欢歌曲',
  dislike_track: '不喜欢歌曲',
  toggle_playback: '暂停 / 继续播放',
  confirm: '确认发送消息',
  reject: '摇头拒绝',
  stable_gaze: '稳定注视',
  send_message: '准备消息发送',
  cancel: '取消操作',
  query_schedule: '查询日程',
  cancel_music_selection: '取消歌曲选择',
  authorize_memo: '授权备忘录',
  revoke_memo: '取消备忘录授权',
  toggle_event_completion: '更新日程完成状态',
  create_reminder: '创建提醒',
  decline_reminder: '取消提醒',
  prepare_message: '准备消息',
  confirm_send: '发送消息',
  cancel_message: '取消发送',
  start_mode: '进入音乐模式',
  complete_track: '完整收听',
  next_track: '切换下一首',
  advance_track: '自动切歌',
  stop_mode: '停止音乐模式',
  play_music: '播放歌曲',
  query_priority: '查询优先级事项',
  decline_reminder: '取消提醒',
  update_reminder_preference: '设置提醒偏好',
  request_edit_memo: '请求修改备忘录',
};

function resolveTargetName(page, targetId) {
  if (!targetId) return '';
  if (page === 'message') {
    const contact = data.contacts.find((item) => item.id === targetId);
    return contact?.name || '';
  }
  if (page === 'music') {
    const track = data.music_library.find((item) => item.id === targetId);
    return track?.title || '';
  }
  if (page === 'memo') return targetId; // 授权文件名、日程标题等后端已给出可读文本
  return '';
}

const HISTORY_MODALITY_HINT = {
  speech_text: '通过语音',
  gaze: '通过视线',
  head_gesture: '通过头部动作',
  hand_gesture: '通过手势',
};

function describeHistoryRecord(record) {
  const name = resolveTargetName(record.page, record.target_id);
  const via = HISTORY_MODALITY_HINT[record.modality];
  const prefix = via ? `${via} ` : '';
  switch (record.action) {
    case 'open_page': return `打开了${HISTORY_PAGE_LABELS[record.page] || record.page}页面`;
    case 'select_contact': return name ? `${prefix}选中了联系人「${name}」` : `${prefix}选择了联系人`;
    case 'like_track': return name ? `${prefix}喜欢了歌曲《${name}》` : `${prefix}喜欢了当前歌曲`;
    case 'dislike_track': return name ? `${prefix}不喜欢歌曲《${name}》` : `${prefix}不喜欢了当前歌曲`;
    case 'toggle_playback': return `${prefix}暂停 / 继续了播放`;
    case 'undo': return '撤销了一步操作';
    case 'send_message': return name ? `${prefix}准备了发送给「${name}」的消息` : `${prefix}准备了消息发送`;
    case 'confirm': return `${prefix}确认发送了消息`;
    case 'cancel': return `${prefix}取消了操作`;
    case 'query_schedule': return `${prefix}查询了日程`;
    case 'next_track': return `${prefix}切换了下一首`;
    case 'cancel_music_selection': return `${prefix}取消了歌曲选择`;
    case 'authorize_memo': return name ? `授权了备忘录「${name}」` : '授权了备忘录';
    case 'revoke_memo': return name ? `取消了备忘录「${name}」的授权` : '取消了备忘录授权';
    case 'toggle_event_completion': return name ? `更新了日程「${name}」的完成状态` : '更新了日程完成状态';
    case 'create_reminder': return name ? `创建了「${name}」的提醒` : '创建了提醒';
    case 'decline_reminder': return '取消了本次提醒';
    case 'confirm_send': return '发送了消息';
    case 'cancel_message': return '取消了发送';
    case 'reject': return '摇头拒绝';
    case 'stable_gaze': return name ? `稳定注视了「${name}」` : '稳定注视了页面对象';
    default: return HISTORY_ACTION_LABELS[record.action] || record.action;
  }
}

function renderHistory(result) {
  const node = $('#history-list');
  if (!node) return;
  const records = result.records || [];
  if (!records.length) {
    node.textContent = '暂无本地交互历史。';
    return;
  }
  node.innerHTML = records.map((record) => {
    const time = new Date(record.timestamp_ms).toLocaleTimeString('zh-CN', { hour12: false });
    return `<div class="history-row"><span>${escapeHtml(describeHistoryRecord(record))}</span><small>${escapeHtml(time)}</small></div>`;
  }).join('');
}

function renderGestureProfile(profile) {
  const node = $('#gesture-profile-info');
  if (!node) return;
  if (!profile) {
    node.textContent = '';
    return;
  }
  const samples = profile.confirmed_samples || [];
  const undone = profile.undone_samples || [];
  const threshold = Number(profile.head_min_strength || 0);
  if (typeof threshold === 'number') adaptiveHeadMinStrength = threshold;
  node.textContent = samples.length
    ? `头部动作自适应画像：已学习 ${samples.length} 次确认幅度（最近 ${samples[samples.length - 1].toFixed(3)}），${undone.length} 次未通过样本，当前最小触发阈值 ${threshold.toFixed(3)}。`
    : '头部动作自适应画像：尚无样本。点头/摇头确认后会自动学习幅度并微调触发阈值。';
}

async function refreshHistory() {
  try {
    const result = await api('get_interaction_history');
    renderHistory(result);
    renderGestureProfile(result.gesture_profile);
  } catch (error) {
    toast(error.message);
  }
}

async function runSuggestedAction(actionId) {
  try {
    if (actionId === 'focus_contacts') {
      const profile = data.profiles.find((item) => item.id === activeProfileId);
      const frequent = new Set(profile?.frequent_contacts || []);
      const names = data.contacts.filter((contact) => frequent.has(contact.id)).map((contact) => contact.name);
      $('#contacts')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      toast(names.length ? `常用联系人：${names.join('、')}` : '暂无常用联系人记录。');
      return;
    }
    if (actionId === 'prepare_message') { toast('点击「🎤 语音输入」说出要发送的内容。'); return; }
    if (actionId === 'start_focus') { await requestMode('focus'); return; }
    if (actionId === 'resume_music') { await activateGeneralMusic(); return; }
    if (actionId === 'query_schedule') { showSchedule(await api('query_schedule')); return; }
    if (actionId === 'query_today') { showSchedule(await api('query_schedule', { scope: 'today' })); return; }
    toast('该建议暂不支持直接执行。');
  } catch (error) { toast(error.message); }
}

function renderPageSuggestions(result) {
  const text = $('#page-suggestion-text');
  const actions = $('#page-suggestion-actions');
  if (text) text.textContent = result.message || '根据你的本地使用习惯，可能想进行这些操作：';
  if (!actions) return;
  const list = result.actions || [];
  actions.innerHTML = list.length
    ? list.map((item) => `<button class="secondary suggestion-action" data-action-id="${escapeHtml(item.id)}">${escapeHtml(item.label)}</button>`).join('')
    : '<small class="note">暂无可用建议。</small>';
  actions.querySelectorAll('.suggestion-action').forEach((button) => {
    button.onclick = () => runSuggestedAction(button.dataset.actionId);
  });
}

async function loadPageSuggestions(page) {
  try {
    renderPageSuggestions(await api('open_page', { page }));
  } catch (error) {
    const text = $('#page-suggestion-text');
    if (text) text.textContent = '暂无法生成本地建议。';
  }
}

function bindEvents() {
  document.querySelectorAll('.nav-button').forEach((node) => node.addEventListener('click', async () => {
    // 切换页面不停止音乐模式：模式只在显式“停止模式”、切换演示用户或刷新页面时结束。
    // 离开音乐页时暂停本地演示播放，回到音乐页自动恢复。
    document.querySelectorAll('.nav-button, .page').forEach((item) => item.classList.remove('active'));
    node.classList.add('active');
    $(`#${node.dataset.page}-page`).classList.add('active');
    setCameraContext(node.textContent.trim());
    void recordScreenContext();
    void loadPageSuggestions(node.dataset.page);
    if (node.dataset.page === 'music') {
      resumeDemoPlayback();
      if (!activeMode) await activateGeneralMusic();
    } else {
      pauseDemoPlayback();
    }
  }));

  $('#refresh-history').onclick = refreshHistory;
  $('#undo-nontext').onclick = async () => {
    try {
      const result = await api('undo_last_nontext_operation');
      toast(result.message);
      if (result.kind === 'toggle_playback') {
        toggleDemoPlayback();
      } else if (result.kind === 'clear_contact') {
        selectedContactId = undefined;
        selectedContactSource = undefined;
        pendingMessage = undefined;
        document.querySelectorAll('.contact').forEach((item) => item.classList.remove('selected'));
        $('#message-result').innerHTML = '';
      } else if (result.kind === 'remove_authorized_sources') {
        renderAuthorizedSources(result.authorized_sources || [], result.message);
        $('#memo-result').innerHTML = '';
      } else if (result.kind === 'restore_event_completion') {
        showSchedule(await api('query_schedule', { record_history: false }));
      } else if (result.kind === 'restore_reminder_offer') {
        // 撤销“不用提醒”：重新给出该事项的提醒建议（日程页可见时刷新）。
        if (document.querySelector('#memo-result .schedule-box')) showSchedule(await api('query_schedule', { record_history: false }));
      }
      await refreshHistory();
    } catch (error) { toast(error.message); }
  };
  $('#clear-history').onclick = async () => {
    if (!window.confirm('确定清空本地交互历史、撤销记录和手势自适应信息吗？')) return;
    try {
      const result = await api('clear_interaction_history');
      toast(result.message);
      await refreshHistory();
    } catch (error) { toast(error.message); }
  };

  $('#profile').onchange = async (event) => {
    try {
      const result = await api('select_profile', { profile_id: event.target.value });
      activeProfileId = event.target.value;
      data.state = result.state;
      activeMode = result.state.active_mode;
      selectedContactId = undefined;
      selectedContactSource = undefined;
      pendingMessage = undefined;
      currentTrack = undefined;
      currentPreferencePlaylist = [];
      $('#message-result').innerHTML = '';
      $('#music-result').innerHTML = '';
      $('#memo-result').innerHTML = '';
      renderContacts();
      renderModes();
      renderAuthorizedSources(result.state.authorized_sources || [], result.message);
      toast(result.message);
    } catch (error) {
      event.target.value = activeProfileId;
      toast(error.message);
    }
  };
  document.querySelectorAll('.voice-input').forEach((node) => node.addEventListener('click', () => {
    // 侧边栏共用一个语音按钮：按当前激活页面决定指令路由。
    const page = document.querySelector('.page.active')?.id.replace('-page', '') || 'message';
    void toggleVoiceInput(page);
  }));
  $('#voice-confirm').onclick = confirmVoiceText;
  $('#voice-clear').onclick = () => {
    const box = $('#voice-result');
    if (box) box.value = '';
    setAsrStatus('已清空语音识别结果。');
  };
  $('#start-camera').onclick = startCamera;
  $('#calibrate-gaze').onclick = startGazeCalibration;
  $('#clear-gaze-calibration').onclick = clearSavedGazeCalibration;
  $('#stop-camera').onclick = () => stopCamera();
  $('#cancel-calibration').onclick = () => finishCalibration(false, '已取消视线校准。');

  $('#choose-memo').onclick = () => openMemoPicker('merge');
  $('#sync-memo').onclick = () => openMemoPicker('merge');
  $('#memo-file').onchange = authorizeSelectedMemos;
  $('#query-schedule').onclick = async () => {
    try { showSchedule(await api('query_schedule')); }
    catch (error) { toast(error.message); }
  };
}

function resetMessageForm(message) {
  pendingMessage = undefined;
  messageDecisionInProgress = false;
  selectedContactId = undefined;
  selectedContactSource = undefined;
  clearGazeSuggestion();
  clearGazeTarget();
  $('#message-result').innerHTML = '';
  document.querySelectorAll('.contact').forEach((item) => item.classList.remove('selected'));
  toast(message);
}

async function init() {
  data = await (await fetch('/api/bootstrap')).json();
  gazeMapper = loadGazeCalibration();
  $('#profile').innerHTML = data.profiles.map((profile) => `<option value="${profile.id}">${profile.display_name}</option>`).join('');
  activeProfileId = data.state.active_profile_id;
  $('#profile').value = activeProfileId;
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
  void loadPageSuggestions('message');
}

init().catch((error) => toast(`初始化失败：${error.message}`));

window.addEventListener('pagehide', () => {
  stopCamera(false);
  if (!activeMode) return;
  navigator.sendBeacon('/api/action', JSON.stringify({ action: 'stop_mode_silent' }));
});
