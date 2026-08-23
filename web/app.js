let data;
let pendingMessage;
let currentTrack;
let activeMode;
let switchArmed = false;
let cameraStream;
let faceLandmarker;
let faceDetectionFrame;
let lastFaceDetectionAt = 0;
let lastFacePresence;
let latestEyeFeatures;
let gazeMapper;
let previousGazeMapper;
let gazeTargetElement;
let gazeTargetSince = 0;
let gazeTargetLocked = false;
let calibrationActive = false;
let calibrationTimer;
let calibrationSamples = [];

const FACE_TASK_VERSION = '1.0.1';
const FACE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const FACE_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${FACE_TASK_VERSION}/wasm`;
const FACE_BUNDLE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${FACE_TASK_VERSION}/vision_bundle.mjs`;
const EYE_LANDMARKS = [33, 133, 159, 145, 160, 158, 153, 144, 362, 263, 386, 374, 385, 387, 373, 380, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477];
const CALIBRATION_POINTS = [
  { x: 0.18, y: 0.27 }, { x: 0.82, y: 0.27 }, { x: 0.50, y: 0.52 },
  { x: 0.18, y: 0.77 }, { x: 0.82, y: 0.77 },
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
  $('#stop-camera').disabled = !active;
  $('#camera-panel').hidden = !active;
}

function setCameraContext(label) {
  $('#camera-context').textContent = `上下文：${label}`;
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
  return left && right ? [...left, ...right] : undefined;
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

function fitGazeMapper(samples) {
  const width = samples[0].features.length + 1;
  const normal = Array.from({ length: width }, () => Array(width).fill(0));
  const targetX = Array(width).fill(0);
  const targetY = Array(width).fill(0);
  samples.forEach((sample) => {
    const row = [1, ...sample.features];
    row.forEach((left, i) => row.forEach((right, j) => { normal[i][j] += left * right; }));
    row.forEach((value, i) => {
      targetX[i] += value * sample.screenX;
      targetY[i] += value * sample.screenY;
    });
  });
  normal.forEach((row, index) => { row[index] += 0.0001; });
  return { x: solveLinearSystem(normal, targetX), y: solveLinearSystem(normal, targetY) };
}

function predictGazePoint(features) {
  if (!gazeMapper) return undefined;
  const row = [1, ...features];
  const dot = (weights) => weights.reduce((sum, value, index) => sum + value * row[index], 0);
  return {
    x: Math.max(0, Math.min(window.innerWidth, dot(gazeMapper.x))),
    y: Math.max(0, Math.min(window.innerHeight, dot(gazeMapper.y))),
  };
}

function clearGazeTarget() {
  document.querySelectorAll('.gaze-candidate, .gaze-focused').forEach((node) => {
    node.classList.remove('gaze-candidate', 'gaze-focused');
  });
  gazeTargetElement = undefined;
  gazeTargetLocked = false;
  gazeTargetSince = 0;
}

function eligibleGazeElements() {
  const activePage = document.querySelector('.page.active')?.id;
  const selector = activePage === 'message-page' ? '.contact' : activePage === 'music-page' ? '.genre' : '.schedule-item';
  return [...document.querySelectorAll(selector)].filter((node) => {
    const rect = node.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  });
}

function distanceToRect(point, rect) {
  const dx = Math.max(rect.left - point.x, 0, point.x - rect.right);
  const dy = Math.max(rect.top - point.y, 0, point.y - rect.bottom);
  return Math.hypot(dx, dy);
}

async function lockGazeTarget(node) {
  node.classList.add('gaze-focused');
  const label = node.querySelector('strong')?.textContent || node.querySelector('.schedule-title')?.textContent || '当前项目';
  $('#gaze-feedback').textContent = `已持续注视：${label}`;
  if (node.classList.contains('contact')) {
    try {
      await api('select_contact', { contact_id: node.dataset.id });
      document.querySelectorAll('.contact').forEach((item) => item.classList.remove('selected'));
      node.classList.add('selected');
      toast(`已通过摄像头注视选择：${label}`);
    } catch (error) { toast(error.message); }
  }
}

function updateGazeTarget(point) {
  if (!point || calibrationActive) return;
  const candidates = eligibleGazeElements();
  let closest;
  let closestDistance = Infinity;
  candidates.forEach((node) => {
    const distance = distanceToRect(point, node.getBoundingClientRect());
    if (distance < closestDistance) {
      closest = node;
      closestDistance = distance;
    }
  });
  if (!closest || closestDistance > 68) {
    clearGazeTarget();
    $('#gaze-feedback').textContent = '正在估计注视位置，请将视线停留在页面卡片上。';
    return;
  }
  if (gazeTargetElement !== closest) {
    clearGazeTarget();
    gazeTargetElement = closest;
    gazeTargetSince = performance.now();
    closest.classList.add('gaze-candidate');
    const label = closest.querySelector('strong')?.textContent || closest.querySelector('.schedule-title')?.textContent || '当前项目';
    $('#gaze-feedback').textContent = `候选注视：${label}（停留约 0.8 秒确认）`;
    return;
  }
  if (!gazeTargetLocked && performance.now() - gazeTargetSince >= 800) {
    gazeTargetLocked = true;
    lockGazeTarget(closest);
  }
}

function updateCalibrationPoint(point) {
  const target = $('#gaze-calibration-point');
  target.style.left = `${point.x * 100}vw`;
  target.style.top = `${point.y * 100}vh`;
}

function finishCalibration(success, message) {
  clearTimeout(calibrationTimer);
  calibrationActive = false;
  $('#gaze-calibration').hidden = true;
  if (success) {
    $('#gaze-feedback').textContent = '校准完成。请注视页面中的卡片，停留约 0.8 秒。';
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
  updateCalibrationPoint(point);
  $('#calibration-instruction').textContent = `请持续注视蓝点。正在采集第 ${index + 1} / ${CALIBRATION_POINTS.length} 个位置…`;
  calibrationTimer = window.setTimeout(() => {
    if (!latestEyeFeatures) {
      $('#calibration-instruction').textContent = '没有检测到眼部关键点，请正对摄像头后保持注视。将重新采集此位置…';
      calibrationTimer = window.setTimeout(() => runCalibrationStep(index), 1300);
      return;
    }
    calibrationSamples.push({
      features: [...latestEyeFeatures],
      screenX: window.innerWidth * point.x,
      screenY: window.innerHeight * point.y,
    });
    calibrationTimer = window.setTimeout(() => runCalibrationStep(index + 1), 650);
  }, 2200);
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
  const vision = await FilesetResolver.forVisionTasks(FACE_WASM_URL);
  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: FACE_MODEL_URL },
    runningMode: 'VIDEO',
    numFaces: 1,
    minFaceDetectionConfidence: 0.55,
    minFacePresenceConfidence: 0.55,
    minTrackingConfidence: 0.55,
  });
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
  if (now - lastFaceDetectionAt >= 120) {
    lastFaceDetectionAt = now;
    try {
      const result = faceLandmarker.detectForVideo(video, now);
      const landmarks = result.faceLandmarks?.[0];
      if (landmarks) {
        drawFaceLandmarks(landmarks);
        latestEyeFeatures = extractEyeFeatures(landmarks);
        if (gazeMapper && latestEyeFeatures) updateGazeTarget(predictGazePoint(latestEyeFeatures));
      } else {
        latestEyeFeatures = undefined;
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
  latestEyeFeatures = undefined;
  gazeMapper = undefined;
  clearGazeTarget();
  clearFaceOverlay();
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = undefined;
  const video = $('#camera-preview');
  video.srcObject = null;
  updateCameraControls(false);
  setCameraStatus('摄像头已关闭。', 'idle');
  if (showMessage) toast('摄像头已关闭，本机画面已停止。');
}

function renderContacts() {
  $('#contacts').innerHTML = data.contacts.map((contact) => `
    <button class="card contact" data-id="${contact.id}">
      <strong>${contact.name}</strong><small>${contact.relationship}${contact.frequent ? ' · 常用联系人' : ''}</small>
    </button>`).join('');
  document.querySelectorAll('.contact').forEach((node) => node.addEventListener('click', async () => {
    await api('select_contact', { contact_id: node.dataset.id });
    document.querySelectorAll('.contact').forEach((item) => item.classList.remove('selected'));
    node.classList.add('selected');
    toast(`已模拟注视：${node.querySelector('strong').textContent}`);
  }));
}

const MODE_LABELS = { focus: '专注模式', driving: '开车模式', entertainment: '娱乐模式' };

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
    $('#mode-status').textContent = '当前未启用音乐模式。请选择一种模式后开始播放。';
    $('#genres').classList.add('muted');
    return;
  }
  $('#mode-status').innerHTML = `当前正在使用<strong>${MODE_LABELS[activeMode]}</strong>。<button class="secondary" id="stop-mode">停止当前模式</button><button class="secondary" id="switch-mode">切换模式</button>`;
  $('#genres').classList.remove('muted');
  $('#stop-mode').onclick = stopMode;
  $('#switch-mode').onclick = () => {
    switchArmed = true;
    $('#mode-decision').textContent = '请从上方选择想切换到的模式。';
  };
}

async function requestMode(nextMode) {
  if (activeMode && activeMode !== nextMode) {
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
    const result = await api('start_mode', { mode, mode_label: MODE_LABELS[mode] });
    activeMode = mode;
    switchArmed = false;
    $('#mode-decision').innerHTML = '';
    updateModeStatus();
    toast(result.message);
  } catch (error) { toast(error.message); }
}

async function stopMode() {
  try {
    const result = await api('stop_mode');
    activeMode = undefined;
    switchArmed = false;
    $('#mode-decision').innerHTML = '';
    updateModeStatus();
    toast(result.message);
  } catch (error) { toast(error.message); }
}

function renderGenres() {
  const genres = [
    ['lofi', 'Lo-fi', '轻节奏、低干扰'],
    ['light_music', '轻音乐', '平稳、舒缓'],
    ['pure_music', '纯音乐', '安静、无歌词'],
    ['classical', '古典音乐', '器乐与交响乐'],
    ['pop', '流行音乐', '轻松、易听'],
    ['jazz', '爵士乐', '松弛、有律动'],
    ['rock', '摇滚乐', '高能量、适合通勤'],
    ['electronic', '电子音乐', '节奏鲜明、适合驾车'],
  ];
  $('#genres').innerHTML = genres.map(([id, name, note]) => `
    <button class="card genre" data-genre="${id}"><strong>${name}</strong><small>${note}</small></button>`).join('');
  document.querySelectorAll('.genre').forEach((node) => node.addEventListener('click', async () => {
    try {
      const result = await api('play_genre', { genre: node.dataset.genre, mode_label: MODE_LABELS[activeMode] });
      currentTrack = result.track;
      document.querySelectorAll('.genre').forEach((item) => item.classList.remove('selected'));
      node.classList.add('selected');
      renderNowPlaying(result.message);
    } catch (error) { toast(error.message); }
  }));
}

function renderNowPlaying(message) {
  if (!currentTrack) return;
  $('#music-result').innerHTML = `<div class="result-box"><strong>正在播放：${currentTrack.title}</strong><p>${message}</p><button class="secondary" id="like-track">我喜欢这首</button><button class="secondary" id="skip-track">不喜欢 / 下一首</button></div>`;
  $('#like-track').onclick = likeCurrentTrack;
  $('#skip-track').onclick = nextTrack;
}

async function likeCurrentTrack() {
  try {
    const result = await api('like_track', { genre: currentTrack.genre });
    renderNowPlaying(result.message);
    toast(result.message);
  } catch (error) { toast(error.message); }
}

async function nextTrack() {
  try {
    const result = await api('next_track', { genre: currentTrack.genre, current_track_id: currentTrack.id });
    currentTrack = result.track;
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
  $('#memo-result').innerHTML = `<div class="result-box schedule-box"><div class="schedule-summary"><strong>全部日程</strong><span>共 ${result.total} 项 · 已完成 ${result.completed} 项 · 已过时间 ${result.past} 项</span></div>${items}</div>`;
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

async function authorizeSelectedMemos() {
  const input = $('#memo-file');
  const files = [...input.files];
  if (!files.length) return;
  try {
    const selectedFiles = await Promise.all(files.map(readTextFile));
    const result = await api('authorize_memo_files', { files: selectedFiles });
    const names = result.sources.map((source) => source.display_name).join('、');
    $('#authorization-status').textContent = `当前已授权 ${result.sources.length} 个文件：${names}（仅本地读取）`;
    input.value = '';
    $('#memo-result').innerHTML = '';
    toast('备忘录授权成功。');
  } catch (error) {
    input.value = '';
    toast(error.message);
  }
}

function bindEvents() {
  document.querySelectorAll('.nav-button').forEach((node) => node.addEventListener('click', async () => {
    if (node.dataset.page !== 'music' && activeMode) await stopMode();
    document.querySelectorAll('.nav-button, .page').forEach((item) => item.classList.remove('active'));
    node.classList.add('active');
    $(`#${node.dataset.page}-page`).classList.add('active');
    setCameraContext(node.textContent.trim());
  }));

  $('#prepare-message').onclick = async () => {
    try {
      const result = await api('prepare_message', { content: $('#message-content').value });
      pendingMessage = result.pending;
      $('#message-result').innerHTML = `<div class="result-box">${result.message}<p><button class="primary" id="confirm-send">确认发送</button><button class="secondary" id="cancel-send">取消</button></p></div>`;
      $('#confirm-send').onclick = async () => resetMessageForm((await api('confirm_send', pendingMessage)).message);
      $('#cancel-send').onclick = async () => resetMessageForm((await api('cancel_message')).message);
    } catch (error) { toast(error.message); }
  };

  $('#start-camera').onclick = startCamera;
  $('#calibrate-gaze').onclick = startGazeCalibration;
  $('#stop-camera').onclick = () => stopCamera();
  $('#cancel-calibration').onclick = () => finishCalibration(false, '已取消视线校准。');

  $('#choose-memo').onclick = () => $('#memo-file').click();
  $('#memo-file').onchange = authorizeSelectedMemos;
  $('#query-schedule').onclick = async () => {
    try { showSchedule(await api('query_schedule')); }
    catch (error) { toast(error.message); }
  };
}

function resetMessageForm(message) {
  pendingMessage = undefined;
  $('#message-content').value = '';
  $('#message-result').innerHTML = '';
  document.querySelectorAll('.contact').forEach((item) => item.classList.remove('selected'));
  toast(message);
}

async function init() {
  data = await (await fetch('/api/bootstrap')).json();
  $('#profile').innerHTML = data.profiles.map((profile) => `<option value="${profile.id}">${profile.display_name}</option>`).join('');
  if (data.state.authorized_sources?.length) {
    const names = data.state.authorized_sources.map((source) => source.display_name || source).join('、');
    $('#authorization-status').textContent = `当前已授权 ${data.state.authorized_sources.length} 个文件：${names}（仅本地读取）`;
  }
  renderContacts();
  activeMode = data.state.active_mode;
  renderModes();
  renderGenres();
  bindEvents();
  updateCameraControls(false);
}

init().catch((error) => toast(`初始化失败：${error.message}`));

window.addEventListener('pagehide', () => {
  stopCamera(false);
  if (!activeMode) return;
  navigator.sendBeacon('/api/action', JSON.stringify({ action: 'stop_mode_silent' }));
});
