let data;
let pendingMessage;
let currentTrack;
let activeMode;
let switchArmed = false;

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
}

init().catch((error) => toast(`初始化失败：${error.message}`));

window.addEventListener('pagehide', () => {
  if (!activeMode) return;
  navigator.sendBeacon('/api/action', JSON.stringify({ action: 'stop_mode_silent' }));
});
