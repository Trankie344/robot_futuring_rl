const PLAYBACK_RATES = [0.25, 0.5, 0.75, 1];
const DEFAULT_PLAYBACK_RATE = 0.5;
const LABEL_DONE = 2;
const VALID_FRAME_STATES = new Set([-1, 0, 1, LABEL_DONE]);

const STORAGE_KEYS = {
  deviceId: "tristate.deviceId",
  nickname: "tristate.nickname",
  playbackRate: "tristate.playbackRate",
};

const CAMERA_ROLES = {
  main: "main",
  left: "left",
  right: "right",
};

const state = {
  deviceId: loadDeviceId(),
  nickname: safeLocalStorageGet(STORAGE_KEYS.nickname) || "",
  config: null,
  task: null,
  frameLabels: [],
  selectionStart: 0,
  selectionEnd: 1,
  editingSegment: null,
  videoRoleKeys: {},
  playing: false,
  busy: false,
  heartbeatTimer: null,
  progressTimer: null,
  playbackRate: loadPlaybackRate(),
};

const videos = {
  [CAMERA_ROLES.main]: document.getElementById("videoGround"),
  [CAMERA_ROLES.left]: document.getElementById("videoLeftWrist"),
  [CAMERA_ROLES.right]: document.getElementById("videoRightWrist"),
};

const elements = {
  datasetName: document.getElementById("datasetName"),
  taskMeta: document.getElementById("taskMeta"),
  lockMeta: document.getElementById("lockMeta"),
  playButton: document.getElementById("playButton"),
  progressBar: document.getElementById("progressBar"),
  timeMeta: document.getElementById("timeMeta"),
  selectionStart: document.getElementById("selectionStart"),
  selectionEnd: document.getElementById("selectionEnd"),
  selectionStartInput: document.getElementById("selectionStartInput"),
  selectionEndInput: document.getElementById("selectionEndInput"),
  jumpUnlabeledStartButton: document.getElementById("jumpUnlabeledStartButton"),
  selectionSummary: document.getElementById("selectionSummary"),
  selectedRangeMeta: document.getElementById("selectedRangeMeta"),
  selectionBand: document.getElementById("selectionBand"),
  expertOverlay: document.getElementById("expertOverlay"),
  openLabelModalButton: document.getElementById("openLabelModalButton"),
  labelModal: document.getElementById("labelModal"),
  closeLabelModalButton: document.getElementById("closeLabelModalButton"),
  deleteSelectionButton: document.getElementById("deleteSelectionButton"),
  modalRangeText: document.getElementById("modalRangeText"),
  stateButtons: Array.from(document.querySelectorAll("[data-state]")),
  segmentList: document.getElementById("segmentList"),
  coverageMeta: document.getElementById("coverageMeta"),
  coverageFill: document.getElementById("coverageFill"),
  skipButton: document.getElementById("skipButton"),
  message: document.getElementById("message"),
  completedCount: document.getElementById("completedCount"),
  pendingCount: document.getElementById("pendingCount"),
  lockedCount: document.getElementById("lockedCount"),
  reviewCount: document.getElementById("reviewCount"),
  speedSelect: document.getElementById("speedSelect"),
  submitButton: document.getElementById("submitButton"),
  taskControls: Array.from(document.querySelectorAll(".js-task-control")),
  videoTiles: Array.from(document.querySelectorAll(".video-tile")),
  phaseBadges: Array.from(document.querySelectorAll("[data-phase-badge]")),
  cameraLabelMain: document.getElementById("cameraLabelMain"),
  cameraLabelLeft: document.getElementById("cameraLabelLeft"),
  cameraLabelRight: document.getElementById("cameraLabelRight"),
};

safeLocalStorageSet(STORAGE_KEYS.deviceId, state.deviceId);

function safeLocalStorageGet(key) {
  try {
    return window.localStorage.getItem(key);
  } catch (_error) {
    return null;
  }
}

function safeLocalStorageSet(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch (_error) {
    // Labeling still works if a browser blocks localStorage.
  }
}

function loadDeviceId() {
  const existing = safeLocalStorageGet(STORAGE_KEYS.deviceId);
  return existing || createDeviceId();
}

function loadPlaybackRate() {
  const saved = Number(safeLocalStorageGet(STORAGE_KEYS.playbackRate));
  return PLAYBACK_RATES.includes(saved) ? saved : DEFAULT_PLAYBACK_RATE;
}

function applyPlaybackRate() {
  if (elements.speedSelect) {
    elements.speedSelect.value = String(state.playbackRate);
  }
  for (const video of Object.values(videos)) {
    video.playbackRate = state.playbackRate;
  }
}

function createDeviceId() {
  const cryptoApi = window.crypto;
  if (cryptoApi && typeof cryptoApi.randomUUID === "function") {
    return cryptoApi.randomUUID();
  }
  if (cryptoApi && typeof cryptoApi.getRandomValues === "function") {
    const bytes = new Uint8Array(16);
    cryptoApi.getRandomValues(bytes);
    return `device-${Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("")}`;
  }
  return `device-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function devicePayload() {
  const payload = { device_id: state.deviceId };
  if (state.nickname) {
    payload.nickname = state.nickname;
  }
  return payload;
}

async function api(path, options = {}) {
  const headers = {
    ...(options.body ? { "Content-Type": "application/json" } : {}),
    ...(options.headers || {}),
  };
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${extractErrorDetail(text)}`.trim());
  }
  return response.json();
}

function extractErrorDetail(text) {
  if (!text) {
    return "API request failed";
  }
  try {
    const parsed = JSON.parse(text);
    return parsed.detail || text;
  } catch (_error) {
    return text;
  }
}

function setMessage(text, tone = "info") {
  elements.message.textContent = text;
  elements.message.classList.toggle("is-error", tone === "error");
  elements.message.classList.toggle("is-ok", tone === "ok");
}

function setBusy(isBusy) {
  state.busy = isBusy;
  updateControls();
}

function updateControls() {
  const hasTask = Boolean(state.task);
  for (const control of elements.taskControls) {
    control.disabled = state.busy || !hasTask;
  }
  elements.playButton.disabled = !hasTask;
  elements.openLabelModalButton.disabled = state.busy || !hasTask;
  elements.submitButton.disabled = state.busy || !hasTask || !frameLabelsComplete();
}

function frameLabelsComplete() {
  return frameLabelsValidationError() === null;
}

function frameLabelsValidationError(labels = state.frameLabels) {
  if (!state.task || labels.length !== frameCount(state.task) || labels.length === 0) {
    return "Frame label count does not match the episode.";
  }
  const completionFrames = [];
  for (let index = 0; index < labels.length; index += 1) {
    const label = labels[index];
    if (!Number.isInteger(label) || !VALID_FRAME_STATES.has(label)) {
      return `Frame ${index} must be an integer state in -1, 0, 1, 2.`;
    }
    if (label === LABEL_DONE) {
      completionFrames.push(index);
    }
  }
  if (completionFrames.length > 1) {
    return "Completion state 2 may appear at most once per episode.";
  }
  if (completionFrames.length === 1 && completionFrames[0] !== labels.length - 1) {
    return "Completion state 2 may only appear at the final episode frame.";
  }
  return null;
}

function frameCount(task) {
  return Number(task?.frame_count ?? Math.max(0, Number(task?.end_frame ?? 0) - Number(task?.start_frame ?? 0)));
}

function playbackStart(task) {
  return Number(task?.start_time ?? 0);
}

function playbackEnd(task) {
  return Number(task?.end_time ?? 0);
}

function windowDuration(task) {
  return Math.max(0.001, playbackEnd(task) - playbackStart(task));
}

function resolveVideoRoles(videoKeys) {
  const keys = Array.isArray(videoKeys) ? videoKeys : [];
  return {
    [CAMERA_ROLES.main]:
      keys.find((key) => key.endsWith(".top")) ||
      keys.find((key) => key.endsWith(".ground")) ||
      keys[0],
    [CAMERA_ROLES.left]: keys.find((key) => key.endsWith(".left_wrist")),
    [CAMERA_ROLES.right]: keys.find((key) => key.endsWith(".right_wrist")),
  };
}

function updateCameraLabels() {
  elements.cameraLabelMain.textContent = state.videoRoleKeys[CAMERA_ROLES.main] || "--";
  elements.cameraLabelLeft.textContent = state.videoRoleKeys[CAMERA_ROLES.left] || "--";
  elements.cameraLabelRight.textContent = state.videoRoleKeys[CAMERA_ROLES.right] || "--";
}

async function runAction(action) {
  if (state.busy) {
    return;
  }
  setBusy(true);
  try {
    await action();
  } catch (error) {
    handleError(error);
  } finally {
    setBusy(false);
  }
}

function handleError(error) {
  const message = error instanceof Error ? error.message : String(error);
  setMessage(message, "error");
}

function isAutoplayPolicyError(error) {
  return error?.name === "NotAllowedError";
}

async function loadConfig() {
  state.config = await api("/api/config");
  state.videoRoleKeys = resolveVideoRoles(state.config.video_keys);
  updateCameraLabels();
  elements.datasetName.textContent = `${state.config.dataset_name} - ${state.config.fps}fps`;
}

async function claimNextTask(excludeTaskId) {
  pauseAll();
  stopHeartbeat();
  clearTaskDisplay();
  const payload = devicePayload();
  if (excludeTaskId !== undefined && excludeTaskId !== null) {
    payload.exclude_task_id = excludeTaskId;
  }
  const data = await api("/api/tasks/claim", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  state.task = data.task;
  if (!state.task) {
    clearVideos();
    elements.taskMeta.textContent = "No pending task";
    elements.lockMeta.textContent = "Unlocked";
    setMessage("No pending task available.");
    await refreshProgress();
    updateControls();
    return;
  }
  await bindTask(state.task);
  startHeartbeat();
  await refreshProgress();
  updateControls();
  await startAutoPlayback();
}

async function bindTask(task) {
  const datasetName = task.dataset_name || state.config?.dataset_name || "dataset";
  const fps = task.fps ?? state.config?.fps ?? "--";
  const start = playbackStart(task);
  const end = playbackEnd(task);
  const duration = Math.max(0.001, end - start);

  elements.datasetName.textContent = `${datasetName} - ${fps}fps`;
  elements.taskMeta.textContent = `#${task.id} - ep ${task.episode_index} - frames ${task.start_frame}-${task.end_frame}`;
  elements.lockMeta.textContent = `device ${state.deviceId.slice(0, 8)} - heartbeat 10s`;
  elements.timeMeta.textContent = `0.00 / ${duration.toFixed(2)}s`;
  elements.progressBar.style.width = "0%";
  state.frameLabels = Array(frameCount(task)).fill(null);
  configureSelection(task);
  renderExpertSegments(task);
  renderAnnotationState();
  setSegmentPhase("episode");

  const readyPromises = [];
  for (const [role, video] of Object.entries(videos)) {
    const videoKey = state.videoRoleKeys[role];
    video.pause();
    video.src = videoKey ? task.videos[videoKey] || "" : "";
    video.preload = "auto";
    video.load();
    readyPromises.push(waitForVideoMetadata(video));
  }
  await Promise.allSettled(readyPromises);
  seekAll(start);
  applyPlaybackRate();
  setMessage(`Task claimed: ${datasetName}, episode ${task.episode_index}.`, "ok");
}

function waitForVideoMetadata(video) {
  if (video.readyState >= 1) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const finish = () => {
      video.removeEventListener("loadedmetadata", finish);
      video.removeEventListener("error", finish);
      resolve();
    };
    video.addEventListener("loadedmetadata", finish, { once: true });
    video.addEventListener("error", finish, { once: true });
  });
}

function clearTaskDisplay() {
  state.task = null;
  state.frameLabels = [];
  state.selectionStart = 0;
  state.selectionEnd = 1;
  state.editingSegment = null;
  elements.taskMeta.textContent = "Waiting for task";
  elements.lockMeta.textContent = "Unlocked";
  elements.progressBar.style.width = "0%";
  elements.timeMeta.textContent = "0.00 / 0.00s";
  elements.expertOverlay?.replaceChildren();
  setExpertActive(false);
  renderAnnotationState();
  updateControls();
}

function clearVideos() {
  for (const video of Object.values(videos)) {
    video.pause();
    video.removeAttribute("src");
    video.load();
  }
}

function configureSelection(task) {
  const min = Number(task.start_frame);
  const max = Number(task.end_frame);
  elements.selectionStart.min = String(min);
  elements.selectionStart.max = String(Math.max(min, max - 1));
  elements.selectionEnd.min = String(min + 1);
  elements.selectionEnd.max = String(max);
  elements.selectionStartInput.min = String(min);
  elements.selectionStartInput.max = String(Math.max(min, max - 1));
  elements.selectionEndInput.min = String(min + 1);
  elements.selectionEndInput.max = String(max);
  setSelection(min, Math.min(max, min + 1), { seek: false });
}

function setSelection(startFrame, endFrame, options = {}) {
  if (!state.task) {
    return;
  }
  const min = Number(state.task.start_frame);
  const max = Number(state.task.end_frame);
  let start = Math.max(min, Math.min(max - 1, Math.round(Number(startFrame))));
  let end = Math.max(min + 1, Math.min(max, Math.round(Number(endFrame))));
  if (end <= start) {
    end = Math.min(max, start + 1);
  }
  state.selectionStart = start;
  state.selectionEnd = end;
  elements.selectionEnd.min = String(Math.min(max, start + 1));
  elements.selectionEndInput.min = String(Math.min(max, start + 1));
  elements.selectionStart.value = String(start);
  elements.selectionEnd.value = String(end);
  elements.selectionStartInput.value = String(start);
  elements.selectionEndInput.value = String(end);
  renderSelection();
  if (options.seek !== false) {
    const seekFrame = options.changed === "end" ? end : start;
    seekAll(seekFrame / Number(state.task.fps));
  }
}

function renderSelection() {
  if (!state.task) {
    elements.selectionSummary.textContent = "--";
    elements.selectedRangeMeta.textContent = "--";
    elements.modalRangeText.textContent = "Selected interval";
    elements.selectionBand.style.setProperty("--selection-left", "0%");
    elements.selectionBand.style.setProperty("--selection-width", "0%");
    return;
  }
  const totalFrames = frameCount(state.task);
  const startOffset = state.selectionStart - Number(state.task.start_frame);
  const endOffset = state.selectionEnd - Number(state.task.start_frame);
  const left = totalFrames > 0 ? (startOffset / totalFrames) * 100 : 0;
  const width = totalFrames > 0 ? ((endOffset - startOffset) / totalFrames) * 100 : 0;
  const text = `frames ${state.selectionStart}-${state.selectionEnd}`;
  elements.selectionSummary.textContent = text;
  elements.selectedRangeMeta.textContent = text;
  elements.modalRangeText.textContent = text;
  elements.selectionBand.style.setProperty("--selection-left", `${Math.max(0, Math.min(100, left))}%`);
  elements.selectionBand.style.setProperty("--selection-width", `${Math.max(0, Math.min(100, width))}%`);
}

function openLabelModal() {
  if (!state.task) {
    return;
  }
  state.editingSegment = null;
  renderSelection();
  if (typeof elements.labelModal.showModal === "function") {
    elements.labelModal.showModal();
  } else {
    elements.labelModal.removeAttribute("hidden");
  }
}

function closeLabelModal() {
  if (typeof elements.labelModal.close === "function") {
    elements.labelModal.close();
  } else {
    elements.labelModal.setAttribute("hidden", "hidden");
  }
}

function parseStateValue(value) {
  const label = Number(value);
  if (!Number.isInteger(label) || !VALID_FRAME_STATES.has(label)) {
    throw new Error(`Invalid frame state: ${value}`);
  }
  return label;
}

function applyStateToSelection(stateValue) {
  if (!state.task) {
    return;
  }
  const label = parseStateValue(stateValue);
  const taskEnd = Number(state.task.end_frame);
  if (label === LABEL_DONE && (state.selectionStart !== taskEnd - 1 || state.selectionEnd !== taskEnd)) {
    setMessage("Completion state 2 can only label the single final episode frame.", "error");
    return;
  }
  if (selectionConflictsWithExistingLabels(state.selectionStart, state.selectionEnd, state.editingSegment)) {
    setMessage("Selection overlaps an existing labeled interval. Edit or delete that interval first.", "error");
    return;
  }
  const taskStart = Number(state.task.start_frame);
  for (let frame = state.selectionStart; frame < state.selectionEnd; frame += 1) {
    state.frameLabels[frame - taskStart] = label;
  }
  state.editingSegment = null;
  renderAnnotationState();
  closeLabelModal();
  setMessage(`Marked frames ${state.selectionStart}-${state.selectionEnd} as ${label}.`, "ok");
}

function selectionConflictsWithExistingLabels(startFrame, endFrame, allowedSegment = null) {
  if (!state.task) {
    return false;
  }
  if (
    allowedSegment &&
    Number(allowedSegment.start_frame) === Number(startFrame) &&
    Number(allowedSegment.end_frame) === Number(endFrame)
  ) {
    return false;
  }
  const taskStart = Number(state.task.start_frame);
  for (let frame = startFrame; frame < endFrame; frame += 1) {
    if (state.frameLabels[frame - taskStart] !== null) {
      return true;
    }
  }
  return false;
}

function firstUnlabeledInterval() {
  if (!state.task) {
    return null;
  }
  const firstIndex = state.frameLabels.findIndex((label) => label === null);
  if (firstIndex === -1) {
    return null;
  }
  let endIndex = firstIndex + 1;
  while (endIndex < state.frameLabels.length && state.frameLabels[endIndex] === null) {
    endIndex += 1;
  }
  const taskStart = Number(state.task.start_frame);
  return {
    start_frame: taskStart + firstIndex,
    end_frame: taskStart + endIndex,
  };
}

function firstUnlabeledFrame() {
  return firstUnlabeledInterval()?.start_frame ?? null;
}

function jumpToFirstUnlabeledFrame() {
  const interval = firstUnlabeledInterval();
  if (!interval) {
    setMessage("No unlabeled frames left.", "ok");
    return;
  }
  state.editingSegment = null;
  setSelection(interval.start_frame, interval.end_frame, { changed: "start" });
  setMessage(`Selected first unlabeled interval ${interval.start_frame}-${interval.end_frame}.`, "ok");
}

function deleteSelectedRange() {
  if (!state.task) {
    return;
  }
  const taskStart = Number(state.task.start_frame);
  for (let frame = state.selectionStart; frame < state.selectionEnd; frame += 1) {
    state.frameLabels[frame - taskStart] = null;
  }
  state.editingSegment = null;
  renderAnnotationState();
  closeLabelModal();
  setMessage(`Deleted labels for frames ${state.selectionStart}-${state.selectionEnd}.`, "ok");
}

function renderAnnotationState() {
  renderSelection();
  const total = state.frameLabels.length;
  const labeled = state.frameLabels.filter((label) => label !== null).length;
  elements.coverageMeta.textContent = `${labeled} / ${total} frames`;
  elements.coverageFill.style.width = total > 0 ? `${Math.round((labeled / total) * 1000) / 10}%` : "0%";
  renderSegments();
  updateControls();
}

function currentSegments() {
  if (!state.task) {
    return [];
  }
  const segments = [];
  const taskStart = Number(state.task.start_frame);
  let current = null;
  let start = null;
  for (let index = 0; index < state.frameLabels.length; index += 1) {
    const label = state.frameLabels[index];
    if (label === null) {
      if (current !== null && start !== null) {
        segments.push({ start_frame: taskStart + start, end_frame: taskStart + index, state: current });
      }
      current = null;
      start = null;
      continue;
    }
    if (current === null) {
      current = label;
      start = index;
      continue;
    }
    if (label !== current) {
      segments.push({ start_frame: taskStart + start, end_frame: taskStart + index, state: current });
      current = label;
      start = index;
    }
  }
  if (current !== null && start !== null) {
    segments.push({ start_frame: taskStart + start, end_frame: taskStart + state.frameLabels.length, state: current });
  }
  return segments;
}

function renderSegments() {
  elements.segmentList.replaceChildren();
  const segments = currentSegments();
  for (const segment of segments) {
    const item = document.createElement("article");
    item.className = "segment-item";

    const text = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = `${segment.start_frame}-${segment.end_frame}`;
    const detail = document.createElement("span");
    detail.textContent = `state ${segment.state}`;
    text.append(title, detail);

    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Edit";
    button.addEventListener("click", () => {
      setSelection(segment.start_frame, segment.end_frame);
      state.editingSegment = { ...segment };
      openLabelModal();
      state.editingSegment = { ...segment };
    });

    item.append(text, button);
    elements.segmentList.appendChild(item);
  }
}

async function submitLabels() {
  if (!state.task) {
    setMessage("No task claimed.", "error");
    return;
  }
  const validationError = frameLabelsValidationError();
  if (validationError !== null) {
    setMessage(validationError, "error");
    return;
  }
  const currentTask = state.task;
  const frameLabels = state.frameLabels.slice();
  await api(`/api/tasks/${currentTask.id}/submit`, {
    method: "POST",
    body: JSON.stringify({ ...devicePayload(), frame_labels: frameLabels }),
  });
  setMessage("Saved frame labels and refreshed tri-state metadata.", "ok");
  await claimNextTask();
}

async function skipTask() {
  if (!state.task) {
    setMessage("No task claimed.", "error");
    return;
  }
  const currentTask = state.task;
  await api(`/api/tasks/${currentTask.id}/skip`, {
    method: "POST",
    body: JSON.stringify(devicePayload()),
  });
  setMessage("Skipped task; requesting another episode.", "ok");
  await claimNextTask(currentTask.id);
}

function startHeartbeat() {
  stopHeartbeat();
  sendHeartbeat();
  state.heartbeatTimer = window.setInterval(sendHeartbeat, 10000);
}

function stopHeartbeat() {
  if (state.heartbeatTimer) {
    window.clearInterval(state.heartbeatTimer);
  }
  state.heartbeatTimer = null;
}

async function sendHeartbeat() {
  if (!state.task) {
    return;
  }
  try {
    await api(`/api/tasks/${state.task.id}/heartbeat`, {
      method: "POST",
      body: JSON.stringify(devicePayload()),
    });
    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    elements.lockMeta.textContent = `lock renewed ${time}`;
  } catch (error) {
    handleError(error);
  }
}

async function refreshProgress() {
  const progress = await api("/api/progress");
  elements.completedCount.textContent = progress.completed ?? 0;
  elements.pendingCount.textContent = progress.pending ?? 0;
  elements.lockedCount.textContent = progress.locked ?? 0;
  elements.reviewCount.textContent = progress.review ?? 0;
}

function safeSeek(video, time) {
  if (!Number.isFinite(time)) {
    return;
  }
  try {
    video.currentTime = time;
  } catch (_error) {
    video.addEventListener(
      "loadedmetadata",
      () => {
        try {
          video.currentTime = time;
        } catch (__error) {
          // Some corrupt source files cannot seek; keep the UI responsive.
        }
      },
      { once: true },
    );
  }
}

function seekAll(time) {
  for (const video of Object.values(videos)) {
    safeSeek(video, time);
  }
}

async function startAutoPlayback() {
  if (!state.task) {
    return;
  }
  seekAll(playbackStart(state.task));
  try {
    await playAll();
  } catch (error) {
    state.playing = false;
    updatePlayButton();
    if (isAutoplayPolicyError(error)) {
      setMessage("Task claimed. Press Play if the browser blocks autoplay.", "ok");
      return;
    }
    handleError(error);
  }
}

async function togglePlayback() {
  if (!state.task) {
    return;
  }
  if (state.playing) {
    pauseAll();
    return;
  }
  await playAll();
}

async function playAll() {
  if (!state.task) {
    return;
  }
  applyPlaybackRate();
  syncToMain(true);
  state.playing = true;
  updatePlayButton();
  const results = await Promise.allSettled(Object.values(videos).map((video) => video.play()));
  const rejected = results.find((result) => result.status === "rejected");
  if (rejected) {
    const playbackError = rejected.reason instanceof Error
      ? rejected.reason
      : new Error("Playback failed; check that the served video files are browser-readable.");
    pauseAll();
    throw playbackError;
  }
}

function pauseAll() {
  for (const video of Object.values(videos)) {
    video.pause();
  }
  state.playing = false;
  updatePlayButton();
}

function updatePlayButton() {
  elements.playButton.textContent = state.playing ? "Pause" : "Play";
  elements.playButton.setAttribute("aria-pressed", String(state.playing));
}

function syncToMain(force = false) {
  const main = videos[CAMERA_ROLES.main];
  const anchor = main.currentTime;
  if (!Number.isFinite(anchor)) {
    return;
  }
  for (const [role, video] of Object.entries(videos)) {
    if (role === CAMERA_ROLES.main) {
      continue;
    }
    if (force || Math.abs(video.currentTime - anchor) > 0.08) {
      safeSeek(video, anchor);
    }
  }
}

function setSegmentPhase(phase) {
  const isEpisode = phase === "episode";
  for (const tile of elements.videoTiles) {
    tile.classList.toggle("is-label-segment", isEpisode);
  }
  for (const badge of elements.phaseBadges) {
    badge.textContent = isEpisode ? "FULL EPISODE" : "WAITING";
  }
}

function renderExpertSegments(task) {
  if (!elements.expertOverlay) {
    return;
  }
  elements.expertOverlay.replaceChildren();
  const totalFrames = frameCount(task);
  const taskStart = Number(task.start_frame);
  for (const segment of task.expert_segments || []) {
    const left = ((Number(segment.start_frame) - taskStart) / totalFrames) * 100;
    const width = ((Number(segment.end_frame) - Number(segment.start_frame)) / totalFrames) * 100;
    const band = document.createElement("div");
    band.className = "expert-band";
    band.style.left = `${Math.max(0, Math.min(100, left))}%`;
    band.style.width = `${Math.max(0, Math.min(100, width))}%`;
    elements.expertOverlay.appendChild(band);
  }
}

function isCurrentTimeExpert(task, current) {
  return (task.expert_segments || []).some(
    (segment) => current >= Number(segment.start_time) && current < Number(segment.end_time),
  );
}

function setExpertActive(isExpert) {
  for (const tile of elements.videoTiles) {
    tile.classList.toggle("is-expert-segment", isExpert);
  }
  for (const badge of elements.phaseBadges) {
    badge.classList.toggle("is-expert-segment", isExpert);
    badge.textContent = isExpert ? "EXPERT CORRECTION" : "FULL EPISODE";
  }
}

function updatePlaybackWindow() {
  if (state.task) {
    const main = videos[CAMERA_ROLES.main];
    const start = playbackStart(state.task);
    const end = playbackEnd(state.task);
    const duration = windowDuration(state.task);
    let current = Number.isFinite(main.currentTime) ? main.currentTime : start;

    if (current < start - 0.08) {
      seekAll(start);
      current = start;
      if (state.playing) {
        syncToMain(true);
      }
    } else if (current < end) {
      if (state.playing) {
        syncToMain(false);
      }
    } else if (current >= end) {
      current = end;
      pauseAll();
    }

    const elapsed = Math.min(duration, Math.max(0, current - start));
    elements.progressBar.style.width = `${Math.round((elapsed / duration) * 1000) / 10}%`;
    elements.timeMeta.textContent = `${elapsed.toFixed(2)} / ${duration.toFixed(2)}s`;
    setExpertActive(isCurrentTimeExpert(state.task, current));
  }
  window.requestAnimationFrame(updatePlaybackWindow);
}

function handleKeydown(event) {
  const target = event.target;
  if (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement ||
    target?.isContentEditable
  ) {
    return;
  }

  if (event.code === "Space") {
    event.preventDefault();
    runAction(togglePlayback);
    return;
  }

  const key = event.key.toLowerCase();
  if (key === "1") {
    applyStateToSelection(1);
  } else if (key === "2") {
    applyStateToSelection(-1);
  } else if (key === "3") {
    applyStateToSelection(0);
  } else if (key === "d") {
    event.preventDefault();
    applyStateToSelection(LABEL_DONE);
  } else if (event.key === "Enter") {
    event.preventDefault();
    runAction(submitLabels);
  } else if (key === "s") {
    event.preventDefault();
    runAction(skipTask);
  }
}

function bindEvents() {
  elements.selectionStart.addEventListener("input", () => {
    state.editingSegment = null;
    setSelection(elements.selectionStart.value, state.selectionEnd, { changed: "start" });
  });
  elements.selectionEnd.addEventListener("input", () => {
    state.editingSegment = null;
    setSelection(state.selectionStart, elements.selectionEnd.value, { changed: "end" });
  });
  elements.selectionStartInput.addEventListener("change", () => {
    state.editingSegment = null;
    setSelection(elements.selectionStartInput.value, state.selectionEnd, { changed: "start" });
  });
  elements.selectionEndInput.addEventListener("change", () => {
    state.editingSegment = null;
    setSelection(state.selectionStart, elements.selectionEndInput.value, { changed: "end" });
  });
  elements.jumpUnlabeledStartButton.addEventListener("click", jumpToFirstUnlabeledFrame);
  elements.openLabelModalButton.addEventListener("click", openLabelModal);
  elements.closeLabelModalButton.addEventListener("click", closeLabelModal);
  elements.deleteSelectionButton.addEventListener("click", deleteSelectedRange);
  for (const button of elements.stateButtons) {
    button.addEventListener("click", () => applyStateToSelection(button.dataset.state));
  }
  elements.submitButton.addEventListener("click", () => runAction(submitLabels));
  elements.skipButton.addEventListener("click", () => runAction(skipTask));
  elements.playButton.addEventListener("click", () => runAction(togglePlayback));
  elements.speedSelect.addEventListener("change", () => {
    const selected = Number(elements.speedSelect.value);
    state.playbackRate = PLAYBACK_RATES.includes(selected) ? selected : DEFAULT_PLAYBACK_RATE;
    safeLocalStorageSet(STORAGE_KEYS.playbackRate, String(state.playbackRate));
    applyPlaybackRate();
  });
  document.addEventListener("keydown", handleKeydown);
}

async function initialize() {
  updateControls();
  bindEvents();
  applyPlaybackRate();
  await loadConfig();
  await refreshProgress();
  await claimNextTask();
  state.progressTimer = window.setInterval(() => refreshProgress().catch(handleError), 15000);
  updatePlaybackWindow();
}

initialize().catch((error) => {
  handleError(error);
  updatePlaybackWindow();
});
