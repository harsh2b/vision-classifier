/**
 * VisionChunk AI — Web Interface Logic
 * Connects frontend to local Tornado API for real-time video classification and verification.
 */

// Category icons & friendly names
const CATEGORY_ICONS = {
  "Sports_HighMotion": "⚽",
  "Gameplay_Esports": "🎮",
  "Podcast_TalkingHead": "🎙️",
  "Wildlife_Nature": "🦁",
  "LifeRecord_Egocentric": "📹",
  "Performance_Narrative": "🎭"
};

const CATEGORY_LABELS = {
  "Sports_HighMotion": "Sports & High-Motion",
  "Gameplay_Esports": "Gameplay & Esports",
  "Podcast_TalkingHead": "Podcast & Talking Head",
  "Wildlife_Nature": "Wildlife & Nature",
  "LifeRecord_Egocentric": "Life Record / Egocentric",
  "Performance_Narrative": "Performance & Narrative"
};

// Global App State
let allSamples = [];
let selectedSample = null;
let uploadedFile = null;
let currentInputMode = 'samples'; // 'samples' or 'upload'
let latestPrediction = null;

// DOM Elements
const samplesListEl = document.getElementById('keyframesGrid');
const mainVideoPlayer = document.getElementById('mainVideoPlayer');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const classifyBtn = document.getElementById('classifyBtn');
const classifyBtnText = document.getElementById('classifyBtnText');
const btnSpinner = document.getElementById('btnSpinner');

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  loadSamples();
  loadStats();
  setupDragAndDrop();
});

// Switch between Sample Explorer and File Upload
function switchInputTab(tab) {
  currentInputMode = tab;
  const tabSamplesBtn = document.getElementById('tabSamplesBtn');
  const tabUploadBtn = document.getElementById('tabUploadBtn');
  const samplesContent = document.getElementById('samplesTabContent');
  const uploadContent = document.getElementById('uploadTabContent');

  if (tab === 'samples') {
    tabSamplesBtn.classList.add('active');
    tabUploadBtn.classList.remove('active');
    samplesContent.style.display = 'block';
    uploadContent.style.display = 'none';

    if (selectedSample) {
      setVideoSource(selectedSample.stream_url || `/api/video/samples/${selectedSample.filename}`, selectedSample.filename);
      classifyBtn.disabled = false;
    } else {
      classifyBtn.disabled = true;
    }
  } else {
    tabUploadBtn.classList.add('active');
    tabSamplesBtn.classList.remove('active');
    uploadContent.style.display = 'block';
    samplesContent.style.display = 'none';

    if (uploadedFile) {
      classifyBtn.disabled = false;
    } else {
      classifyBtn.disabled = true;
    }
  }
}

// Fetch Sample Videos from Server
async function loadSamples() {
  const container = document.getElementById('samplesList');
  try {
    const res = await fetch('/api/samples');
    const data = await res.json();
    allSamples = data.samples || [];
    renderSamples(allSamples);

    // Auto-select first sample for immediate testing convenience
    if (allSamples.length > 0) {
      selectSample(allSamples[0].filename);
    }
  } catch (err) {
    container.innerHTML = `<div class="prob-empty" style="color: #FB7185;">Error loading samples: ${err.message}</div>`;
  }
}

// Render sample video items into the scroll list
function renderSamples(samples) {
  const container = document.getElementById('samplesList');
  if (samples.length === 0) {
    container.innerHTML = '<div class="prob-empty">No videos found for this filter.</div>';
    return;
  }

  container.innerHTML = samples.map(item => {
    const icon = CATEGORY_ICONS[item.ground_truth_category] || '🎞️';
    const cleanName = item.filename.replace('.mp4', '');
    const isSelected = selectedSample && selectedSample.filename === item.filename ? 'selected' : '';
    return `
      <div class="sample-item ${isSelected}" id="sample-${item.filename}" onclick="selectSample('${item.filename}')">
        <div class="sample-item-left">
          <span class="sample-item-icon">${icon}</span>
          <div>
            <div class="sample-item-name">${cleanName}</div>
            <div class="sample-item-sub">${item.sub_category} • ${item.file_size_mb} MB</div>
          </div>
        </div>
        <div class="sample-item-tag">${item.ground_truth_category}</div>
      </div>
    `;
  }).join('');
}

// Filter samples by category
function filterSamples() {
  const category = document.getElementById('categoryFilter').value;
  if (category === 'ALL') {
    renderSamples(allSamples);
  } else {
    const filtered = allSamples.filter(s => s.ground_truth_category === category);
    renderSamples(filtered);
  }
}

// Select a sample video
function selectSample(filename) {
  const found = allSamples.find(s => s.filename === filename);
  if (!found) return;
  selectedSample = found;

  // Highlight selected in list
  document.querySelectorAll('.sample-item').forEach(el => el.classList.remove('selected'));
  const activeEl = document.getElementById(`sample-${filename}`);
  if (activeEl) activeEl.classList.add('selected');

  // Update video player
  const streamUrl = `/api/video/samples/${filename}`;
  setVideoSource(streamUrl, filename, `${found.ground_truth_category} • ${found.file_size_mb} MB`);
  classifyBtn.disabled = false;
}

// Update HTML5 video player source
function setVideoSource(url, title, meta = '') {
  document.getElementById('activeVideoTitle').innerText = title;
  document.getElementById('activeVideoMeta').innerText = meta;
  videoPlaceholder.style.display = 'none';
  mainVideoPlayer.style.display = 'block';
  mainVideoPlayer.src = url;
  mainVideoPlayer.load();
}

// Drag & Drop Setup for User Uploads
function setupDragAndDrop() {
  const dropzone = document.getElementById('dropzone');

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      processUploadedFile(files[0]);
    }
  });
}

function handleFileSelect(event) {
  const files = event.target.files;
  if (files.length > 0) {
    processUploadedFile(files[0]);
  }
}

function processUploadedFile(file) {
  uploadedFile = file;
  const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
  const infoEl = document.getElementById('uploadFileInfo');
  infoEl.innerText = `Selected File: ${file.name} (${sizeMb} MB)`;
  infoEl.style.display = 'block';

  // Preview directly in video player via Object URL
  const objUrl = URL.createObjectURL(file);
  setVideoSource(objUrl, file.name, `Local Upload • ${sizeMb} MB`);
  classifyBtn.disabled = false;
}

// Execute Inference & Classification
async function runInference() {
  if (classifyBtn.disabled) return;

  classifyBtn.disabled = true;
  btnSpinner.style.display = 'inline-block';
  classifyBtnText.innerText = 'Extracting Keyframes & Running Inference...';

  try {
    let response;
    if (currentInputMode === 'upload' && uploadedFile) {
      const formData = new FormData();
      formData.append('video_file', uploadedFile);
      response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });
    } else if (selectedSample) {
      response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: jsonBody({ filename: selectedSample.filename })
      });
    } else {
      throw new Error("No video selected.");
    }

    const result = await response.json();
    if (!result.success) {
      throw new Error(result.error || "Inference failed.");
    }

    latestPrediction = result;
    displayResults(result);

  } catch (err) {
    alert("Inference Error: " + err.message);
  } finally {
    classifyBtn.disabled = false;
    btnSpinner.style.display = 'none';
    classifyBtnText.innerText = 'Classify Video & Verify Correctness';
  }
}

function jsonBody(obj) {
  return JSON.stringify(obj);
}

// Display Results & Verification Breakdown
function displayResults(data) {
  // 1. Primary Prediction & Confidence
  const icon = CATEGORY_ICONS[data.predicted_category] || '🎬';
  document.getElementById('predCategoryDisplay').innerHTML = `${icon} ${CATEGORY_LABELS[data.predicted_category] || data.predicted_category}`;
  document.getElementById('predConfidenceDisplay').innerText = `${data.confidence}%`;
  document.getElementById('latencyBadge').innerText = `⚡ Latency: ${data.inference_latency_ms} ms`;

  // Uncertainty badge
  const uncertStatus = document.getElementById('uncertaintyStatus');
  const uncertText = document.getElementById('uncertaintyText');
  if (data.is_uncertain) {
    uncertStatus.className = 'uncertainty-status status-uncertain';
    uncertText.innerText = `Uncertainty Flag: YES (Low confidence margin: ${data.margin}) — Tier-2 Escalation`;
  } else {
    uncertStatus.className = 'uncertainty-status status-certain';
    uncertText.innerText = `High Confidence (Certainty Margin: ${data.margin})`;
  }

  // 2. Verification Banner (Is it predicting correct or not?)
  const banner = document.getElementById('verificationBanner');
  const verdictIcon = document.getElementById('verdictIcon');
  const verdictHeading = document.getElementById('verdictHeading');
  const verdictDetail = document.getElementById('verdictDetail');
  const verdictTag = document.getElementById('verdictTag');

  banner.className = 'verification-banner'; // reset classes

  if (data.has_ground_truth) {
    if (data.is_correct) {
      banner.classList.add('banner-correct');
      verdictIcon.innerText = '✅';
      verdictHeading.innerText = 'PREDICTION ACCURATE: Matches Ground Truth!';
      verdictDetail.innerHTML = `The model accurately identified <strong>${data.predicted_category}</strong> with <strong>${data.confidence}%</strong> confidence. Matches official Video-MME dataset benchmark truth.`;
      verdictTag.innerText = 'CORRECT';
    } else {
      banner.classList.add('banner-incorrect');
      verdictIcon.innerText = '❌';
      verdictHeading.innerText = 'PREDICTION MISCLASSIFIED (Ground Truth Mismatch)';
      verdictDetail.innerHTML = `Model predicted <strong>${data.predicted_category}</strong> (${data.confidence}%), but benchmark Ground Truth is <strong>${data.ground_truth}</strong> (${data.sub_category}).`;
      verdictTag.innerText = 'MISMATCH';
    }
  } else {
    banner.classList.add('banner-custom');
    verdictIcon.innerText = '🎯';
    verdictHeading.innerText = `Prediction: ${CATEGORY_LABELS[data.predicted_category] || data.predicted_category}`;
    verdictDetail.innerHTML = `Analyzed custom video locally in <strong>${data.inference_latency_ms} ms</strong>. Confirm if the model prediction is accurate using the verification feedback buttons below.`;
    verdictTag.innerText = 'CUSTOM VIDEO';
  }

  // 3. Recommended Chunking Policy
  document.getElementById('chunkingStrategyDisplay').innerText = data.chunking_strategy;
  document.getElementById('chunkingTriggerDisplay').innerText = data.chunking_trigger;
  document.getElementById('chunkingRationaleDisplay').innerText = data.chunking_rationale;

  // 4. Probability Distribution Bars
  renderProbabilities(data.probabilities, data.predicted_category);

  // 5. Sparse Keyframes & Attention Timeline
  renderKeyframes(data.keyframes);

  // Hide disagree picker if open
  document.getElementById('disagreePicker').style.display = 'none';
}

// Render animated probability bars
function renderProbabilities(probs, winnerCategory) {
  const container = document.getElementById('probBarsList');
  if (!probs) return;

  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

  container.innerHTML = sorted.map(([cat, val]) => {
    const isWinner = cat === winnerCategory;
    const friendlyName = CATEGORY_LABELS[cat] || cat;
    const icon = CATEGORY_ICONS[cat] || '•';
    return `
      <div class="prob-item ${isWinner ? 'winner' : ''}">
        <div class="prob-info">
          <span class="prob-name">${icon} ${friendlyName}</span>
          <span class="prob-val">${val}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" style="width: ${Math.max(val, 2)}%;"></div>
        </div>
      </div>
    `;
  }).join('');
}

// Render Keyframes & Gated Attention weights
function renderKeyframes(keyframes) {
  const container = document.getElementById('keyframesGrid');
  if (!keyframes || keyframes.length === 0) return;

  // Find max attention weight to highlight keyframe
  const maxAttn = Math.max(...keyframes.map(k => k.attention_weight));

  container.innerHTML = keyframes.map(kf => {
    const isHighest = kf.attention_weight === maxAttn && maxAttn > 0.15;
    const attnPercent = (kf.attention_weight * 100).toFixed(1);
    const motionPercent = Math.min(100, (kf.motion_energy * 200)).toFixed(0);

    return `
      <div class="keyframe-card ${isHighest ? 'high-attention' : ''}">
        <div class="frame-thumb">
          <img src="${kf.thumbnail_b64}" alt="Frame ${kf.index}">
        </div>
        <div class="frame-info">
          <div class="frame-tag-row">
            <span class="frame-index">#${kf.index} (${kf.timestamp_normalized}s)</span>
            <span class="frame-attn-badge">Attn: ${attnPercent}%</span>
          </div>
          <div class="frame-motion-bar" title="Motion Delta: ${kf.motion_energy}">
            <div class="frame-motion-fill" style="width: ${motionPercent}%;"></div>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

// Human Verification Feedback Handlers
async function submitFeedback(verdict, userCat = null) {
  if (!latestPrediction) return;

  const payload = {
    verdict: verdict,
    video_name: latestPrediction.filename,
    predicted_category: latestPrediction.predicted_category,
    user_category: userCat || latestPrediction.predicted_category
  };

  try {
    const res = await fetch('/api/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const stats = await res.json();
    updateFeedbackCounter(stats);

    // Provide visual confirmation
    if (verdict === 'correct') {
      alert("✅ Verification recorded: Confirmed as Correct!");
    } else {
      alert("📝 Correction recorded: Marked as Incorrect!");
    }
    document.getElementById('disagreePicker').style.display = 'none';
  } catch (err) {
    console.error("Failed to submit feedback:", err);
  }
}

function openDisagreeModal() {
  const picker = document.getElementById('disagreePicker');
  picker.style.display = picker.style.display === 'none' ? 'block' : 'none';
}

function confirmIncorrectFeedback() {
  const selectedCat = document.getElementById('userCorrectCategorySelect').value;
  submitFeedback('incorrect', selectedCat);
}

function updateFeedbackCounter(stats) {
  const counterEl = document.getElementById('feedbackCounter');
  counterEl.innerText = `Session Accuracy: ${stats.user_accuracy_pct}% (${stats.correct_count} / ${stats.total_verified} verified)`;
}

// Fetch stats on launch
async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    if (data.benchmark) {
      const f1El = document.getElementById('headerF1');
      if (f1El) f1El.innerText = data.benchmark.macro_f1;
    }
    if (data.user_verification) {
      updateFeedbackCounter(data.user_verification);
    }
  } catch (e) {
    console.warn("Could not load initial stats:", e);
  }
}
