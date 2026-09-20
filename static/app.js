/**
 * KsheerRaksha-AI — Frontend Application Controller
 * Handles video ingestion, preset quick-loading, live API communication,
 * and unified dual-branch results rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements - Navigation & Inputs
  const presetContainer = document.getElementById('presetContainer');
  const dropzone = document.getElementById('dropzone');
  const videoFileInput = document.getElementById('videoFileInput');
  const browseBtn = document.getElementById('browseBtn');
  const changeVideoBtn = document.getElementById('changeVideoBtn');
  const videoPreviewWrapper = document.getElementById('videoPreviewWrapper');
  const videoPlayer = document.getElementById('videoPlayer');
  const videoFilename = document.getElementById('videoFilename');
  const videoDurationBadge = document.getElementById('videoDurationBadge');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const scanOverlay = document.getElementById('scanOverlay');

  // State Containers
  const loadingState = document.getElementById('loadingState');
  const resultsContent = document.getElementById('resultsContent');

  // Results Header & Meta
  const metaFilename = document.getElementById('metaFilename');
  const metaLatency = document.getElementById('metaLatency');
  const metaDevice = document.getElementById('metaDevice');

  // Stream 1: Lameness Elements
  const statusBox = document.getElementById('lamenessStatusBox');
  const statusTitle = document.getElementById('statusTitle');
  const statusBadge = document.getElementById('statusBadge');
  const confidenceVal = document.getElementById('confidenceVal');
  const riskProbVal = document.getElementById('riskProbVal');
  const riskMeterFill = document.getElementById('riskMeterFill');
  const explanationText = document.getElementById('explanationText');
  const actionCard = document.getElementById('actionCard');
  const actionBadge = document.getElementById('actionBadge');
  const actionText = document.getElementById('actionText');

  // Stream 2: Visual Udder Elements
  const udderStatusBox = document.getElementById('udderStatusBox');
  const udderStatusTitle = document.getElementById('udderStatusTitle');
  const udderStatusBadge = document.getElementById('udderStatusBadge');
  const udderConfidenceVal = document.getElementById('udderConfidenceVal');
  const udderAbnormalProbVal = document.getElementById('udderAbnormalProbVal');
  const udderMeterFill = document.getElementById('udderMeterFill');
  const udderExplanationText = document.getElementById('udderExplanationText');
  const udderUsableFrames = document.getElementById('udderUsableFrames');
  const udderErythema = document.getElementById('udderErythema');
  const udderAsymmetry = document.getElementById('udderAsymmetry');
  const udderRoughness = document.getElementById('udderRoughness');

  // Summary Strip & Details Elements
  const sumLamenessStatus = document.getElementById('sumLamenessStatus');
  const sumUdderStatus = document.getElementById('sumUdderStatus');
  const detLamenessTime = document.getElementById('detLamenessTime');
  const detUdderTime = document.getElementById('detUdderTime');

  let currentVideoFile = null;
  let currentSamplePath = null;

  // 1. Fetch & Render Quick-Load Presets
  fetch('/api/samples')
    .then(res => res.json())
    .then(samples => {
      presetContainer.innerHTML = '';
      samples.forEach(s => {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'preset-chip';
        chip.innerHTML = `
          <span>${s.name}</span>
          <span class="preset-tag ${s.expected === 'NORMAL' ? 'tag-normal' : 'tag-lame'}">${s.expected}</span>
        `;
        chip.addEventListener('click', () => selectPreset(s, chip));
        presetContainer.appendChild(chip);
      });
    })
    .catch(() => {
      presetContainer.innerHTML = '<span style="font-size:12px;color:#94a3b8;">Preset evaluation samples ready upon server boot.</span>';
    });

  function selectPreset(sample, chipElement) {
    document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
    chipElement.classList.add('active');

    currentVideoFile = null;
    currentSamplePath = sample.path;

    videoFilename.textContent = sample.filename;
    videoPlayer.src = `/video/${encodeURIComponent(sample.filename)}`;
    videoPreviewWrapper.style.display = 'flex';
    dropzone.style.display = 'none';
    videoPlayer.load();

    resetResults();
  }

  // 2. File Upload & Drag-and-Drop Handling
  browseBtn.addEventListener('click', () => videoFileInput.click());
  if (changeVideoBtn) {
    changeVideoBtn.addEventListener('click', () => {
      videoFileInput.value = '';
      currentVideoFile = null;
      currentSamplePath = null;
      videoPreviewWrapper.style.display = 'none';
      dropzone.style.display = 'block';
      document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
      resetResults();
    });
  }

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  videoFileInput.addEventListener('change', () => {
    if (videoFileInput.files.length > 0) {
      handleFileSelected(videoFileInput.files[0]);
    }
  });

  function handleFileSelected(file) {
    document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
    currentVideoFile = file;
    currentSamplePath = null;

    videoFilename.textContent = file.name;
    const url = URL.createObjectURL(file);
    videoPlayer.src = url;
    videoPreviewWrapper.style.display = 'flex';
    dropzone.style.display = 'none';
    videoPlayer.load();

    resetResults();
  }

  videoPlayer.addEventListener('loadedmetadata', () => {
    videoDurationBadge.textContent = `${videoPlayer.duration.toFixed(1)}s`;
  });

  function resetResults() {
    loadingState.style.display = 'none';
    resultsContent.style.display = 'none';
    scanOverlay.style.display = 'none';
  }

  // 3. Execution / Screening Call
  analyzeBtn.addEventListener('click', async () => {
    if (!currentVideoFile && !currentSamplePath) return;

    // Transition to loading state
    loadingState.style.display = 'flex';
    resultsContent.style.display = 'none';
    scanOverlay.style.display = 'block';
    analyzeBtn.disabled = true;

    // Scroll smoothly to loading indicator
    loadingState.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    const startTime = performance.now();

    try {
      let res;
      if (currentVideoFile) {
        const formData = new FormData();
        formData.append('video', currentVideoFile);
        res = await fetch('/api/screen', { method: 'POST', body: formData });
      } else {
        res = await fetch('/api/screen', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ video_path: currentSamplePath })
        });
      }

      if (!res.ok) {
        throw new Error(`Server returned HTTP status ${res.status}`);
      }

      const data = await res.json();
      const elapsed = Math.round(performance.now() - startTime);

      if (data.error) {
        alert(`Screening Notice: ${data.error}`);
        resetResults();
      } else {
        renderResults(data, elapsed);
      }
    } catch (err) {
      alert(`Screening service unavailable: ${err.message}. Please verify the backend is running.`);
      resetResults();
    } finally {
      loadingState.style.display = 'none';
      scanOverlay.style.display = 'none';
      analyzeBtn.disabled = false;
    }
  });

  // 4. Render Live Dual-Branch Results
  function renderResults(data, totalElapsedMs) {
    resultsContent.style.display = 'flex';

    // Header metadata
    const fname = data.video_file || (currentVideoFile ? currentVideoFile.name : (currentSamplePath ? currentSamplePath.split('\\').pop().split('/').pop() : 'cattle_video.mp4'));
    metaFilename.textContent = fname;
    metaLatency.textContent = `${totalElapsedMs} ms`;
    metaDevice.textContent = data.device || 'CUDA:0';

    // ----------------------------------------------------
    // Branch 1: Lameness Screening Render
    // ----------------------------------------------------
    const lameness = data.lameness || {};
    const lamenessStatus = lameness.status || data.status || 'UNKNOWN';
    const lamenessProb = (lameness.lameness_probability !== undefined) ? lameness.lameness_probability : (data.lameness_probability || 0.0);
    const lamenessConf = (lameness.confidence_percentage !== undefined) ? lameness.confidence_percentage : (data.confidence_percentage || 0.0);

    statusTitle.textContent = lamenessStatus;
    riskProbVal.textContent = lamenessProb.toFixed(3);
    confidenceVal.textContent = `${Math.round(lamenessConf)}%`;
    explanationText.textContent = lameness.explanation || data.explanation || 'Temporal gait cycle evaluated across uniform 16-frame sequence.';

    // Risk Meter Width
    const fillPercent = Math.min(Math.max(lamenessProb * 100, 3), 100);
    riskMeterFill.style.width = `${fillPercent}%`;

    // Status Pill & Advisory Theming
    if (lamenessStatus === 'CHECK') {
      statusBadge.textContent = 'HIGH RISK (ACTION ADVISED)';
      statusBadge.style.background = 'var(--status-check-bg)';
      statusBadge.style.color = 'var(--status-check-text)';
      statusBadge.style.border = '1px solid var(--status-check-border)';
      
      actionCard.style.borderColor = 'var(--status-check-border)';
      actionCard.style.background = 'var(--status-check-bg)';
      actionBadge.style.color = 'var(--status-check-text)';
      actionText.textContent = 'Flagged for clinical inspection. Schedule hoof trimmer or veterinary gait exam within 24–48 hours to prevent severe claw lesion progression.';
    } else if (lamenessStatus === 'WATCH') {
      statusBadge.textContent = 'MODERATE RISK (WATCH LIST)';
      statusBadge.style.background = 'var(--status-watch-bg)';
      statusBadge.style.color = 'var(--status-watch-text)';
      statusBadge.style.border = '1px solid var(--status-watch-border)';

      actionCard.style.borderColor = 'var(--status-watch-border)';
      actionCard.style.background = 'var(--status-watch-bg)';
      actionBadge.style.color = 'var(--status-watch-text)';
      actionText.textContent = 'Place cow on automated watch list. Re-screen during subsequent parlor exit passes. Inspect stall comfort and walking surface.';
    } else {
      statusBadge.textContent = 'LOW RISK (CLEAR)';
      statusBadge.style.background = 'var(--status-normal-bg)';
      statusBadge.style.color = 'var(--status-normal-text)';
      statusBadge.style.border = '1px solid var(--status-normal-border)';

      actionCard.style.borderColor = 'var(--primary-200)';
      actionCard.style.background = 'var(--primary-50)';
      actionBadge.style.color = 'var(--primary-dark)';
      actionText.textContent = 'No clinical hoof-trimming or separation required. Locomotion cadence is balanced and symmetrical. Cow cleared for standard herd rotation.';
    }

    // ----------------------------------------------------
    // Branch 2: Visual Udder Screening Render
    // ----------------------------------------------------
    const udder = data.visual_udder_screening || data.udder_visual || data.visual_udder_analysis || {};
    
    if (udder.status === 'INSUFFICIENT_VISUAL_DATA') {
      // Graceful No-Usable-Frames state
      udderStatusTitle.textContent = 'INSUFFICIENT DATA';
      udderStatusBadge.textContent = 'NO CLEAR UDDER FRAMES';
      udderStatusBadge.style.background = '#f1f5f9';
      udderStatusBadge.style.color = '#64748b';
      udderStatusBadge.style.border = '1px solid #cbd5e1';

      udderConfidenceVal.textContent = '--%';
      udderAbnormalProbVal.textContent = 'N/A';
      udderMeterFill.style.width = '0%';

      udderExplanationText.textContent = udder.message || 'Not enough usable udder-visible frames met optical clarity thresholds for screening.';
      udderUsableFrames.textContent = '0 frames';
      udderErythema.textContent = '--';
      udderAsymmetry.textContent = '--';
      udderRoughness.textContent = '--';

      sumUdderStatus.textContent = 'INSUFFICIENT DATA';
    } else if (udder.abnormal_probability !== null && udder.abnormal_probability !== undefined) {
      // Successful visual inference
      const uClass = udder.class || 'UNKNOWN';
      const abnProb = udder.abnormal_probability;
      const uConf = udder.confidence_percentage || Math.round(Math.max(abnProb, 1 - abnProb) * 100);

      udderStatusTitle.textContent = uClass === 'NORMAL' ? 'NORMAL' : (uClass === 'VISUALLY_ABNORMAL' ? 'VISUALLY ABNORMAL' : 'INCONCLUSIVE');
      udderAbnormalProbVal.textContent = abnProb.toFixed(3);
      udderConfidenceVal.textContent = `${Math.round(uConf)}%`;

      const uFill = Math.min(Math.max(abnProb * 100, 3), 100);
      udderMeterFill.style.width = `${uFill}%`;

      if (uClass === 'VISUALLY_ABNORMAL') {
        udderStatusBadge.textContent = 'HIGH VISUAL ANOMALY';
        udderStatusBadge.style.background = 'var(--status-purple-bg)';
        udderStatusBadge.style.color = 'var(--status-purple-text)';
        udderStatusBadge.style.border = '1px solid var(--status-purple-border)';
      } else if (uClass === 'NORMAL') {
        udderStatusBadge.textContent = 'LOW RISK (CLEAR)';
        udderStatusBadge.style.background = 'var(--status-normal-bg)';
        udderStatusBadge.style.color = 'var(--status-normal-text)';
        udderStatusBadge.style.border = '1px solid var(--status-normal-border)';
      } else {
        udderStatusBadge.textContent = 'MODERATE ANOMALY (WATCH)';
        udderStatusBadge.style.background = 'var(--status-watch-bg)';
        udderStatusBadge.style.color = 'var(--status-watch-text)';
        udderStatusBadge.style.border = '1px solid var(--status-watch-border)';
      }

      udderExplanationText.textContent = udder.explanation || 'Visual screening benchmark evaluated across quality-filtered candidate keyframes.';
      udderUsableFrames.textContent = `${udder.frames_analyzed || 0} frames`;

      const opt = udder.optical_indicators || {};
      udderErythema.textContent = (opt.mean_erythema_index !== undefined) ? opt.mean_erythema_index.toFixed(3) : '--';
      udderAsymmetry.textContent = (opt.mean_asymmetry_index !== undefined) ? opt.mean_asymmetry_index.toFixed(3) : '--';
      udderRoughness.textContent = (opt.mean_texture_roughness !== undefined) ? opt.mean_texture_roughness.toFixed(3) : '--';

      sumUdderStatus.textContent = udderStatusTitle.textContent;
    } else {
      // Pending / Uninitialized
      udderStatusTitle.textContent = 'PENDING';
      udderStatusBadge.textContent = 'AWAITING VIDEO';
      udderConfidenceVal.textContent = '--%';
      udderAbnormalProbVal.textContent = '--';
      udderMeterFill.style.width = '0%';
      udderUsableFrames.textContent = '--';
      udderErythema.textContent = '--';
      udderAsymmetry.textContent = '--';
      udderRoughness.textContent = '--';
      sumUdderStatus.textContent = 'PENDING';
    }

    // Summary Strip
    sumLamenessStatus.textContent = lamenessStatus;

    // Technical Details Accordion
    detLamenessTime.textContent = `${lameness.inference_time_ms || data.inference_time_ms || '--'} ms`;
    detUdderTime.textContent = `${udder.processing_time_ms || '--'} ms`;

    // Smooth scroll into results view
    resultsContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
});
