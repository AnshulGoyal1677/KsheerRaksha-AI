document.addEventListener('DOMContentLoaded', () => {
  const presetContainer = document.getElementById('presetContainer');
  const dropzone = document.getElementById('dropzone');
  const videoFileInput = document.getElementById('videoFileInput');
  const browseBtn = document.getElementById('browseBtn');
  const videoPreviewWrapper = document.getElementById('videoPreviewWrapper');
  const videoPlayer = document.getElementById('videoPlayer');
  const videoFilename = document.getElementById('videoFilename');
  const videoDurationBadge = document.getElementById('videoDurationBadge');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const scanOverlay = document.getElementById('scanOverlay');

  const placeholderState = document.getElementById('placeholderState');
  const loadingState = document.getElementById('loadingState');
  const resultsContent = document.getElementById('resultsContent');

  const statusBanner = document.getElementById('statusBanner');
  const statusTitle = document.getElementById('statusTitle');
  const statusBadge = document.getElementById('statusBadge');
  const confidenceVal = document.getElementById('confidenceVal');
  const riskProbVal = document.getElementById('riskProbVal');
  const riskMeterFill = document.getElementById('riskMeterFill');
  const explanationText = document.getElementById('explanationText');
  const inferenceTime = document.getElementById('inferenceTime');
  const actionCard = document.getElementById('actionCard');
  const actionText = document.getElementById('actionText');

  // Udder Stream Elements
  const udderProcessingTime = document.getElementById('udderProcessingTime');
  const udderStatusBanner = document.getElementById('udderStatusBanner');
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

  let currentVideoFile = null;
  let currentSamplePath = null;

  // 1. Fetch and render evaluation sample buttons
  fetch('/api/samples')
    .then(res => res.json())
    .then(samples => {
      presetContainer.innerHTML = '';
      samples.forEach((s, idx) => {
        const btn = document.createElement('button');
        btn.className = 'preset-btn';
        btn.innerHTML = `
          <span>${s.name}</span>
          <span class="preset-tag ${s.expected === 'NORMAL' ? 'tag-normal' : 'tag-lame'}">${s.expected}</span>
        `;
        btn.addEventListener('click', () => selectPreset(s, btn));
        presetContainer.appendChild(btn);
      });
    })
    .catch(err => {
      presetContainer.innerHTML = '<span style="font-size:12px;color:#94a3b8;">Samples ready upon server boot</span>';
    });

  function selectPreset(sample, btnElement) {
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    btnElement.classList.add('active');

    currentVideoFile = null;
    currentSamplePath = sample.path;

    videoFilename.textContent = sample.filename;
    videoPlayer.src = `/video/${encodeURIComponent(sample.filename)}`;
    videoPreviewWrapper.style.display = 'flex';
    videoPlayer.load();

    resetResultsView();
  }

  // 2. File Upload handling
  browseBtn.addEventListener('click', () => videoFileInput.click());
  dropzone.addEventListener('click', (e) => {
    if (e.target !== browseBtn) videoFileInput.click();
  });

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
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    currentVideoFile = file;
    currentSamplePath = null;

    videoFilename.textContent = file.name;
    const url = URL.createObjectURL(file);
    videoPlayer.src = url;
    videoPreviewWrapper.style.display = 'flex';
    videoPlayer.load();

    resetResultsView();
  }

  videoPlayer.addEventListener('loadedmetadata', () => {
    videoDurationBadge.textContent = `${videoPlayer.duration.toFixed(1)}s`;
  });

  function resetResultsView() {
    placeholderState.style.display = 'flex';
    loadingState.style.display = 'none';
    resultsContent.style.display = 'none';
  }

  // 3. Analyze Execution
  analyzeBtn.addEventListener('click', async () => {
    if (!currentVideoFile && !currentSamplePath) return;

    placeholderState.style.display = 'none';
    loadingState.style.display = 'flex';
    resultsContent.style.display = 'none';
    scanOverlay.style.display = 'block';
    analyzeBtn.disabled = true;

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

      const data = await res.json();
      const elapsed = Math.round(performance.now() - startTime);

      if (data.error) {
        alert(`Screening Error: ${data.error}`);
        resetResultsView();
      } else {
        renderResults(data, elapsed);
      }
    } catch (err) {
      alert(`Network/Inference failure: ${err.message}`);
      resetResultsView();
    } finally {
      loadingState.style.display = 'none';
      scanOverlay.style.display = 'none';
      analyzeBtn.disabled = false;
    }
  });

  // 4. Render Results
  function renderResults(data, elapsedMs) {
    resultsContent.style.display = 'flex';

    statusBanner.className = 'status-banner';
    statusTitle.textContent = data.status;

    if (data.status === 'CHECK') {
      statusBanner.classList.add('status-check');
      statusBadge.textContent = 'HIGH RISK (ACTION ADVISED)';
      statusBadge.style.color = '#f87171';
      actionCard.style.borderColor = 'rgba(239, 68, 68, 0.4)';
      actionCard.style.background = 'rgba(239, 68, 68, 0.08)';
      actionText.textContent = 'Flagged for clinical inspection. Schedule hoof trimmer or veterinary gait exam within 24–48 hours to prevent severe claw lesion progression.';
    } else if (data.status === 'WATCH') {
      statusBanner.classList.add('status-watch');
      statusBadge.textContent = 'MODERATE RISK (WATCH LIST)';
      statusBadge.style.color = '#fbbf24';
      actionCard.style.borderColor = 'rgba(245, 158, 11, 0.4)';
      actionCard.style.background = 'rgba(245, 158, 11, 0.08)';
      actionText.textContent = 'Place cow on automated watch list. Re-screen during subsequent parlor exit passes. Inspect stall comfort and walking surface.';
    } else {
      statusBadge.textContent = 'LOW RISK (CLEAR)';
      statusBadge.style.color = '#34d399';
      actionCard.style.borderColor = 'rgba(16, 185, 129, 0.3)';
      actionCard.style.background = 'rgba(16, 185, 129, 0.08)';
      actionText.textContent = 'No clinical hoof-trimming or separation required. Locomotion is balanced and symmetrical. Cow cleared for standard herd rotation.';
    }

    confidenceVal.textContent = `${Math.round(data.confidence_percentage)}%`;
    riskProbVal.textContent = data.lameness_probability.toFixed(3);

    const fillPercent = Math.min(Math.max(data.lameness_probability * 100, 4), 100);
    riskMeterFill.style.width = `${fillPercent}%`;

    explanationText.textContent = data.explanation;
    inferenceTime.textContent = `${elapsedMs} ms`;
    normalProb.textContent = data.normal_probability.toFixed(3);

    // Render Stream 2: Visual Udder Screening
    const udderData = data.visual_udder_screening || data.udder_visual || data.visual_udder_analysis;
    if (udderData && udderData.status !== 'INSUFFICIENT_VISUAL_DATA' && udderData.abnormal_probability !== null && udderData.abnormal_probability !== undefined) {
      udderProcessingTime.textContent = `${udderData.processing_time_ms || '--'} ms`;
      udderStatusTitle.textContent = udderData.class === 'NORMAL' ? 'NORMAL' : (udderData.class === 'VISUALLY_ABNORMAL' ? 'VISUALLY ABNORMAL' : 'INCONCLUSIVE');
      
      udderStatusBanner.className = 'status-banner udder-status-banner';
      if (udderData.class === 'VISUALLY_ABNORMAL') {
        udderStatusBanner.classList.add('status-abnormal');
        udderStatusBadge.textContent = 'HIGH VISUAL ANOMALY';
        udderStatusBadge.style.color = '#c084fc';
      } else if (udderData.class === 'NORMAL') {
        udderStatusBadge.textContent = 'LOW RISK (CLEAR)';
        udderStatusBadge.style.color = '#34d399';
      } else {
        udderStatusBanner.classList.add('status-inconclusive');
        udderStatusBadge.textContent = 'MODERATE ANOMALY (WATCH)';
        udderStatusBadge.style.color = '#fbbf24';
      }
      
      const abnProb = udderData.abnormal_probability;
      udderAbnormalProbVal.textContent = abnProb.toFixed(3);
      const fillPct = Math.min(Math.max(abnProb * 100, 4), 100);
      udderMeterFill.style.width = `${fillPct}%`;
      
      udderConfidenceVal.textContent = `${Math.round(udderData.confidence_percentage || (Math.max(abnProb, 1-abnProb)*100))}%`;
      udderExplanationText.textContent = udderData.explanation || 'Visual screening completed across extracted candidate frames.';
      udderUsableFrames.textContent = `${udderData.frames_analyzed || 0} frames`;
      
      const opt = udderData.optical_indicators || {};
      udderErythema.textContent = (opt.mean_erythema_index !== undefined) ? opt.mean_erythema_index.toFixed(3) : '--';
      udderAsymmetry.textContent = (opt.mean_asymmetry_index !== undefined) ? opt.mean_asymmetry_index.toFixed(3) : '--';
      udderRoughness.textContent = (opt.mean_texture_roughness !== undefined) ? opt.mean_texture_roughness.toFixed(3) : '--';
    } else if (udderData && udderData.status === 'INSUFFICIENT_VISUAL_DATA') {
      udderProcessingTime.textContent = `${udderData.processing_time_ms || '--'} ms`;
      udderStatusBanner.className = 'status-banner udder-status-banner';
      udderStatusTitle.textContent = 'INSUFFICIENT DATA';
      udderStatusBadge.textContent = 'NO CLEAR UDDER FRAMES';
      udderStatusBadge.style.color = '#94a3b8';
      udderAbnormalProbVal.textContent = 'N/A';
      udderMeterFill.style.width = '0%';
      udderConfidenceVal.textContent = '--%';
      udderExplanationText.textContent = udderData.message || 'No unoccluded udder keyframes met optical clarity thresholds.';
      udderUsableFrames.textContent = '0 frames';
      udderErythema.textContent = '--';
      udderAsymmetry.textContent = '--';
      udderRoughness.textContent = '--';
    } else {
      udderProcessingTime.textContent = '-- ms';
      udderStatusBanner.className = 'status-banner udder-status-banner';
      udderStatusTitle.textContent = 'PENDING ANALYSIS';
      udderStatusBadge.textContent = 'AWAITING VIDEO';
      udderAbnormalProbVal.textContent = '--';
      udderMeterFill.style.width = '0%';
      udderConfidenceVal.textContent = '--%';
      udderUsableFrames.textContent = '--';
      udderErythema.textContent = '--';
      udderAsymmetry.textContent = '--';
      udderRoughness.textContent = '--';
    }
  }
});
