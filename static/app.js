/* ── ArcTrace – app.js ── */
'use strict';

// ── DOM refs ──
const form          = document.querySelector('#analysis-form');
const refInput      = document.querySelector('#reference');
const refDrop       = document.querySelector('.reference-drop');
const refPreview    = document.querySelector('#reference-preview');
const videosInput   = document.querySelector('#videos');
const fileList      = document.querySelector('#file-list');
const thresholdEl   = document.querySelector('#threshold');
const thresholdVal  = document.querySelector('#threshold-value');
const sampleEvery   = document.querySelector('#sample-every');
const sampleVal     = document.querySelector('#sample-value');
const detConfEl     = document.querySelector('#det-conf');
const detConfVal    = document.querySelector('#det-conf-value');
const progressSec   = document.querySelector('#progress');
const progressTitle = document.querySelector('#progress-title');
const progressText  = document.querySelector('#progress-text');
const progressBar   = document.querySelector('#progress-bar');
const liveThumbs    = document.querySelector('#live-thumbs');
const errorSec      = document.querySelector('#error');
const resultsSec    = document.querySelector('#results');
const analyzeBtn    = document.querySelector('#analyze-button');
const statusDot     = document.querySelector('#status-dot');
const statusText    = document.querySelector('#status-text');

// ── Sliders ──
thresholdEl.addEventListener('input', () => {
  thresholdVal.textContent = parseFloat(thresholdEl.value).toFixed(2);
});
sampleEvery.addEventListener('input', () => {
  const v = sampleEvery.value;
  sampleVal.textContent = `${v} frame${v === '1' ? '' : 's'}`;
});
detConfEl.addEventListener('input', () => {
  detConfVal.textContent = parseFloat(detConfEl.value).toFixed(2);
});

// ── Reference image preview ──
refInput.addEventListener('change', () => {
  const file = refInput.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  refDrop.style.backgroundImage = `url(${url})`;
  refDrop.classList.add('has-image');
  refPreview.innerHTML = '<strong>Reference loaded · click to replace</strong>';
});

// Drag-and-drop highlight
['dragover', 'dragenter'].forEach(ev => {
  document.querySelectorAll('.dropzone').forEach(el => {
    el.addEventListener(ev, e => { e.preventDefault(); el.style.borderColor = 'var(--rust)'; });
  });
});
['dragleave', 'drop'].forEach(ev => {
  document.querySelectorAll('.dropzone').forEach(el => {
    el.addEventListener(ev, () => { el.style.borderColor = ''; });
  });
});

// ── Video file list ──
videosInput.addEventListener('change', () => {
  const files = [...videosInput.files];
  fileList.innerHTML = files.length
    ? files.map(f => `<div class="file-item"><span>${esc(f.name)}</span><span>${(f.size / 1024 / 1024).toFixed(1)} MB</span></div>`).join('')
    : '<span>No footage selected</span>';
});

// ── Utility ──
function esc(str) {
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function fmt(seconds) {
  const m = Math.floor(seconds / 60), s = seconds % 60;
  return `${String(m).padStart(2,'0')}:${s.toFixed(2).padStart(5,'0')}`;
}
function setStatus(state, label) {
  statusDot.className = 'status-dot ' + (state || '');
  statusText.textContent = label;
}
function setProgress(pct, title, detail) {
  progressBar.style.width = `${Math.round(pct * 100)}%`;
  if (title)  progressTitle.textContent = title;
  if (detail) progressText.textContent  = detail;
}

// ── Form submit ──
form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const name    = document.querySelector('#character-name').value.trim();
  const refFile = refInput.files[0];
  const vidFiles= videosInput.files;

  if (!name || !refFile || !vidFiles.length) {
    showError('Please fill in the character name, reference photo, and at least one video.');
    return;
  }

  // Reset UI
  errorSec.classList.add('hidden');
  resultsSec.classList.add('hidden');
  progressSec.classList.remove('hidden');
  liveThumbs.innerHTML = '';
  progressBar.style.width = '0%';
  analyzeBtn.disabled = true;
  setStatus('busy', 'ANALYZING');
  setProgress(0, 'Starting analysis…', 'Building ArcFace embedding for reference identity…');

  // Build FormData – manually handle checkbox so unchecked still sends 0
  const body = new FormData(form);
  const genderCheckbox = document.querySelector('#use-gender');
  // FormData only includes checked checkboxes; force-set correct value
  body.set('use_gender', genderCheckbox.checked ? '1' : '0');

  try {
    const res = await fetch('/api/analyze/stream', { method: 'POST', body });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      throw new Error(j.detail || `Server error ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let totalVideos = 1;
    let doneVideos = 0;
    let finalData = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop(); // keep incomplete chunk

      for (const chunk of lines) {
        if (!chunk.trim()) continue;
        const eventMatch = chunk.match(/^event:\s*(\S+)/m);
        const dataMatch  = chunk.match(/^data:\s*(.+)/ms);
        if (!dataMatch) continue;

        const event = eventMatch ? eventMatch[1] : 'message';
        let payload;
        try { payload = JSON.parse(dataMatch[1].trim()); } catch { continue; }

        switch (event) {
          case 'start':
            totalVideos = payload.videoCount || 1;
            setProgress(.05, `Searching for ${esc(payload.characterName)}…`, 'Reference embedding ready.');
            break;

          case 'video_start':
            setProgress(
              .1 + (doneVideos / totalVideos) * .85,
              `Scanning video ${payload.index + 1} / ${totalVideos}`,
              esc(payload.name)
            );
            break;

          case 'video_progress': {
            const frac = payload.totalFrames ? payload.framesScanned / payload.totalFrames : 0;
            setProgress(
              .1 + ((doneVideos + frac) / totalVideos) * .85,
              `Scanning video ${payload.index + 1} / ${totalVideos}`,
              `${esc(payload.name)} — ${Math.round(frac * 100)}% (${payload.framesScanned}/${payload.totalFrames} frames)`
            );
            break;
          }

          case 'video_done':
            doneVideos++;
            setProgress(
              .1 + (doneVideos / totalVideos) * .85,
              `Scanned ${doneVideos} / ${totalVideos} videos`,
              `${payload.matchCount} match${payload.matchCount !== 1 ? 'es' : ''} found in ${esc(payload.name)}`
            );
            // Show live thumbnails
            (payload.thumbnails || []).forEach(thumb => {
              if (!thumb.thumbnail) return;
              const div = document.createElement('div');
              div.className = 'live-thumb';
              div.innerHTML = `<img src="data:image/jpeg;base64,${thumb.thumbnail}" loading="lazy" alt="${thumb.time}">
                               <div class="thumb-score">${thumb.time}</div>`;
              liveThumbs.appendChild(div);
            });
            break;

          case 'video_error':
            showError(`Video ${esc(payload.name)}: ${esc(payload.message)}`);
            break;

          case 'done':
            finalData = payload;
            setProgress(1, 'Complete!', `${payload.totalMatches} total match${payload.totalMatches !== 1 ? 'es' : ''} found.`);
            break;

          case 'error':
            throw new Error(payload.message);
        }
      }
    }

    if (finalData) {
      setTimeout(() => {
        progressSec.classList.add('hidden');
        renderResults(finalData);
        setStatus('done', 'DONE');
      }, 600);
    }

  } catch (err) {
    showError(err.message || 'Analysis failed. Check the server logs.');
    setStatus('', 'READY');
  } finally {
    analyzeBtn.disabled = false;
  }
});

// ── Error display ──
function showError(msg) {
  errorSec.textContent = '⚠ ' + msg;
  errorSec.classList.remove('hidden');
  progressSec.classList.add('hidden');
}

// ── Results renderer ──
function renderResults(data) {
  const totalSec = data.videos.reduce((a, v) => a + v.duration, 0);
  const totalFrames = data.videos.reduce((a, v) => a + v.framesScanned, 0);

  const statsHtml = `
    <div class="stats-row">
      <div class="stat-box"><div class="stat-value">${data.totalMatches}</div><div class="stat-label">MATCH POINTS</div></div>
      <div class="stat-box"><div class="stat-value">${data.videos.length}</div><div class="stat-label">VIDEOS SCANNED</div></div>
      <div class="stat-box"><div class="stat-value">${totalFrames.toLocaleString()}</div><div class="stat-label">FRAMES CHECKED</div></div>
      <div class="stat-box"><div class="stat-value">${fmtDur(totalSec)}</div><div class="stat-label">TOTAL DURATION</div></div>
    </div>`;

  const cards = data.videos.map(renderVideoCard).join('');

  resultsSec.innerHTML = `
    <div class="result-header">
      <div>
        <p class="eyebrow">SEARCH COMPLETE</p>
        <h2>${esc(data.characterName)}</h2>
      </div>
      <div class="meta">Threshold ${data.threshold ?? 0.42} · Every ${data.sampleEvery ?? 5} frames</div>
    </div>
    ${statsHtml}
    ${cards}`;

  resultsSec.classList.remove('hidden');
  resultsSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderVideoCard(video) {
  const matchBadge = `<span class="badge-count">${video.matchCount} match${video.matchCount !== 1 ? 'es' : ''}</span>`;

  const chipsHtml = video.detections.length
    ? `<div class="chips">${video.detections.map(h => `<span class="chip" title="${h.score}">${h.time} · ${Math.round(h.score * 100)}%</span>`).join('')}</div>`
    : '<p class="empty">No confident matches found in sampled frames.</p>';

  const thumbsHtml = (video.thumbnails || []).filter(t => t.thumbnail).map(t => `
    <div class="gallery-thumb" title="${t.time} — ${Math.round(t.score * 100)}%">
      <img src="data:image/jpeg;base64,${t.thumbnail}" loading="lazy" alt="${t.time}">
      <div class="gt-overlay">${t.time}</div>
    </div>`).join('');

  const gallerySection = thumbsHtml
    ? `<div class="thumb-gallery">${thumbsHtml}</div>`
    : '';

  return `
    <article class="result-card">
      <div class="result-card-head">
        <div>
          <h3>${esc(video.name)}</h3>
          <div class="card-meta">${fmtDur(video.duration)} · ${video.fps} fps · ${video.framesScanned.toLocaleString()} frames scanned</div>
        </div>
        <div class="result-card-actions">
          ${matchBadge}
          <a class="download-btn" href="${esc(video.videoUrl)}" download>↓ DOWNLOAD MARKED VIDEO</a>
        </div>
      </div>
      <div class="timeline-section">
        <p class="timeline-label">MATCH TIMESTAMPS</p>
        ${chipsHtml}
      </div>
      ${gallerySection}
    </article>`;
}

function fmtDur(sec) {
  if (sec < 60) return `${Math.round(sec)}s`;
  const m = Math.floor(sec / 60), s = Math.round(sec % 60);
  return `${m}m ${s}s`;
}
