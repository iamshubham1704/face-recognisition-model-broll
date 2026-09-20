/* ── ArcTrace – app.js ── */
'use strict';

// ── DOM refs ──
const form            = document.querySelector('#analysis-form');
const characterList   = document.querySelector('#character-list');
const addCharacterBtn = document.querySelector('#add-character-btn');
const videosInput     = document.querySelector('#videos');
const fileList        = document.querySelector('#file-list');
const thresholdEl     = document.querySelector('#threshold');
const thresholdVal    = document.querySelector('#threshold-value');
const scanFpsEl       = document.querySelector('#scan-fps');
const scanFpsVal      = document.querySelector('#scan-fps-value');
const detConfEl       = document.querySelector('#det-conf');
const detConfVal      = document.querySelector('#det-conf-value');
const progressSec     = document.querySelector('#progress');
const progressTitle   = document.querySelector('#progress-title');
const progressText    = document.querySelector('#progress-text');
const progressBar     = document.querySelector('#progress-bar');
const liveThumbs      = document.querySelector('#live-thumbs');
const errorSec        = document.querySelector('#error');
const resultsSec      = document.querySelector('#results');
const analyzeBtn      = document.querySelector('#analyze-button');
const statusDot       = document.querySelector('#status-dot');
const statusText      = document.querySelector('#status-text');

// ── Sliders ──
thresholdEl.addEventListener('input', () => {
  thresholdVal.textContent = parseFloat(thresholdEl.value).toFixed(2);
});
scanFpsEl.addEventListener('input', () => {
  scanFpsVal.textContent = `${scanFpsEl.value} fps`;
});
detConfEl.addEventListener('input', () => {
  detConfVal.textContent = parseFloat(detConfEl.value).toFixed(2);
});

// ── Multi-character reference list ──
function wireCharacterRow(row) {
  const photoInput = row.querySelector('input[type=file]');
  const photoDrop  = row.querySelector('.char-photo-drop');
  const removeBtn  = row.querySelector('.remove-character-btn');

  photoInput.addEventListener('change', () => {
    const file = photoInput.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    photoDrop.style.backgroundImage = `url(${url})`;
    photoDrop.classList.add('has-image');
  });

  ['dragover', 'dragenter'].forEach(ev => {
    photoDrop.addEventListener(ev, e => { e.preventDefault(); photoDrop.style.borderColor = 'var(--rust)'; });
  });
  ['dragleave', 'drop'].forEach(ev => {
    photoDrop.addEventListener(ev, () => { photoDrop.style.borderColor = ''; });
  });

  removeBtn.addEventListener('click', () => {
    if (document.querySelectorAll('.character-row').length <= 1) return;
    row.remove();
    updateRemoveButtons();
  });
}

function updateRemoveButtons() {
  const rows = document.querySelectorAll('.character-row');
  rows.forEach(row => {
    row.querySelector('.remove-character-btn').classList.toggle('hidden', rows.length <= 1);
  });
}

// Wire the initial row present in the HTML.
document.querySelectorAll('.character-row').forEach(wireCharacterRow);

addCharacterBtn.addEventListener('click', () => {
  const template = document.querySelector('.character-row');
  const row = template.cloneNode(true);
  row.querySelector('input[type=file]').value = '';
  row.querySelector('.char-name-input').value = '';
  row.querySelector('.char-photo-drop').classList.remove('has-image');
  row.querySelector('.char-photo-drop').style.backgroundImage = '';
  characterList.appendChild(row);
  wireCharacterRow(row);
  updateRemoveButtons();
});

// Drag-and-drop highlight for the video dropzone.
['dragover', 'dragenter'].forEach(ev => {
  document.querySelectorAll('.video-drop').forEach(el => {
    el.addEventListener(ev, e => { e.preventDefault(); el.style.borderColor = 'var(--rust)'; });
  });
});
['dragleave', 'drop'].forEach(ev => {
  document.querySelectorAll('.video-drop').forEach(el => {
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

  const rows     = [...document.querySelectorAll('.character-row')];
  const vidFiles = videosInput.files;
  const incompleteRow = rows.find(row =>
    !row.querySelector('input[type=file]').files[0] || !row.querySelector('.char-name-input').value.trim());

  if (incompleteRow || !vidFiles.length) {
    showError('Please fill in a photo and name for every character row, and add at least one video.');
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
  setProgress(0, 'Starting analysis…', 'Building ArcFace embeddings for reference identities…');

  // FormData auto-collects every `references` file input and `character_names`
  // text input in DOM order (multiple same-name inputs, not a single multi-file
  // input), so row N's photo pairs with row N's name without any manual work here.
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
          case 'start': {
            totalVideos = payload.videoCount || 1;
            const names = (payload.characterNames || []).join(', ');
            setProgress(.05, `Searching for ${esc(names)}…`, 'Building reference embeddings.');
            break;
          }

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

          case 'video_done': {
            doneVideos++;
            setProgress(
              .1 + (doneVideos / totalVideos) * .85,
              `Scanned ${doneVideos} / ${totalVideos} videos`,
              `${payload.matchCount} match${payload.matchCount !== 1 ? 'es' : ''} found in ${esc(payload.name)}`
            );
            // Show live thumbnails — one per character that had a match in this video
            (payload.characters || []).forEach(char => {
              (char.thumbnails || []).forEach(thumb => {
                if (!thumb.thumbnail) return;
                const div = document.createElement('div');
                div.className = 'live-thumb';
                div.innerHTML = `<img src="data:image/png;base64,${thumb.thumbnail}" loading="lazy" alt="${char.name} @ ${thumb.time}">
                                 <div class="thumb-score">${esc(char.name)} · ${thumb.time}</div>`;
                liveThumbs.appendChild(div);
              });
            });
            break;
          }

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
  const names = (data.characterNames || []).join(', ');

  const statsHtml = `
    <div class="stats-row">
      <div class="stat-box"><div class="stat-value">${data.totalMatches}</div><div class="stat-label">MATCH POINTS</div></div>
      <div class="stat-box"><div class="stat-value">${(data.characterNames || []).length}</div><div class="stat-label">CHARACTERS</div></div>
      <div class="stat-box"><div class="stat-value">${data.videos.length}</div><div class="stat-label">VIDEOS SCANNED</div></div>
      <div class="stat-box"><div class="stat-value">${totalFrames.toLocaleString()}</div><div class="stat-label">FRAMES CHECKED</div></div>
      <div class="stat-box"><div class="stat-value">${fmtDur(totalSec)}</div><div class="stat-label">TOTAL DURATION</div></div>
    </div>`;

  const cards = data.videos.map(renderVideoCard).join('');

  resultsSec.innerHTML = `
    <div class="result-header">
      <div>
        <p class="eyebrow">SEARCH COMPLETE</p>
        <h2>${esc(names)}</h2>
      </div>
      <div class="meta">Threshold ${data.threshold ?? 0.50} · ${data.scanFps ?? 2} fps scan rate</div>
    </div>
    ${statsHtml}
    ${cards}`;

  resultsSec.classList.remove('hidden');
  resultsSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderVideoCard(video) {
  const matchBadge = `<span class="badge-count">${video.matchCount} match${video.matchCount !== 1 ? 'es' : ''}</span>`;
  const charactersHtml = (video.characters || []).map(renderCharacterSection).join('');

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
      ${charactersHtml}
    </article>`;
}

function renderCharacterSection(char) {
  const chipsHtml = char.detections.length
    ? `<div class="chips">${char.detections.map(h => `<span class="chip" title="${h.score}">${h.time} · ${Math.round(h.score * 100)}%</span>`).join('')}</div>`
    : '<p class="empty">No confident matches found in sampled frames.</p>';

  const thumbsHtml = (char.thumbnails || []).filter(t => t.thumbnail).map(t => `
    <div class="gallery-thumb" title="${char.name} — ${t.time} — ${Math.round(t.score * 100)}%">
      <img src="data:image/png;base64,${t.thumbnail}" loading="lazy" alt="${char.name} @ ${t.time}">
      <div class="gt-overlay">${esc(char.name)} · ${t.time}</div>
    </div>`).join('');

  const gallerySection = thumbsHtml
    ? `<div class="thumb-gallery">${thumbsHtml}</div>`
    : '';

  return `
    <div class="character-section">
      <div class="timeline-section">
        <p class="timeline-label">${esc(char.name).toUpperCase()} · ${char.matchCount} match${char.matchCount !== 1 ? 'es' : ''}</p>
        ${chipsHtml}
      </div>
      ${gallerySection}
    </div>`;
}

function fmtDur(sec) {
  if (sec < 60) return `${Math.round(sec)}s`;
  const m = Math.floor(sec / 60), s = Math.round(sec % 60);
  return `${m}m ${s}s`;
}
