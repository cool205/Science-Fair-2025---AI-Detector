console.log('QuackScan content script loaded');

// === Utility: Check if image is large and not an icon ===
function isLargeAndNotIcon(img) {
  const minWidth = 100;
  const minHeight = 100;
  const iconKeywords = ['icon', 'logo', 'profile', 'avatar', 'favicon'];
  const src = img.src?.toLowerCase() || '';
  const alt = img.alt?.toLowerCase() || '';
  const className = img.className?.toLowerCase() || '';

  if (img.naturalWidth < minWidth || img.naturalHeight < minHeight) return false;
  return !iconKeywords.some(kw => src.includes(kw) || alt.includes(kw) || className.includes(kw));
}

// === Create the top-right transparent popup ===
function createAIPopup() {
  const popup = document.createElement('div');
  popup.id = 'ai-confidence-popup';
  Object.assign(popup.style, {
    position: 'fixed',
    top: '12px',
    right: '12px',
    padding: '10px 16px',
    backgroundColor: 'rgba(229, 57, 53, 0.85)',
    color: '#fff',
    fontSize: '14px',
    fontFamily: 'Arial, sans-serif',
    borderRadius: '6px',
    zIndex: '9999',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
    backdropFilter: 'blur(4px)',
    transition: 'opacity 0.2s ease',
    opacity: '0',
    pointerEvents: 'none'
  });
  document.body.appendChild(popup);
}

function showAIPopup(confidence) {
  const popup = document.getElementById('ai-confidence-popup');
  if (popup && confidence > 70) {
    popup.textContent = `AI Confidence: ${confidence}%`;
    popup.style.opacity = '1';
  }
}

function hideAIPopup() {
  const popup = document.getElementById('ai-confidence-popup');
  if (popup) popup.style.opacity = '0';
}

// === Create saliency-style heatmap overlay ===
function createHeatmapOverlay(img, confidence) {
  const overlay = document.createElement('div');
  overlay.className = 'ai-heatmap-overlay';
  const intensity = Math.max(0, Math.min((confidence - 70) / 100, 0.3)) * 1.5;

  Object.assign(overlay.style, {
    position: 'absolute',
    top: `${img.offsetTop}px`,
    left: `${img.offsetLeft}px`,
    width: `${img.offsetWidth}px`,
    height: `${img.offsetHeight}px`,
    backgroundColor: `rgba(229, 57, 53, ${intensity})`,
    pointerEvents: 'none',
    zIndex: '9998',
    borderRadius: '6px',
  });

  const parent = img.parentElement;
  if (parent && getComputedStyle(parent).position === 'static') {
    parent.style.position = 'relative';
  }

  parent.appendChild(overlay);
}

// === Add hover listeners to an image ===
function attachHoverEvents(img, confidence) {
  if (img.dataset.aiHoverAttached === 'true') return;
  img.addEventListener('mouseenter', () => showAIPopup(confidence));
  img.addEventListener('mouseleave', hideAIPopup);
  img.dataset.aiHoverAttached = 'true';
}

// === Process a single image ===
function processImage(img) {
  if (!isLargeAndNotIcon(img)) return;
  if (img.dataset.aiProcessed === 'true') return;

  let confidence = Number(img.dataset.aiConfidence);
  if (!confidence) {
    confidence = Math.floor(Math.random() * 101);
    img.dataset.aiConfidence = confidence;
  }

  img.dataset.aiProcessed = 'true';

  if (confidence > 70) {
    createHeatmapOverlay(img, confidence);
    img.dataset.aiFlagged = 'true';
    attachHoverEvents(img, confidence);
    flaggedImages.push(img.src);
    confidences.push(confidence);
  }

  scannedCount++;
}


// === Observe images entering the viewport ===
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      processImage(entry.target);
    }
  });
}, { threshold: 0.5 });

function observeImages() {
  const images = document.querySelectorAll('img');
  images.forEach(img => observer.observe(img));
}

// === Watch for dynamically added images ===
function watchForNewImages() {
  const mo = new MutationObserver(mutations => {
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (!(node instanceof Element)) continue;
        if (node.tagName && node.tagName.toLowerCase() === 'img') {
          const img = node;
          observer.observe(img);
          if (img.complete) {
            try { processImage(img); } catch (e) { console.warn('processImage error on added img:', e); }
          } else {
            img.addEventListener('load', function onLoad() {
              img.removeEventListener('load', onLoad);
              try { processImage(img); } catch (e) { console.warn('processImage error on added img load:', e); }
            });
          }
        } else {
          const imgs = node.querySelectorAll ? node.querySelectorAll('img') : [];
          imgs.forEach(img => {
            observer.observe(img);
            if (img.complete) {
              try { processImage(img); } catch (e) { console.warn('processImage error on added subtree img:', e); }
            } else {
              img.addEventListener('load', function onLoad() {
                img.removeEventListener('load', onLoad);
                try { processImage(img); } catch (e) { console.warn('processImage error on added subtree img load:', e); }
              });
            }
          });
        }
      }
    }
  });

  mo.observe(document.documentElement || document.body, { childList: true, subtree: true });
}

// === Data storage ===
let flaggedImages = [];
let confidences = [];
let scannedCount = 0;

// === Message listener ===
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'check-images') {
    const images = Array.from(document.querySelectorAll('img'));
    let scanned = 0;
    const foundFlagged = [];
    const foundConfidences = [];

    images.forEach(img => {
      try {
        if (img.dataset.aiProcessed !== 'true') {
          processImage(img);
        }
        if (img.dataset.aiProcessed === 'true') scanned++;
        if (img.dataset.aiFlagged === 'true') {
          foundFlagged.push(img.src);
          const c = Number(img.dataset.aiConfidence) || 0;
          foundConfidences.push(c);
        }
      } catch (err) {
        console.warn('Error handling image during check:', err);
      }
    });

    const isAI = foundFlagged.length > 0;
    const response = { scannedCount: scanned, flaggedImages: foundFlagged, confidences: foundConfidences, isAI };
    console.log('QuackScan content script: received check-images, sending response', response);
    try {
      sendResponse(response);
    } catch (err) {
      console.error('Error sending response from content script:', err);
    }
    return;
  }

  if (message.type === 'apply-highlights') {
    try {
      const urls = message.flaggedImages || [];
      const confs = message.confidences || [];
      urls.forEach((url, i) => {
        let target = url;
        try { target = decodeURI(url); } catch (e) {}
        function normalize(u) {
          try { const a = document.createElement('a'); a.href = u; return a.href; } catch (e) { return u; }
        }
        const normTarget = normalize(target);

        const imgs = Array.from(document.querySelectorAll('img[src]')).filter(img => {
          const src = img.src || '';
          const normSrc = normalize(src);
          if (normSrc === normTarget) return true;
          if (src === url || src === target) return true;
          try {
            const srcName = normSrc.split('/').pop();
            const tgtName = normTarget.split('/').pop();
            if (srcName && tgtName && srcName === tgtName) return true;
          } catch (e) {}
          return false;
        });

        imgs.forEach(img => {
          try {
            img.dataset.aiFlagged = 'true';
            img.dataset.aiConfidence = String(confs[i] || img.dataset.aiConfidence || 0);
            createHeatmapOverlay(img, Number(img.dataset.aiConfidence) || 0);
            attachHoverEvents(img, Number(img.dataset.aiConfidence) || 0);
          } catch (e) {
            console.warn('Error applying highlight to image:', e);
          }
        });
      });
      sendResponse({ applied: true });
    } catch (err) {
      console.error('Error in apply-highlights handler:', err);
      try { sendResponse({ applied: false, error: String(err) }); } catch (e) {}
    }
    return;
  }
});
// === Initialize content script on page ===
// Initialize content script immediately if the document is already loaded (injection after load case)
function initContentScript() {
  try {
    createAIPopup();
    observeImages();
    watchForNewImages();
    console.log('QuackScan content script initialized');
  } catch (err) {
    console.error('Error initializing content script:', err);
  }
}

if (document.readyState === 'complete' || document.readyState === 'interactive') {
  initContentScript();
} else {
  window.addEventListener('load', initContentScript);
}
