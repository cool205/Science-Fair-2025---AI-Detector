// === Utility: Check if image is large and not an icon ===
console.log('QuackScan content script loaded');
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

// === Show/hide popup on hover ===
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

  const confidence = Math.floor(Math.random() * 101);
  img.dataset.aiConfidence = confidence;
  img.dataset.aiProcessed = 'true';

  if (confidence > 70) {
    img.style.border = '4px solid #e53935';
    img.style.borderRadius = '6px';
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

// === Data storage for this content script ===
let flaggedImages = [];
let confidences = [];
let scannedCount = 0;
let currentIndex = 0;

// === Respond to popup requests ===
// When the popup sends { type: 'check-images' } we scan visible images and return results.
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'check-images') {
    // Build results by reading per-image dataset attributes so previously-processed images are included.
    // Also process any unprocessed images to pick up newly-loaded images.
    const images = Array.from(document.querySelectorAll('img'));
    let scanned = 0;
    const foundFlagged = [];
    const foundConfidences = [];

    images.forEach(img => {
      try {
        if (img.dataset.aiProcessed !== 'true') {
          // Process new images (this will set dataset flags on the element)
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
    return; // no async response
  }
});

// === Initialize content script on page ===
// Initialize content script immediately if the document is already loaded (injection after load case)
function initContentScript() {
  try {
    createAIPopup();
    observeImages();
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
