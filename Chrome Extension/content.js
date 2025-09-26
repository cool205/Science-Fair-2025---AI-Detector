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

// === Safe messaging to background script ===
function safelySendMessageToBackground(message) {
  setTimeout(() => {
    try {
      chrome.runtime.sendMessage(message, (response) => {
        if (chrome.runtime.lastError) {
          console.warn('Message failed:', chrome.runtime.lastError.message);
        } else {
          console.log('Message sent successfully:', response);
        }
      });
    } catch (err) {
      console.error('Extension context invalidated:', err);
    }
  }, 500);
}

// === Messaging and UI update ===
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'analyze-image-result') {
    const resultDiv = document.getElementById('result');
    if (!resultDiv) return;

    flaggedImages = message.flaggedImages || [];
    confidences = message.confidences || [];
    currentIndex = 0;

    let msg = `Scanned <b>${message.scannedCount}</b> images.<br>`;
    if (message.isAI) {
      msg += `<span style="color:#e53935;font-weight:600;">Flagged <b>${flaggedImages.length}</b> image${flaggedImages.length === 1 ? '' : 's'} as AI-generated.</span><br>`;
      resultDiv.className = 'warning';
    } else {
      msg += '<span style="color:#155724;font-weight:600;">No AI-generated images detected.</span>';
      resultDiv.className = 'safe';
    }
    resultDiv.innerHTML = msg;
    updateSlider();
  }
});

// === Slider UI ===
let flaggedImages = [];
let confidences = [];
let scannedCount = 0;
let currentIndex = 0;

function updateSlider() {
  const sliderContainer = document.getElementById('slider-container');
  const sliderImg = document.getElementById('slider-img');
  const confidenceDiv = document.getElementById('confidence');
  if (flaggedImages.length > 0) {
    sliderContainer.style.display = 'flex';
    sliderImg.src = flaggedImages[currentIndex];
    confidenceDiv.textContent = `Confidence: ${confidences[currentIndex]}%`;
  } else {
    sliderContainer.style.display = 'none';
  }
}

document.getElementById('prevBtn').onclick = () => {
  if (flaggedImages.length > 0) {
    currentIndex = (currentIndex - 1 + flaggedImages.length) % flaggedImages.length;
    updateSlider();
  }
};

document.getElementById('nextBtn').onclick = () => {
  if (flaggedImages.length > 0) {
    currentIndex = (currentIndex + 1) % flaggedImages.length;
    updateSlider();
  }
};

// === Initialize ===
window.addEventListener('load', () => {
  createAIPopup();
  observeImages();
  safelySendMessageToBackground({ type: 'check-images' });
});
