// === Utility: Check if image is large and not an icon ===
function isLargeAndNotIcon(img) {
    const minWidth = 100;
    const minHeight = 100;
    const iconKeywords = ['icon', 'logo', 'profile', 'avatar', 'favicon'];
    const src = img.src.toLowerCase();
    const alt = (img.alt || '').toLowerCase();
    const className = (img.className || '').toLowerCase();

    if (img.naturalWidth < minWidth || img.naturalHeight < minHeight) return false;

    for (const kw of iconKeywords) {
        if (src.includes(kw) || alt.includes(kw) || className.includes(kw)) return false;
    }
    return true;
}

// === Fake AI detection function (replace with real model) ===
function isAIImage(src) {
    return Math.random() < 0.2; // 20% chance for demo
}

// === Create the top-right transparent popup ===
function createAIPopup() {
    const popup = document.createElement('div');
    popup.id = 'ai-confidence-popup';
    popup.style.position = 'fixed';
    popup.style.top = '12px';
    popup.style.right = '12px';
    popup.style.padding = '10px 16px';
    popup.style.backgroundColor = 'rgba(229, 57, 53, 0.85)'; // transparent red
    popup.style.color = '#fff';
    popup.style.fontSize = '14px';
    popup.style.fontFamily = 'Arial, sans-serif';
    popup.style.borderRadius = '6px';
    popup.style.zIndex = '9999';
    popup.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.15)';
    popup.style.backdropFilter = 'blur(4px)';
    popup.style.transition = 'opacity 0.2s ease';
    popup.style.opacity = '0'; // hidden by default
    popup.style.pointerEvents = 'none'; // allow mouse to pass through
    document.body.appendChild(popup);
}

// === Show/hide popup on hover ===
function showAIPopup(confidence) {
    const popup = document.getElementById('ai-confidence-popup');
    if (popup) {
        popup.textContent = `AI Confidence: ${confidence}%`;
        popup.style.opacity = '1';
    }
}

function hideAIPopup() {
    const popup = document.getElementById('ai-confidence-popup');
    if (popup) {
        popup.style.opacity = '0';
    }
}

// === Add hover listeners to an image ===
function attachHoverEvents(img, confidence) {
    // Prevent adding listeners more than once
    if (img.dataset.aiHoverAttached === 'true') return;

    img.addEventListener('mouseenter', () => showAIPopup(confidence));
    img.addEventListener('mouseleave', hideAIPopup);
    img.dataset.aiHoverAttached = 'true';
}

// === Process all images (initial + on mutation) ===
function checkImagesAndRespond() {
    const images = document.querySelectorAll('img');
    let foundAI = false;
    let scannedCount = 0;
    let flaggedImages = [];
    let confidences = [];

    images.forEach(img => {
        if (isLargeAndNotIcon(img)) {
            scannedCount++;
            const alreadyFlagged = img.dataset.aiFlagged === 'true';

            if (!alreadyFlagged && isAIImage(img.src)) {
                foundAI = true;
                const confidence = Math.floor(Math.random() * 40) + 60;

                flaggedImages.push(img.src);
                confidences.push(confidence);

                img.style.border = '4px solid #e53935';
                img.style.borderRadius = '6px';
                img.dataset.aiConfidence = confidence;
                img.dataset.aiFlagged = 'true';

                attachHoverEvents(img, confidence);
            }
        }
    });

    chrome.runtime.sendMessage({
        type: 'analyze-image-result',
        isAI: foundAI,
        scannedCount,
        flaggedImages,
        confidences
    });
}

// === Observe dynamic DOM changes ===
const observer = new MutationObserver(() => {
    checkImagesAndRespond();
});
observer.observe(document.body, { childList: true, subtree: true });

// === Run once on load ===
createAIPopup();
checkImagesAndRespond();
