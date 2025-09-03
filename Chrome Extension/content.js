// This content script finds all images on the page and sends their src URLs to the background or popup for processing.

function isLargeAndNotIcon(img) {
    // Example filters: size, alt text, class, src
    const minWidth = 100;
    const minHeight = 100;
    const iconKeywords = ['icon', 'logo', 'profile', 'avatar', 'favicon'];
    const src = img.src.toLowerCase();
    const alt = (img.alt || '').toLowerCase();
    const className = (img.className || '').toLowerCase();
    // Filter by size
    if (img.naturalWidth < minWidth || img.naturalHeight < minHeight) return false;
    // Filter by keywords
    for (const kw of iconKeywords) {
        if (src.includes(kw) || alt.includes(kw) || className.includes(kw)) return false;
    }
    return true;
}

function processAllImages() {
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        if (isLargeAndNotIcon(img)) {
            // Send image src to background/popup for analysis
            chrome.runtime.sendMessage({
                type: 'analyze-image',
                src: img.src
            });
        }
    });
}

// Simulate AI detection (replace with real model call)
function isAIImage(src) {
    // Placeholder: randomly flag images as AI for demo
    return Math.random() < 0.2;
}

function checkImagesAndRespond() {
    const images = document.querySelectorAll('img');
    let foundAI = false;
    let scannedCount = 0;
    let flaggedImages = [];
    let confidences = [];
    images.forEach(img => {
        if (isLargeAndNotIcon(img)) {
            scannedCount++;
            if (isAIImage(img.src)) {
                foundAI = true;
                flaggedImages.push(img.src);
                // Generate a random confidence rate between 60% and 99%
                confidences.push(Math.floor(Math.random() * 40) + 60);
            }
        }
    });
    chrome.runtime.sendMessage({
        type: 'analyze-image-result',
        isAI: foundAI,
        scannedCount: scannedCount,
        flaggedImages: flaggedImages,
        confidences: confidences
    });
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'check-images') {
        checkImagesAndRespond();
    }
});

// Run when the content script loads
processAllImages();

// Automatically check images for AI when the script loads
checkImagesAndRespond();

// Also check whenever new images are added dynamically
const observer = new MutationObserver(() => {
    checkImagesAndRespond();
});
observer.observe(document.body, { childList: true, subtree: true });
