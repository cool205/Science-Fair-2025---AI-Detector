// When the popup opens, request the content script to scan images
chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
  chrome.tabs.sendMessage(tabs[0].id, {type: 'check-images'});
  const resultDiv = document.getElementById('result');
  resultDiv.textContent = 'Scanning images...';
  resultDiv.className = '';
});

// Listen for messages from content.js about image analysis
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'analyze-image-result') {
    const resultDiv = document.getElementById('result');
    let msg = `Scanned ${message.scannedCount} images.\n`;
    flaggedImages = message.flaggedImages || [];
    confidences = message.confidences || [];
    currentIndex = 0;
    if (message.isAI) {
      msg += 'Warning: At least one image on this website might be AI-generated!';
      resultDiv.className = 'warning';
    } else {
      msg += 'No AI-generated images detected.';
      resultDiv.className = 'safe';
    }
    resultDiv.textContent = msg;
    updateSlider();
  }
});

let flaggedImages = [];
let confidences = [];
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

document.getElementById('prevBtn').onclick = function() {
  if (flaggedImages.length > 0) {
    currentIndex = (currentIndex - 1 + flaggedImages.length) % flaggedImages.length;
    updateSlider();
  }
};
document.getElementById('nextBtn').onclick = function() {
  if (flaggedImages.length > 0) {
    currentIndex = (currentIndex + 1) % flaggedImages.length;
    updateSlider();
  }
};
