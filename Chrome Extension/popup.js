document.addEventListener('DOMContentLoaded', () => {
  const resultDiv = document.getElementById('result');
  const sliderContainer = document.getElementById('slider-container');
  const sliderImg = document.getElementById('slider-img');
  const confidenceDiv = document.getElementById('confidence');
  const prevBtn = document.getElementById('prevBtn');
  const nextBtn = document.getElementById('nextBtn');

  let flaggedImages = [];
  let confidences = [];
  let currentIndex = 0;

  function updateSlider() {
    if (flaggedImages.length > 0) {
      sliderContainer.style.display = 'flex';
      sliderImg.src = flaggedImages[currentIndex];
      confidenceDiv.textContent = `Confidence: ${confidences[currentIndex]}%`;
    } else {
      sliderContainer.style.display = 'none';
    }
  }

  prevBtn.onclick = () => {
    if (flaggedImages.length > 0) {
      currentIndex = (currentIndex - 1 + flaggedImages.length) % flaggedImages.length;
      updateSlider();
    }
  };

  nextBtn.onclick = () => {
    if (flaggedImages.length > 0) {
      currentIndex = (currentIndex + 1) % flaggedImages.length;
      updateSlider();
    }
  };

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs.length === 0) return;

    chrome.scripting.executeScript({
      target: { tabId: tabs[0].id },
      files: ['content.js']
    }, () => {
      chrome.tabs.sendMessage(tabs[0].id, { type: 'check-images' }, (response) => {
        if (chrome.runtime.lastError) {
          resultDiv.textContent = 'Error: Could not connect to content script.';
          resultDiv.className = 'warning';
          console.error('Messaging failed:', chrome.runtime.lastError.message);
        } else {
          console.log('Message sent:', response);
        }
      });
    });
  });

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'analyze-image-result') {
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
});
