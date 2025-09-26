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

  function handleResponse(message) {
    if (!message) return;
    flaggedImages = message.flaggedImages || [];
    confidences = message.confidences || [];
    currentIndex = 0;

    let msg = `Scanned <b>${message.scannedCount || 0}</b> images.<br>`;
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

    const tabId = tabs[0].id;

    function isRestrictedUrl(url) {
      if (!url) return true;
      const lower = url.toLowerCase();
      // Pages where content scripts cannot be injected
      const restrictedPrefixes = ['chrome://', 'edge://', 'about:', 'view-source:', 'extension://'];
      if (restrictedPrefixes.some(p => lower.startsWith(p))) return true;
      // Chrome Web Store and internal extension pages are restricted
      if (lower.includes('chrome.google.com/webstore')) return true;
      // file: pages require explicit host permission which most extensions avoid
      if (lower.startsWith('file:')) return true;
      return false;
    }

    function trySendMessage() {
      // show scanning indicator while we wait
      resultDiv.innerHTML = 'Scanning images...';
      resultDiv.className = '';

      chrome.tabs.sendMessage(tabId, { type: 'check-images' }, (response) => {
        if (!chrome.runtime.lastError) {
          console.log('Message sent (initial):', response);
          handleResponse(response);
          return;
        }

        console.warn('Initial messaging failed:', chrome.runtime.lastError.message);
        // Attempt programmatic injection as a fallback (requires scripting & host permissions)
        if (!chrome.scripting) {
          // scripting API not available; show a helpful error
          resultDiv.textContent = 'Error: Cannot connect to content script and scripting API unavailable.';
          resultDiv.className = 'warning';
          return;
        }

        // Check URL first and give a clearer message if it's a restricted page.
        const tabUrl = tabs[0].url || '';
        if (isRestrictedUrl(tabUrl)) {
          resultDiv.innerHTML = 'Error: This page is restricted for extensions (e.g., chrome://, the Web Store, or local file pages). Try the extension on a normal website like https://example.com.';
          resultDiv.className = 'warning';
          console.error('Page is restricted for content scripts:', tabUrl);
          return;
        }

        // Try injecting the content script programmatically
        chrome.scripting.executeScript({ target: { tabId }, files: ['content.js'] }, () => {
          if (chrome.runtime.lastError) {
            resultDiv.textContent = 'Error: Could not inject content script (injection failed).';
            resultDiv.className = 'warning';
            console.error('Injection failed:', chrome.runtime.lastError.message);
            return;
          }

          // Retry messaging several times after successful injection so the content script has time to register its listener.
          const maxAttempts = 5;
          let attempt = 0;
          function retrySend() {
            chrome.tabs.sendMessage(tabId, { type: 'check-images' }, (resp) => {
              if (!chrome.runtime.lastError) {
                console.log('Message sent after injection:', resp);
                handleResponse(resp);
                return;
              }
              attempt++;
              if (attempt < maxAttempts) {
                console.warn(`Messaging after injection failed (attempt ${attempt}), retrying...`, chrome.runtime.lastError.message);
                setTimeout(retrySend, 250);
              } else {
                resultDiv.textContent = 'Error: Could not connect to content script after injection.';
                resultDiv.className = 'warning';
                console.error('Messaging failed after injection:', chrome.runtime.lastError.message);
              }
            });
          }
          retrySend();
        });
      });
    }

    trySendMessage();
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
