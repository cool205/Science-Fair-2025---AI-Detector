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
    if (message.isAI) {
      msg += 'AI Images Detected on Website';
      resultDiv.className = 'warning';
    } else {
      msg += 'No AI Images Detected on Website';
      resultDiv.className = 'safe';
    }
    resultDiv.textContent = msg;
  }
});
