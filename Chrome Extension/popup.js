// When the popup opens, request the content script to scan images
chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
  chrome.tabs.sendMessage(tabs[0].id, {type: 'check-images'});
  const resultDiv = document.getElementById('result');
  resultDiv.textContent = 'Scanning images...';
});

// Listen for messages from content.js about image analysis
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'analyze-image-result') {
    const resultDiv = document.getElementById('result');
    if (message.isAI) {
      resultDiv.textContent = 'Warning: At least one image on this website might be AI-generated!';
    } else {
      resultDiv.textContent = 'No AI-generated images detected.';
    }
  }
});
