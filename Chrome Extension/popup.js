document.getElementById('detectBtn').onclick = function() {
  const input = document.getElementById('imageInput');
  const resultDiv = document.getElementById('result');
  if (!input.files[0]) {
    resultDiv.textContent = 'Please select an image.';
    return;
  }
  // Placeholder for AI detection logic
  resultDiv.textContent = 'AI detection not implemented (demo only).';
};
