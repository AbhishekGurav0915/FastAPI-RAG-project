const uploadForm = document.getElementById('uploadForm');
const queryForm = document.getElementById('queryForm');
const messages = document.getElementById('messages');
const uploadStatus = document.getElementById('uploadStatus');

uploadForm.addEventListener('submit', async e => {
    e.preventDefault();
    const file = document.getElementById('pdfFile').files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch('/upload', { method: 'POST', body: formData });
    const data = await res.json();
    uploadStatus.textContent = data.message || data.error;
    uploadStatus.style.color = data.error ? 'red' : 'green';
});

queryForm.addEventListener('submit', async e => {
    e.preventDefault();
    const q = document.getElementById('question').value.trim();
    if (!q) return;

    // user message
    messages.innerHTML += `<div class="message user"><strong>You:</strong> ${q}</div>`;
    document.getElementById('question').value = '';

    const formData = new FormData();
    formData.append('question', q);

    const res = await fetch('/query', { method: 'POST', body: formData });
    const data = await res.json();

    messages.innerHTML += `<div class="message bot"><strong>AI:</strong> ${data.answer}</div>`;
    messages.scrollTop = messages.scrollHeight;
});