// Create a small canvas and export as PNG for testing
const canvas = document.createElement('canvas');
canvas.width = 100;
canvas.height = 100;
const ctx = canvas.getContext('2d');
ctx.fillStyle = 'blue';
ctx.fillRect(0, 0, 100, 100);
const dataURL = canvas.toDataURL('image/png');