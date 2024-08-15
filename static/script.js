
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext("2d");
const sizeSlider = document.getElementById('size-slider');
const tool = document.querySelectorAll(".tool");
const clearCanvas = document.querySelector(".clear-canvas");
const saveCanvas = document.querySelector(".save-img");

let prevMouseX, prevMouseY;
let isDrawing = false;

selectedTool = "brush";
brushWidth = sizeSlider.value;

window.addEventListener('load', () => {
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
});

window.addEventListener('resize', () => {
  requestAnimationFrame(() => {
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
  });
});

const drawing = (e) => {
  if(!isDrawing) return;
  ctx.putImageData(snapshot, 0, 0);
  if(selectedTool === "brush" || selectedTool === "eraser") {
    
    ctx.strokeStyle = selectedTool === "eraser" ? "#fff" : "#000";
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = brushWidth;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
  }
}

const startDraw = (e) => {
  isDrawing = true;
  prevMouseX = e.offsetX;
  prevMouseY = e.offsetY;
  ctx.beginPath();
  ctx.lineWidth = brushWidth;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  snapshot = ctx.getImageData(0, 0, canvas.width, canvas.height);
}

tool.forEach(item => {
  console.log(item)
  item.addEventListener('click', () => {
    console.log("clicked ", item.id)
    document.querySelector(".options .active").classList.remove("active");
    item.classList.add("active");
    selectedTool = item.id;
  })
})
sizeSlider.addEventListener('change', () => {brushWidth = sizeSlider.value;})

clearCanvas.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});

saveCanvas.addEventListener('click', () => {
  document.querySelector("#imgPreview").setAttribute("src", "")
  var dataURL = canvas.toDataURL('image/png');
  fetch('/upload-image', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ image: dataURL })
  })
  .then(response => response.arrayBuffer())
  .then(buffer => document.querySelector("#imgPreview").setAttribute("src", URL.createObjectURL(new Blob([buffer], { type: 'image/jpeg' }))))
})
canvas.addEventListener('mousedown', startDraw)
canvas.addEventListener('mouseup', () => isDrawing = false)
canvas.addEventListener('mousemove', (e) => {
    drawing(e);
})