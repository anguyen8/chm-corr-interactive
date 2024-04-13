const gridSize = 7;
var cellSize = null;
var inputImage = null;
var image = null;
var canvas = null;
var ctx = null;
var canvasBg = null;
let grid = new Array(gridSize).fill(null).map(() => new Array(gridSize).fill(false));
var isInitialized = false;

let selectedCells = 0;

function createGrid() {
    console.log('createGrid')

    for (let i = 0; i < 49; i++) {
        const div = document.createElement('div');
        div.classList.add('checkbox');
        div.innerHTML = '<input type="checkbox">';
        grid.appendChild(div);
    }
}

function loadImage(event) {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onload = function (e) {
        image.src = e.target.result;
    }
    reader.readAsDataURL(file);
}


function handleMouseDown(event) {
    // console.log("handleMouseDown");
}

function handleMouseMove(event) {
    // console.log("handleMouseMove");
}

function handleMouseUp(event) {
    // console.log("handleMouseUp");
}

function handleMouseLeave(event) {
    // console.log("handleMouseLeave");
}


function drawGrid() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawBackground();
    for (let row = 0; row < gridSize; row++) {
        for (let col = 0; col < gridSize; col++) {
            ctx.beginPath();
            ctx.rect(col * cellSize, row * cellSize, cellSize, cellSize);
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 2;
            ctx.stroke();

            if (grid[row][col]) {
                ctx.fillStyle = 'rgba(0, 255, 0, 0.5)';
                ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
            }
        }
    }
}


function initializeEditor() {
    console.log("initializeEditor");

    if (isInitialized) {
        return;
    }
    isInitialized = true;

    image = document.getElementById('image');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');

    // Add click event listener to canvas
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('mouseleave', handleMouseLeave);

    cellSize = canvas.width / gridSize;

    canvas.addEventListener('click', (event) => {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const x = (event.clientX - rect.left) * scaleX;
        const y = (event.clientY - rect.top) * scaleY;
        const row = Math.floor(y / cellSize);
        const col = Math.floor(x / cellSize);

        // If the cell is already selected, it's always allowed to deselect it
        if (grid[row][col]) {
            grid[row][col] = false;
            selectedCells--; // Decrement the selected cell count
        } else {
            // Only select a new cell if less than 50 cells are already selected
            if (selectedCells < 50) {
                grid[row][col] = true;
                selectedCells++; // Increment the selected cell count
            }
        }
        drawGrid();
    });

    drawGrid();
}


function drawBackground() {
    if (canvasBg != null) {
        const canvasWidth = canvas.width;
        const canvasHeight = canvas.height;

        const bgWidth = canvasBg.width;
        const bgHeight = canvasBg.height;

        const scaleX = canvasWidth / bgWidth;
        const scaleY = canvasHeight / bgHeight;

        const scale = Math.max(scaleX, scaleY);

        const newWidth = bgWidth * scale;
        const newHeight = bgHeight * scale;

        const xOffset = (canvasWidth - newWidth) / 2;
        const yOffset = (canvasHeight - newHeight) / 2;

        ctx.drawImage(canvasBg, 0, 0, bgWidth, bgHeight, xOffset, yOffset, newWidth, newHeight);
    }
}

function importBackground(image) {
    if (image == null) {
        canvasBg = null;
        drawGrid();
        return;
    }

    let m = new Image();
    m.src = image;
    m.onload = function () {
        canvasBg = m;
        drawGrid();
    }
}

function read_js_Data() {
    console.log("read_js_Data");
    console.log("read_js_Data");
    console.log("read_js_Data");
    console.log("read_js_Data");
    console.log("read_js_Data");
    return grid;
}


function set_grid_from_data(data) {
    if (data.length !== gridSize || data[0].length !== gridSize) {
        throw new Error('Invalid data dimensions. Expected ' + gridSize + 'x' + gridSize);
    }

    selectedCells = 0; // Reset the selected cell count
    for (let row = 0; row < gridSize; row++) {
        for (let col = 0; col < gridSize; col++) {
            grid[row][col] = data[row][col];
            if (grid[row][col]) {
                selectedCells++; // Count the number of initially selected cells
            }
        }
    }

    drawGrid();
}


function clear_grid() {
    console.log("clearGrid");
    for (let row = 0; row < gridSize; row++) {
        for (let col = 0; col < gridSize; col++) {
            grid[row][col] = false;
        }
    }
    selectedCells = 0; // Reset the selected cell count
    drawGrid();
}