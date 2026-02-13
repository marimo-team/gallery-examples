function render({ model, el }) {
  el.classList.add('suguru-widget-root');
  
  const container = document.createElement('div');
  container.className = 'suguru-container';
  el.appendChild(container);
  
  const controls = document.createElement('div');
  controls.className = 'suguru-controls';
  
  const widthLabel = document.createElement('label');
  widthLabel.textContent = 'Width: ';
  const widthInput = document.createElement('input');
  widthInput.type = 'number';
  widthInput.min = '3';
  widthInput.max = '10';
  widthInput.value = model.get('width') || 5;
  widthLabel.appendChild(widthInput);
  
  const heightLabel = document.createElement('label');
  heightLabel.textContent = 'Height: ';
  const heightInput = document.createElement('input');
  heightInput.type = 'number';
  heightInput.min = '3';
  heightInput.max = '10';
  heightInput.value = model.get('height') || 5;
  heightLabel.appendChild(heightInput);
  
  const newRegionButton = document.createElement('button');
  newRegionButton.textContent = 'New Region';
  newRegionButton.className = 'suguru-button';
  
  const clearButton = document.createElement('button');
  clearButton.textContent = 'Reset';
  clearButton.className = 'suguru-button';
  
  controls.appendChild(widthLabel);
  controls.appendChild(heightLabel);
  controls.appendChild(newRegionButton);
  controls.appendChild(clearButton);
  el.appendChild(controls);
  
  const grid = document.createElement('div');
  grid.className = 'suguru-grid';
  container.appendChild(grid);
  
  let currentRegionId = 0;
  let isDrawing = false;
  let hasMoved = false;
  let startX = -1;
  let startY = -1;
  let shapes = [];
  
  function updateGrid() {
    const width = parseInt(widthInput.value) || 5;
    const height = parseInt(heightInput.value) || 5;
    
    // Initialize shapes if needed or resize
    if (shapes.length !== height || (shapes[0] && shapes[0].length !== width)) {
      shapes = [];
      for (let y = 0; y < height; y++) {
        shapes[y] = [];
        for (let x = 0; x < width; x++) {
          shapes[y][x] = y * width + x; // Start with each cell as its own region
        }
      }
      currentRegionId = width * height;
    }
    
    grid.innerHTML = '';
    grid.style.gridTemplateColumns = `repeat(${width}, 1fr)`;
    grid.style.gridTemplateRows = `repeat(${height}, 1fr)`;
    
    const colors = [
      '#90caf9', '#ce93d8', '#ffcc80', '#a5d6a7',
      '#f48fb1', '#80deea', '#fff59d', '#c5e1a5',
      '#e1bee7', '#b39ddb', '#81c784', '#ffb74d',
      '#64b5f6', '#ba68c8', '#ffa726', '#66bb6a'
    ];
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const cell = document.createElement('div');
        cell.className = 'suguru-cell';
        const regionId = shapes[y][x];
        const color = colors[regionId % colors.length];
        cell.style.backgroundColor = color;
        cell.dataset.x = x;
        cell.dataset.y = y;
        cell.dataset.regionId = regionId;
        
        cell.addEventListener('mousedown', (e) => {
          e.preventDefault();
          isDrawing = true;
          hasMoved = false;
          startX = x;
          startY = y;
          
          // If right-click or ctrl-click, create new region
          if (e.button === 2 || e.ctrlKey || e.metaKey) {
            let maxId = -1;
            for (let y = 0; y < shapes.length; y++) {
              for (let x = 0; x < shapes[y].length; x++) {
                maxId = Math.max(maxId, shapes[y][x]);
              }
            }
            currentRegionId = maxId + 1;
            updateCell(x, y, currentRegionId);
          } else {
            // On left click, create new region for that cell
            let maxId = -1;
            for (let y = 0; y < shapes.length; y++) {
              for (let x = 0; x < shapes[y].length; x++) {
                maxId = Math.max(maxId, shapes[y][x]);
              }
            }
            currentRegionId = maxId + 1;
            updateCell(x, y, currentRegionId);
          }
        });
        
        // Prevent context menu on right-click
        cell.addEventListener('contextmenu', (e) => {
          e.preventDefault();
        });
        
        cell.addEventListener('mouseenter', () => {
          if (isDrawing) {
            hasMoved = true;
            updateCell(x, y, currentRegionId);
          }
        });
        
        grid.appendChild(cell);
      }
    }
    
    updateModel();
  }
  
  function updateCell(x, y, regionId) {
    const cell = grid.querySelector(`[data-x="${x}"][data-y="${y}"]`);
    if (cell) {
      const colors = [
        '#90caf9', '#ce93d8', '#ffcc80', '#a5d6a7',
        '#f48fb1', '#80deea', '#fff59d', '#c5e1a5',
        '#e1bee7', '#b39ddb', '#81c784', '#ffb74d',
        '#64b5f6', '#ba68c8', '#ffa726', '#66bb6a'
      ];
      shapes[y][x] = regionId;
      cell.dataset.regionId = regionId;
      cell.style.backgroundColor = colors[regionId % colors.length];
      updateModel();
    }
  }
  
  function updateModel() {
    // Normalize region IDs to be consecutive starting from 0
    const regionMap = {};
    let newId = 0;
    const normalized = shapes.map(row => 
      row.map(oldId => {
        if (!(oldId in regionMap)) {
          regionMap[oldId] = newId++;
        }
        return regionMap[oldId];
      })
    );
    
    model.set('shapes', normalized);
    model.save_changes();
  }
  
  widthInput.addEventListener('change', () => {
    model.set('width', parseInt(widthInput.value) || 5);
    model.save_changes();
    updateGrid();
  });
  
  heightInput.addEventListener('change', () => {
    model.set('height', parseInt(heightInput.value) || 5);
    model.save_changes();
    updateGrid();
  });
  
  newRegionButton.addEventListener('click', () => {
    // Find the maximum region ID and create a new one
    let maxId = -1;
    for (let y = 0; y < shapes.length; y++) {
      for (let x = 0; x < shapes[y].length; x++) {
        maxId = Math.max(maxId, shapes[y][x]);
      }
    }
    currentRegionId = maxId + 1;
  });
  
  clearButton.addEventListener('click', () => {
    const width = parseInt(widthInput.value) || 5;
    const height = parseInt(heightInput.value) || 5;
    shapes = [];
    for (let y = 0; y < height; y++) {
      shapes[y] = [];
      for (let x = 0; x < width; x++) {
        shapes[y][x] = y * width + x;
      }
    }
    currentRegionId = width * height;
    updateGrid();
  });
  
  document.addEventListener('mouseup', () => {
    isDrawing = false;
    hasMoved = false;
    startX = -1;
    startY = -1;
  });
  
  // Initialize from model
  model.on('change:width', () => {
    widthInput.value = model.get('width') || 5;
    updateGrid();
  });
  
  model.on('change:height', () => {
    heightInput.value = model.get('height') || 5;
    updateGrid();
  });
  
  // Load existing shapes if available
  const existingShapes = model.get('shapes');
  if (existingShapes && Array.isArray(existingShapes) && existingShapes.length > 0) {
    shapes = existingShapes.map(row => [...row]);
    updateGrid();
  } else {
    updateGrid();
  }
  
  return () => {
    // Cleanup
  };
}

export default { render };
