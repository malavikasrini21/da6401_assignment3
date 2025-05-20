def create_interactive_connectivity(attn_matrix, input_seq, output_seq, filename="attention.html"):
    """
    Creates a complete interactive connectivity visualization with:
    - Output characters on top
    - Input characters below
    - Green connection lines
    - Hover/click interaction
    - Threshold slider
    """
    
    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Attention Connectivity Visualization</title>
    <style>
        body { 
            font-family: 'Arial Unicode MS', 'Noto Sans Devanagari', Arial, sans-serif;
            margin: 20px; 
            text-align: center;
        }
        .container { 
            display: inline-block; 
            text-align: center;
            margin: 0 auto;
        }
        .output-chars { 
            display: flex; 
            justify-content: center;
            margin-bottom: 40px;
        }
        .input-chars { 
            display: flex; 
            justify-content: center;
            margin-top: 20px;
        }
        .char { 
            padding: 10px 15px;
            margin: 5px;
            font-size: 24px;
            position: relative;
            cursor: pointer;
            transition: all 0.3s;
            min-width: 30px;
            text-align: center;
        }
        .output-char { 
            background-color: #f0f0f0; 
            border-radius: 5px; 
        }
        .input-char { 
            background-color: #e0e0e0; 
            border-radius: 3px; 
        }
        .connection-line {
            position: absolute;
            background-color: rgba(0, 200, 0, 0.5);
            height: 4px;
            transform-origin: left center;
            z-index: -1;
            pointer-events: none;
        }
        .selected { 
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            box-shadow: 0 0 10px rgba(76, 175, 80, 0.5);
        }
        .highlighted { 
            background-color: rgba(76, 175, 80, 0.3);
            transform: scale(1.1);
        }
        .controls { 
            margin: 20px 0; 
        }
        .slider { 
            width: 300px; 
            margin: 0 10px; 
        }
        .threshold-value { 
            display: inline-block; 
            width: 50px; 
        }
        h2 { color: #333; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Attention Connectivity Visualization</h2>
        
        <div class="controls">
            <label>Connection Threshold: </label>
            <input type="range" min="0" max="100" value="30" class="slider" id="thresholdSlider">
            <span class="threshold-value" id="thresholdValue">0.30</span>
        </div>
        
        <div class="output-chars" id="outputChars"></div>
        <div class="input-chars" id="inputChars"></div>
    </div>

    <script>
        // Convert Python data to JS format
        const attentionData = {attn_matrix};
        const inputChars = {input_seq};
        const outputChars = {output_seq};
        
        let currentSelected = 0;
        let threshold = 0.3;
        
        function initVisualization() {
            renderOutputChars();
            renderInputChars();
            updateConnections();
            
            // Setup threshold slider
            document.getElementById('thresholdSlider').addEventListener('input', function(e) {
                threshold = parseInt(e.target.value) / 100;
                document.getElementById('thresholdValue').textContent = threshold.toFixed(2);
                updateConnections();
            });
            
            // Handle window resize
            window.addEventListener('resize', updateConnections);
        }
        
        function renderOutputChars() {
            const container = document.getElementById('outputChars');
            container.innerHTML = '';
            
            outputChars.forEach((char, idx) => {
                const charElement = document.createElement('div');
                charElement.className = `char output-char ${idx === currentSelected ? 'selected' : ''}`;
                charElement.textContent = char;
                charElement.dataset.index = idx;
                
                charElement.addEventListener('mouseover', () => selectCharacter(idx));
                charElement.addEventListener('click', () => selectCharacter(idx));
                
                container.appendChild(charElement);
            });
        }
        
        function renderInputChars() {
            const container = document.getElementById('inputChars');
            container.innerHTML = '';
            
            inputChars.forEach((char, idx) => {
                const charElement = document.createElement('div');
                charElement.className = 'char input-char';
                charElement.textContent = char;
                charElement.dataset.index = idx;
                container.appendChild(charElement);
            });
        }
        
        function selectCharacter(idx) {
            currentSelected = idx;
            renderOutputChars();
            updateConnections();
        }
        
        function updateConnections() {
            // Clear existing connections
            document.querySelectorAll('.connection-line').forEach(el => el.remove());
            document.querySelectorAll('.input-char').forEach(el => el.classList.remove('highlighted'));
            
            const outputChar = document.querySelector(`.output-char[data-index="${currentSelected}"]`);
            if (!outputChar) return;
            
            const outputRect = outputChar.getBoundingClientRect();
            const attentionWeights = attentionData[currentSelected];
            const maxWeight = Math.max(...attentionWeights);
            
            inputChars.forEach((_, idx) => {
                const inputChar = document.querySelector(`.input-char[data-index="${idx}"]`);
                if (!inputChar) return;
                
                const inputRect = inputChar.getBoundingClientRect();
                const normalizedWeight = attentionWeights[idx] / maxWeight;
                
                if (normalizedWeight >= threshold) {
                    inputChar.classList.add('highlighted');
                    
                    const line = document.createElement('div');
                    line.className = 'connection-line';
                    
                    const startX = outputRect.left + outputRect.width/2 - window.scrollX;
                    const startY = outputRect.top + outputRect.height - window.scrollY;
                    const endX = inputRect.left + inputRect.width/2 - window.scrollX;
                    const endY = inputRect.top - window.scrollY;
                    
                    const length = Math.sqrt(Math.pow(endX - startX, 2) + Math.pow(endY - startY, 2));
                    const angle = Math.atan2(endY - startY, endX - startX) * 180 / Math.PI;
                    
                    line.style.width = `${length}px`;
                    line.style.left = `${startX}px`;
                    line.style.top = `${startY}px`;
                    line.style.transform = `rotate(${angle}deg)`;
                    line.style.opacity = normalizedWeight;
                    
                    document.body.appendChild(line);
                }
            });
        }
        
        // Initialize visualization
        document.addEventListener('DOMContentLoaded', initVisualization);
    </script>
</body>
</html>"""
    
    # Prepare data for JavaScript
    attn_matrix_json = [[float(w) for w in row] for row in attn_matrix]
    input_seq_json = [str(c) for c in input_seq]
    output_seq_json = [str(c) for c in output_seq]
    
    # Insert data into template
    import json
    html_content = html_template \
        .replace('{attn_matrix}', json.dumps(attn_matrix_json)) \
        .replace('{input_seq}', json.dumps(input_seq_json)) \
        .replace('{output_seq}', json.dumps(output_seq_json))
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Visualization saved to {filename}")
    return filename


# Sample data
attn_weights = [
    [0.1, 0.8, 0.1],  # Attention for first output character
    [0.7, 0.2, 0.1],  # Attention for second output character
    [0.1, 0.2, 0.7]   # Attention for third output character  
]
input_chars = ["a", "b", "c"]
output_chars = ["अ", "ब", "क"]  # Devanagari characters

# Generate visualization
html_file = create_interactive_connectivity(
    attn_weights,
    input_chars,
    output_chars
)

# To view in notebook:
from IPython.display import HTML
HTML(filename=html_file)

# To log to wandb:
import wandb
wandb.init(project="da6401_assignment3")

wandb.log({"attention_viz": wandb.Html(open(html_file).read())})