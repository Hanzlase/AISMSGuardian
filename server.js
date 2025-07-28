const express = require('express');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

const app = express();
const PORT = 8000;
const HOST = '0.0.0.0';

// Middleware
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Serve static files from templates directory
app.use(express.static('templates'));

// Route to serve the HTML interface
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'templates', 'index.html'));
});

// Function to call Python script for prediction
function predictWithPython(message) {
    return new Promise((resolve, reject) => {
        // Create a Python script that loads the models and makes predictions
        const pythonScript = `
import sys
import joblib
import numpy as np
import json

# Load the trained models
try:
    nb_classifier = joblib.load('model/nb_classifier.pkl')
    vectorizer = joblib.load('model/vectorizer.pkl')
    
    # Get message from command line argument
    message = sys.argv[1] if len(sys.argv) > 1 else ""
    
    # Convert the message into a TF-IDF vector
    message_vec = vectorizer.transform([message])
    
    # Use the trained classifier to make a prediction
    prediction = nb_classifier.predict(message_vec)
    
    # Get the probability/confidence of the prediction
    probability = np.max(nb_classifier.predict_proba(message_vec))
    
    # Prepare the result
    result = {
        'prediction': 'Spam' if prediction[0] == 1 else 'Ham',
        'probability': f"{probability * 100:.2f}%"
    }
    
    print(json.dumps(result))
    
except Exception as e:
    error_result = {
        'error': str(e)
    }
    print(json.dumps(error_result))
`;

        // Write the Python script to a temporary file
        const scriptPath = path.join(__dirname, 'temp_predict.py');
        fs.writeFileSync(scriptPath, pythonScript);

        // Spawn Python process
        const pythonProcess = spawn('python', [scriptPath, message]);
        
        let output = '';
        let errorOutput = '';

        pythonProcess.stdout.on('data', (data) => {
            output += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            errorOutput += data.toString();
        });

        pythonProcess.on('close', (code) => {
            // Clean up temporary file
            try {
                fs.unlinkSync(scriptPath);
            } catch (err) {
                console.warn('Could not delete temporary Python script:', err.message);
            }

            if (code !== 0) {
                reject(new Error(`Python process exited with code ${code}: ${errorOutput}`));
                return;
            }

            try {
                const result = JSON.parse(output.trim());
                if (result.error) {
                    reject(new Error(result.error));
                } else {
                    resolve(result);
                }
            } catch (parseError) {
                reject(new Error(`Failed to parse Python output: ${parseError.message}`));
            }
        });

        pythonProcess.on('error', (error) => {
            // Clean up temporary file
            try {
                fs.unlinkSync(scriptPath);
            } catch (err) {
                console.warn('Could not delete temporary Python script:', err.message);
            }
            reject(new Error(`Failed to spawn Python process: ${error.message}`));
        });
    });
}

// Route to handle prediction requests
app.post('/predict', async (req, res) => {
    try {
        const { message } = req.body;
        
        if (!message) {
            return res.status(400).json({ error: 'Message is required' });
        }

        // Call Python script to make prediction
        const result = await predictWithPython(message);
        
        res.json(result);
    } catch (error) {
        console.error('Prediction error:', error.message);
        res.status(500).json({ 
            error: 'Failed to make prediction',
            details: error.message 
        });
    }
});

// Start the server
app.listen(PORT, HOST, () => {
    console.log("***********************************");
    console.log("*** Visit : http://localhost:8000 ***");
    console.log("***********************************");
});
