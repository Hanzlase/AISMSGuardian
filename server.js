const express = require('express');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 8000;

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
        const pythonScript = `
import sys
import joblib
import numpy as np
import json

try:
    nb_classifier = joblib.load('model/nb_classifier.pkl')
    vectorizer = joblib.load('model/vectorizer.pkl')
    message = sys.argv[1] if len(sys.argv) > 1 else ""
    message_vec = vectorizer.transform([message])
    prediction = nb_classifier.predict(message_vec)
    probability = np.max(nb_classifier.predict_proba(message_vec))
    result = {
        'prediction': 'Spam' if prediction[0] == 1 else 'Ham',
        'probability': f"{probability * 100:.2f}%"
    }
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({ 'error': str(e) }))
`;

        const scriptPath = path.join(__dirname, 'temp_predict.py');
        fs.writeFileSync(scriptPath, pythonScript);

        const pythonProcess = spawn('python3', [scriptPath, message]);

        let output = '';
        let errorOutput = '';

        pythonProcess.stdout.on('data', (data) => {
            output += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            errorOutput += data.toString();
        });

        pythonProcess.on('close', (code) => {
            fs.unlinkSync(scriptPath);
            if (code !== 0) {
                reject(new Error(`Python exited with code ${code}: ${errorOutput}`));
                return;
            }
            try {
                const result = JSON.parse(output.trim());
                if (result.error) {
                    reject(new Error(result.error));
                } else {
                    resolve(result);
                }
            } catch (err) {
                reject(new Error('Invalid JSON from Python: ' + err.message));
            }
        });

        pythonProcess.on('error', (err) => {
            fs.unlinkSync(scriptPath);
            reject(new Error('Failed to run Python script: ' + err.message));
        });
    });
}

// Route to handle prediction requests
app.post('/predict', async (req, res) => {
    try {
        const { message } = req.body;
        if (!message) return res.status(400).json({ error: 'Message is required' });
        const result = await predictWithPython(message);
        res.json(result);
    } catch (err) {
        console.error('Prediction error:', err.message);
        res.status(500).json({ error: 'Failed to make prediction', details: err.message });
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
