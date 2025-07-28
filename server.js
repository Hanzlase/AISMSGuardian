const express = require('express');
const path = require('path');
const { BayesClassifier } = require('natural');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 8000;

// Middleware
app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(express.static('templates'));

// Initialize classifier with actual training data
const classifier = new BayesClassifier();

// Training data (replace with your actual dataset)
const trainingData = [
  { text: 'Win a free prize now!', classification: 'spam' },
  { text: 'Claim your $1000 reward', classification: 'spam' },
  { text: 'Meeting at 3pm tomorrow', classification: 'ham' },
  { text: 'Your package has shipped', classification: 'ham' }
];

// Train classifier
trainingData.forEach(item => {
  classifier.addDocument(item.text, item.classification);
});
classifier.train();

// Routes
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'templates', 'index.html'));
});

app.post('/predict', async (req, res) => {
  try {
    const { message } = req.body;
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    // Get classification and confidence
    const classification = classifier.classify(message);
    const confidence = classifier.getClassifications(message)[0].value * 100;

    res.json({
      prediction: classification === 'spam' ? 'Spam' : 'Ham',
      probability: `${confidence.toFixed(2)}%`,
      keywords: extractKeywords(message)
    });
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ 
      error: 'Failed to make prediction',
      details: error.message 
    });
  }
});

// Helper function to extract keywords
function extractKeywords(message) {
  const spamKeywords = ['win', 'free', 'prize', 'claim', 'urgent'];
  return spamKeywords.filter(keyword => 
    message.toLowerCase().includes(keyword)
    .map(keyword => ({ word: keyword, score: 0.9 }));
}

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Visit: http://localhost:${PORT}`);
});
