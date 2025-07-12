

# TechTrainee - AI-Powered Career Guidance Platform

![WhatsApp Image 2025-07-12 at 18 21 18_d2bc46e1](https://github.com/user-attachments/assets/e5aa36e3-38a9-48d8-a2b8-c51033c8cb69)
![WhatsApp Image 2025-07-12 at 18 21 14_ed0c447d](https://github.com/user-attachments/assets/9b82ee42-bc07-46ba-a220-d5361deaf359)
![WhatsApp Image 2025-07-12 at 18 21 16_fcee9d63](https://github.com/user-attachments/assets/8c5ef40e-ba86-4ea7-a420-811f3edc2dd7)
![WhatsApp Image 2025-07-12 at 18 21 15_3f0b9906](https://github.com/user-attachments/assets/1329dfae-4dc7-4c75-84d1-ad58544a5965)


## Overview
TechTrainee is an intelligent career guidance application that leverages advanced AI technology to help individuals navigate their professional journey. Our platform features a trained AI chatbot that provides personalized career advice and guidance tailored to your unique situation.

## Target Audience
TechTrainee is designed for three distinct user groups:

1. **Job Enthusiasts**: Individuals who are passionate about finding the right career opportunities and want expert guidance on their job search journey.

2. **Skilled but Uncertain**: Professionals who possess valuable skills but are unsure about their career direction or next steps.

3. **Complete Beginners**: People who are neither sure about their skills nor their career goals and need comprehensive guidance from ground zero.

## Features

### AI-Powered Career Chatbot
- **Intelligent Conversations**: Trained AI model that understands career-related queries and provides relevant guidance
- **Semantic Search**: Advanced natural language processing for accurate response matching
- **Keyword Enhancement**: Optimized search capabilities for better query understanding
- **Confidence Scoring**: Response reliability indicators to help users gauge advice quality

### User Experience
- **Clean Web Interface**: Intuitive chat interface accessible through any web browser
- **Real-time Responses**: Instant AI-powered career advice and guidance
- **Cross-platform Compatibility**: Works on desktop, mobile, and tablet devices
- **No Registration Required**: Start getting career guidance immediately

### Technical Features
- **Sentence Transformer Model**: Utilizes 'all-MiniLM-L6-v2' for high-quality text embeddings
- **Cosine Similarity Matching**: Accurate content matching for relevant responses
- **NLTK Integration**: Advanced text processing and keyword extraction
- **Flask Backend**: Robust and scalable web application framework
- **GPU Acceleration**: Optimized for both CPU and GPU processing

## Technology Stack

### Backend
- **Python 3.x**: Core programming language
- **Flask**: Web application framework
- **Sentence Transformers**: AI model for text embeddings
- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **NLTK**: Natural language processing toolkit
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### Frontend
- **HTML5**: Modern web markup
- **CSS3**: Responsive styling
- **JavaScript**: Interactive functionality
- **AJAX**: Asynchronous communication

### AI/ML Components
- **Pre-trained Language Model**: all-MiniLM-L6-v2
- **Cosine Similarity**: Semantic matching algorithm
- **Keyword Extraction**: NLTK-based preprocessing
- **Pickle Serialization**: Model persistence

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Git (for cloning the repository)

### Setup Instructions
1. Clone the repository:
   git clone https://github.com/yourusername/techtrainee.git
   cd techtrainee

2. Install required dependencies:
   pip install flask sentence-transformers torch scikit-learn nltk pandas numpy python-docx

3. Download NLTK data:
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

4. Ensure you have the trained model file:
   Place 'completedmodel.pkl' in the project root directory

5. Run the application:
   python app.py

6. Open your web browser and navigate to:
   http://localhost:5000

## Usage

### Getting Started
1. Open the TechTrainee application in your web browser
2. You'll be greeted by the AI career assistant
3. Type your career-related questions or concerns
4. Receive personalized guidance based on your queries

### Example Queries
- "What skills do I need for a data science career?"
- "How do I transition from marketing to tech?"
- "What are the best programming languages to learn?"
- "How do I prepare for software engineering interviews?"
- "What career path suits someone with my background?"

### Tips for Best Results
- Be specific about your current situation and goals
- Ask follow-up questions to get more detailed guidance
- Mention your current skills and experience level
- Describe your interests and preferences

## File Structure
techtrainee/
│
├── app.py                          # Main Flask application
├── bot.html                        # Web interface
├── career_guide.py                 # Core chatbot training logic
├── career-guide-implementation.py  # Standalone implementation
├── completedmodel.pkl              # Trained AI model (required)
├── README.txt                      # This file
└── requirements.txt                # Python dependencies

## API Endpoints

### POST /chat
Handles chat interactions with the AI assistant.

**Request Body:**
{
  "message": "Your career question here"
}

**Response:**
{
  "response": "AI assistant response",
  "confidence": 0.85
}

### GET /
Serves the main web interface.

## Model Training
The AI model is trained using career guidance data and utilizes:
- Sentence transformer embeddings for semantic understanding
- Keyword extraction for improved query matching
- Cosine similarity for response relevance scoring
- Content preprocessing for optimal performance

## Performance Optimization
- **GPU Acceleration**: Automatic detection and utilization of available GPUs
- **Keyword Indexing**: Fast content retrieval using keyword mapping
- **Efficient Embeddings**: Optimized text encoding for quick responses
- **Caching**: Model persistence for faster startup times

## Browser Compatibility
- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Contributing
We welcome contributions to TechTrainee! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support
For support, questions, or feedback:
- Create an issue on GitHub
- Email: support@techtrainee.com
- Documentation: Available in the repository

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Sentence Transformers library for providing excellent text embedding models
- NLTK community for natural language processing tools
- Flask framework for web application development
- PyTorch team for the deep learning framework

## Version History
- v1.0.0: Initial release with core AI chatbot functionality
- v1.1.0: Enhanced keyword matching and improved response accuracy
- v1.2.0: Added confidence scoring and better error handling

## Future Enhancements
- User profiles and personalized recommendations
- Integration with job boards and career platforms
- Advanced analytics and career progress tracking
- Mobile application development
- Multi-language support

## Disclaimer
TechTrainee provides AI-generated career guidance based on trained data. While we strive for accuracy and relevance, users should consider multiple sources and professional advice for important career decisions.

---

Built with ❤️ for career seekers everywhere.
TechTrainee - Your AI Career Companion
