from flask import Flask, request, jsonify, send_from_directory
import logging
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

class DataManager:
    def __init__(self):
        self.datasets = {}
        self.combined_data = pd.DataFrame()
        self.keywords_dict = {}

class CareerGuideBot:
    def __init__(self, model_path):
        """Initialize the chatbot with a pre-trained model"""
        self.bot_name = "Career Guide"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load the sentence transformer model
        logger.info("Loading sentence transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        # Initialize DataManager
        self.data_manager = DataManager()
        
        # Load the pre-trained model data
        self.load_model(model_path)
        logger.info("Initialization complete!")

    def preprocess_query(self, query):
        """Preprocess the query to extract keywords"""
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(query.lower())
        keywords = [word for word in tokens if word.isalnum() and 
                   word not in stop_words and len(word) > 2]
        return keywords

    def get_relevant_indices(self, keywords):
        """Get indices of content containing any of the keywords"""
        relevant_indices = set()
        for keyword in keywords:
            if keyword in self.data_manager.keywords_dict:
                relevant_indices.update(self.data_manager.keywords_dict[keyword])
        return list(relevant_indices)

    def get_response(self, query, threshold=0.5):
        """Get chatbot response for the user query"""
        try:
            # Extract keywords from query
            keywords = self.preprocess_query(query)
            
            # Get relevant content indices based on keywords
            relevant_indices = self.get_relevant_indices(keywords)
            
            if not relevant_indices:
                # If no keyword matches, fall back to full semantic search
                query_embedding = self.model.encode(query, convert_to_tensor=True, device=self.device)
                similarities = cosine_similarity(
                    query_embedding.cpu().numpy().reshape(1, -1), 
                    self.content_embeddings.cpu().numpy()
                ).flatten()
            else:
                # Get embeddings only for relevant content
                relevant_content = [self.data_manager.combined_data.iloc[idx]['content'] 
                                  for idx in relevant_indices]
                relevant_embeddings = self.model.encode(
                    relevant_content,
                    convert_to_tensor=True,
                    device=self.device
                )
                
                # Calculate similarities only for relevant content
                query_embedding = self.model.encode(query, convert_to_tensor=True, device=self.device)
                similarities = cosine_similarity(
                    query_embedding.cpu().numpy().reshape(1, -1), 
                    relevant_embeddings.cpu().numpy()
                ).flatten()
                
                # Map similarity scores back to original indices
                full_similarities = np.zeros(len(self.data_manager.combined_data))
                for sim_idx, orig_idx in enumerate(relevant_indices):
                    full_similarities[orig_idx] = similarities[sim_idx]
                similarities = full_similarities

            max_index = np.argmax(similarities)
            max_similarity = similarities[max_index]

            if max_similarity >= threshold:
                answer = self.data_manager.combined_data.iloc[max_index]['content']
                return answer, float(max_similarity)
            else:
                similar_keywords = ", ".join(keywords[:3]) if keywords else "none"
                return f"I couldn't find a highly relevant answer. Keywords detected: {similar_keywords}. Please try rephrasing your question.", float(max_similarity)

        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            return f"An error occurred: {str(e)}", 0.0

    def load_model(self, filename):
        """Load the pre-trained chatbot model"""
        try:
            with open(filename, 'rb') as file:
                self.content_embeddings, self.data_manager = pickle.load(file)
            logger.info(f"Model loaded successfully from {filename}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise Exception(f"Error loading model: {str(e)}")

# Initialize the chatbot
try:
    logger.info("Initializing CareerGuideBot...")
    chatbot = CareerGuideBot("completedmodel.pkl")
    logger.info("CareerGuideBot initialized successfully!")
except Exception as e:
    logger.error(f"Error initializing CareerGuideBot: {str(e)}")
    chatbot = None

@app.route('/')
def home():
    return send_from_directory('.', 'bot.html')

@app.route('/chat', methods=['POST'])
def chat():
    if chatbot is None:
        logger.error("Chatbot not initialized")
        return jsonify({'error': 'Chatbot not initialized'}), 500
        
    try:
        data = request.json
        message = data.get('message', '')
        logger.info(f"Received message: {message}")

        # Get response using the get_response method
        response, confidence = chatbot.get_response(message)
        
        logger.info(f"Response generated with confidence: {confidence}")
        
        return jsonify({
            'response': response,
            'confidence': confidence
        })
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error: {str(e)}'}), 500

# Add CORS support
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    app.run(debug=True)