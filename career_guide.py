# Install required packages
# !pip install python-docx
# !pip install sentence-transformers
# !pip install torch
# !pip install nltk

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from google.colab import files
import os
from docx import Document
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class DataManager:
    def __init__(self):
        self.datasets = {}
        self.combined_data = pd.DataFrame()
        self.keywords_dict = {}

    def add_dataset(self, name, df):
        """Add a new dataset or update existing one"""
        self.datasets[name] = df
        self.update_combined_data()
        self.extract_keywords()
        print(f"Dataset '{name}' added successfully with {len(df)} entries")

    def update_combined_data(self):
        """Update the combined dataset"""
        if self.datasets:
            self.combined_data = pd.concat(self.datasets.values(), ignore_index=True)
            print(f"Total entries in combined dataset: {len(self.combined_data)}")

    def extract_keywords(self):
        """Extract and store keywords from content"""
        stop_words = set(stopwords.words('english'))
        
        for idx, row in self.combined_data.iterrows():
            # Tokenize and clean the content
            tokens = word_tokenize(str(row['content']).lower())
            # Remove stopwords and non-alphabetic tokens
            keywords = [word for word in tokens if word.isalnum() and 
                       word not in stop_words and len(word) > 2]
            # Store keywords with their corresponding content index
            for keyword in keywords:
                if keyword not in self.keywords_dict:
                    self.keywords_dict[keyword] = set()
                self.keywords_dict[keyword].add(idx)

    def list_datasets(self):
        """List all available datasets"""
        print("\nAvailable datasets:")
        for name, df in self.datasets.items():
            print(f"- {name}: {len(df)} entries")
        print(f"Total combined entries: {len(self.combined_data)}")

class ColabDataLoader:
    @staticmethod
    def upload_file():
        """Upload file through Colab interface"""
        print("Please upload your dataset file (Excel/Word/TXT)")
        uploaded = files.upload()
        return next(iter(uploaded))

    @staticmethod
    def parse_txt_file(file_name):
        """Parse text data from a general text file"""
        qa_pairs = []
        try:
            with open(file_name, 'r', encoding='utf-8') as file:
                lines = [line.strip() for line in file.readlines()]

            for i, line in enumerate(lines):
                if line:
                    qa_pairs.append({
                        'content': line,
                        'context': f"Line {i+1}"
                    })

            if not qa_pairs:
                raise ValueError("No content found in the text file")

            return pd.DataFrame(qa_pairs)

        except Exception as e:
            print(f"Error parsing text file: {e}")
            return None

    @staticmethod
    def load_file(file_name):
        """Load data from uploaded file (Excel, Word, or TXT)"""
        try:
            file_extension = os.path.splitext(file_name)[1].lower()

            if file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_name)
                df.columns = ['content'] + list(df.columns[1:])  # Adjusting for different formats
                return df

            elif file_extension == '.docx':
                doc = Document(file_name)
                content_pairs = [{'content': para.text.strip(), 'context': f"Paragraph {i+1}"}
                                 for i, para in enumerate(doc.paragraphs) if para.text.strip()]

                if not content_pairs:
                    raise ValueError("No content found in the Word document")

                return pd.DataFrame(content_pairs)

            elif file_extension == '.txt':
                return ColabDataLoader.parse_txt_file(file_name)

            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

        except Exception as e:
            print(f"Error loading file: {e}")
            return None

class ColabChatbot:
    def __init__(self, bot_name="Career Guide"):
        self.bot_name = bot_name
        display_welcome_banner()

        print("Initializing chatbot components...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        print("Loading sentence transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        self.data_manager = DataManager()
        self.content_embeddings = None
        self.is_trained = False
        print("Initialization complete! Let's get started!\n")

    def prepare_data(self, df):
        """Prepare and validate the training data"""
        try:
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df, columns=['content'])

            df = df[['content']].dropna()
            df['content'] = df['content'].astype(str).apply(lambda x: x.strip())
            df = df[df['content'].str.len() > 0].drop_duplicates()

            print(f"Prepared {len(df)} valid examples")
            return df

        except Exception as e:
            raise ValueError(f"Error preparing data: {str(e)}")

    def add_training_data(self, df, name=None):
        """Add new training data to existing dataset"""
        try:
            df = self.prepare_data(df)
            name = name or f"dataset_{len(self.data_manager.datasets) + 1}"
            self.data_manager.add_dataset(name, df)
            self.is_trained = False
            return True
        except Exception as e:
            print(f"Error adding training data: {e}")
            return False

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
        """Get chatbot response for the user query with keyword enhancement"""
        if not self.is_trained:
            return f"{self.bot_name} has not been trained yet!"

        try:
            start_time = time.time()
            
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

            end_time = time.time()
            print(f"Response generated in {end_time - start_time:.2f} seconds")
            print(f"Confidence score: {max_similarity:.2f}")

            if max_similarity >= threshold:
                answer = self.data_manager.combined_data.iloc[max_index]['content']
                return answer
            else:
                similar_keywords = ", ".join(keywords[:3]) if keywords else "none"
                return f"I couldn't find a highly relevant answer. Keywords detected: {similar_keywords}. Please try rephrasing your question."

        except Exception as e:
            return f"An error occurred: {e}"

    def train_model(self):
        """Generate embeddings for all content in the dataset"""
        if len(self.data_manager.combined_data) == 0:
            print("No training data available!")
            return False

        print("Generating embeddings for content...")
        try:
            start_time = time.time()
            content_list = self.data_manager.combined_data['content'].tolist()
            self.content_embeddings = self.model.encode(
                content_list,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=True
            )
            self.is_trained = True
            end_time = time.time()
            print(f"Training completed successfully in {end_time - start_time:.2f} seconds!")
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False

    def save_model(self, filename="trained_model.pkl"):
        """Save the chatbot model to a file"""
        if not self.is_trained:
            print("The model needs to be trained before saving!")
            return
        try:
            with open(filename, 'wb') as file:
                pickle.dump((self.content_embeddings, self.data_manager), file)
            print(f"Model saved as {filename}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filename):
        """Load the chatbot model from a file"""
        try:
            with open(filename, 'rb') as file:
                self.content_embeddings, self.data_manager = pickle.load(file)
            self.is_trained = True
            print(f"Model loaded from {filename}")
        except Exception as e:
            print(f"Error loading model: {e}")

def display_welcome_banner():
    """Display a welcome banner for Career Guide Chatbot"""
    banner = """
    ===============================================
     ____                           ____       _     _      
    / ___|__ _ _ __ ___  ___ _ __ / ___|_   _(_) __| | ___ 
   | |   / _` | '__/ _ \/ _ \ '__| |  _| | | | |/ _` |/ _ \\
   | |__| (_| | | |  __/  __/ |  | |_| | |_| | | (_| |  __/
    \____\__,_|_|  \___|\___|_|   \____|\__,_|_|\__,_|\___|
                                                            
    Your AI Career Guidance Assistant
    ===============================================
    """
    print(banner)

def run_chatbot():
    chatbot = ColabChatbot()

    while True:
        print("\nCareer Guide Menu:")
        print("1. Add training data (Excel/Word/TXT)")
        print("2. List loaded datasets")
        print("3. Train model")
        print("4. Start conversation")
        print("5. Save model")
        print("6. Load model")
        print("7. Exit")

        try:
            choice = input("\nPlease enter your choice (1-7): ").strip()

            if choice == '1':
                print("\nSupported formats:")
                print("Excel (.xlsx, .xls), Word (.docx), Text (.txt)")
                file_name = ColabDataLoader.upload_file()
                df = ColabDataLoader.load_file(file_name)
                if df is not None:
                    dataset_name = input("Enter a name for this dataset (or press Enter for default): ").strip()
                    chatbot.add_training_data(df, name=dataset_name)
            elif choice == '2':
                chatbot.data_manager.list_datasets()
            elif choice == '3':
                chatbot.train_model()
            elif choice == '4':
                print("\nChat started! Type 'quit' to return to menu.")
                print("You can now ask questions using keywords or complete sentences.")
                while True:
                    query = input("\nYou: ").strip()
                    if query.lower() == 'quit':
                        break
                    response = chatbot.get_response(query)
                    print(f"\n{chatbot.bot_name}: {response}")
            elif choice == '5':
                file_name = input("Enter filename to save the model (default: trained_model.pkl): ").strip() or "trained_model.pkl"
                chatbot.save_model(file_name)
            elif choice == '6':
                file_name = input("Enter filename to load the model: ").strip()
                chatbot.load_model(file_name)
            elif choice == '7':
                print("Thank you for using tech trainee! Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_chatbot()