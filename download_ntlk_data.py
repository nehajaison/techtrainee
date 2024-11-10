# download_nltk_data.py

import nltk
import ssl

def download_nltk_data():
    try:
        # Disable SSL verification if needed
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download all required NLTK data
        print("Downloading NLTK data...")
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        print("NLTK data downloaded successfully!")
        
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

if __name__ == "__main__":
    download_nltk_data()