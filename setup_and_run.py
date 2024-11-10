# setup_and_run.py

import nltk
import ssl

def setup_nltk():
    """Download required NLTK data"""
    try:
        # Handle SSL certificate verification issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Download required NLTK data
        print("Downloading required NLTK data...")
        nltk.download('punkt')
        nltk.download('stopwords')
        print("NLTK setup complete!")
        return True
    except Exception as e:
        print(f"Error during NLTK setup: {e}")
        return False

def run_career_guide():
    """Run the career guide chatbot"""
    from career_guide import CareerGuideBot
    
    try:
        # Initialize bot with your model
        print("Initializing Career Guide...")
        bot = CareerGuideBot("completedmodel.pkl")
        
        print("\nWelcome to Career Guide! Type 'quit' to exit.")
        while True:
            query = input("\nYou: ").strip()
            if query.lower() == 'quit':
                print("Thank you for using Career Guide! Goodbye!")
                break
                
            response, confidence = bot.get_response(query)
            print(f"\nCareer Guide: {response}")
            print(f"Confidence: {confidence:.2f}")
            
    except Exception as e:
        print(f"Error running Career Guide: {e}")

if __name__ == "__main__":
    if setup_nltk():
        run_career_guide()
    else:
        print("Failed to set up NLTK. Please check your internet connection and try again.")