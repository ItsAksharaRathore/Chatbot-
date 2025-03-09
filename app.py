import re
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Download necessary NLTK resources first
# This needs to happen before any NLTK functions are called
print("Downloading required NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
print("Download complete!")

class NLPChatbot:
    def __init__(self, intents_file="intents.json"):
        # Load intents
        try:
            with open(intents_file, 'r') as file:
                self.intents = json.load(file)
        except FileNotFoundError:
            # Create a simple default set of intents
            self.intents = {
                "intents": [
                    {
                        "tag": "greeting",
                        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
                        "responses": ["Hello!", "Hey there!", "Hi! How can I help you today?"]
                    },
                    {
                        "tag": "goodbye",
                        "patterns": ["Bye", "See you later", "Goodbye", "I'm leaving"],
                        "responses": ["Goodbye!", "See you later!", "Take care!"]
                    },
                    {
                        "tag": "thanks",
                        "patterns": ["Thank you", "Thanks", "That's helpful"],
                        "responses": ["You're welcome!", "Happy to help!", "Anytime!"]
                    },
                    {
                        "tag": "help",
                        "patterns": ["Help", "I need help", "Can you help me", "What can you do"],
                        "responses": ["I can answer questions, provide information, or just chat. What do you need help with?"]
                    }
                ]
            }
            print("Created default intents as no intents file was found.")
            
        # Initialize the lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Prepare training data
        self.prepare_training_data()
        
        # Train the model
        self.train_model()
        
        # Session context for maintaining conversation state
        self.context = {}
        
    def prepare_training_data(self):
        """Extract patterns and prepare training data."""
        self.patterns = []
        self.tags = []
        
        for intent in self.intents["intents"]:
            for pattern in intent["patterns"]:
                self.patterns.append(pattern)
                self.tags.append(intent["tag"])
    
    def preprocess_text(self, text):
        """Preprocess text: tokenize, lowercase, remove stopwords, and lemmatize."""
        # Check if text is empty or None
        if not text or not isinstance(text, str):
            return ""
            
        # Tokenize text
        tokens = word_tokenize(text.lower())
        
        # Get English stopwords
        stop_words = set(stopwords.words('english'))
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        
        return " ".join(tokens)
    
    def train_model(self):
        """Train the TF-IDF vectorizer model."""
        # Preprocess all patterns
        processed_patterns = [self.preprocess_text(pattern) for pattern in self.patterns]
        
        # Initialize and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()
        self.X_train = self.vectorizer.fit_transform(processed_patterns)
    
    def predict_intent(self, user_input):
        """Predict the intent of the user input."""
        # Preprocess user input
        processed_input = self.preprocess_text(user_input)
        
        # Vectorize user input
        user_vector = self.vectorizer.transform([processed_input])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(user_vector, self.X_train)
        
        # Get the index of the most similar pattern
        max_similarity_idx = np.argmax(similarities)
        
        # Return the corresponding tag and the similarity score
        similarity_score = similarities[0][max_similarity_idx]
        
        # Only return a match if similarity is above threshold
        if similarity_score > 0.2:
            return self.tags[max_similarity_idx], similarity_score
        else:
            return "unknown", 0.0
    
    def get_response(self, tag):
        """Get a random response for the given intent tag."""
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return "I'm not sure I understand. Could you rephrase that?"
    
    def entity_recognition(self, user_input):
        """Simple rule-based entity recognition."""
        entities = {}
        
        # Extract dates (simple pattern)
        date_pattern = r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'
        dates = re.findall(date_pattern, user_input)
        if dates:
            entities["date"] = dates
            
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, user_input)
        if emails:
            entities["email"] = emails
            
        # Extract numbers
        number_pattern = r'\b\d+\b'
        numbers = re.findall(number_pattern, user_input)
        if numbers:
            entities["number"] = [int(num) for num in numbers]
            
        return entities
    
    def chat(self, user_input):
        """Process user input and return a response."""
        # Check if input is empty
        if not user_input.strip():
            return "I didn't receive any input. How can I help you?"
        
        # Predict intent
        tag, confidence = self.predict_intent(user_input)
        
        # Recognize entities
        entities = self.entity_recognition(user_input)
        
        # Get response based on intent
        response = self.get_response(tag)
        
        # Add entity information if available
        if entities:
            entity_info = "\nI noticed the following information: " + ", ".join(
                [f"{entity_type}: {', '.join(str(e) for e in entity_values)}" 
                 for entity_type, entity_values in entities.items()]
            )
            response += entity_info
            
        return response
    
    def add_intent(self, tag, patterns, responses):
        """Add a new intent to the chatbot."""
        # Check if intent already exists
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                # Update existing intent
                intent["patterns"].extend(patterns)
                intent["responses"].extend(responses)
                break
        else:
            # Add new intent
            self.intents["intents"].append({
                "tag": tag,
                "patterns": patterns,
                "responses": responses
            })
        
        # Update training data
        self.prepare_training_data()
        self.train_model()
        
        return f"Intent '{tag}' has been added/updated with {len(patterns)} patterns and {len(responses)} responses."

# Example usage
if __name__ == "__main__":
    print("Initializing chatbot...")
    chatbot = NLPChatbot()
    
    print("Chatbot: Hello! I'm your NLP chatbot. Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
            
        response = chatbot.chat(user_input)
        print(f"Chatbot: {response}")
        
        # Example of adding a new intent during runtime
        if "teach" in user_input.lower() and "you" in user_input.lower():
            print("Chatbot: I can learn new responses! Please tell me:")
            tag = input("Intent name: ")
            patterns = input("Example questions (comma separated): ").split(",")
            responses = input("Possible responses (comma separated): ").split(",")
            
            result = chatbot.add_intent(tag, patterns, responses)
            print(f"Chatbot: {result}")