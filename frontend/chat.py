import random
from collections import deque
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Initialize a stemmer for basic NLP
stemmer = PorterStemmer()

# A deque to store the chat history (you can adjust the length for how much memory you want to use)
chat_history = deque(maxlen=5)

# Predefined responses for different keywords
response_dict = {
    "greeting": ["Hello! How can I assist you today?", "Hi there! What can I do for you?", "Hey! How can I help?"],
    "goodbye": ["Goodbye! Have a great day!", "Bye! Take care.", "Farewell! Hope to see you soon!"],
    "help": ["I'm here to assist you. What do you need help with?", "How can I help you today?", "Feel free to ask anything!"],
    "weather": ["It seems like a nice day, isn't it?", "I hope the weather is good where you are.", "I'm not sure, but it feels like a sunny day!"],
    "default": ["Sorry, I didn't quite get that.", "Could you rephrase that?", "I'm not sure I understand. Can you clarify?"]
}

# Function to preprocess user input (tokenization + stemming)
def preprocess_message(message):
    tokens = word_tokenize(message.lower())  # Tokenize and lower the message
    return [stemmer.stem(token) for token in tokens]  # Stem each token

# Function to check for keywords in the user's message
def detect_intent(tokens):
    keywords = {
        "greeting": ["hello", "hi", "hey"],
        "goodbye": ["bye", "goodbye", "farewell"],
        "help": ["help", "assist", "support"],
        "weather": ["weather", "sunny", "rain", "cloudy"]
    }

    for intent, keyword_list in keywords.items():
        if any(stemmer.stem(word) in tokens for word in keyword_list):
            return intent
    return "default"

# The main chatbot logic
def chatbot(user_message):
    tokens = preprocess_message(user_message)
    intent = detect_intent(tokens)

    # Store the message in the chat history
    chat_history.append(user_message)

    # Generate a response based on detected intent
    response = random.choice(response_dict.get(intent, response_dict["default"]))

    # Add contextual responses based on chat history
    if len(chat_history) > 1 and "help" in chat_history[-2].lower():
        response += " It seems like you were asking for help earlier. Would you like to know more?"
    
    return response
