from transformers import pipeline

chatbot = pipeline(model="facebook/blenderbot-400M-distill")

conversation = chatbot("Hi I'm Shaw, how are you?")

print(conversation)