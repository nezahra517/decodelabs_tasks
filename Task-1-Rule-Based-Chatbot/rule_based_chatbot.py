# Project 1 - Rule Based Chatbot
# knowledge base is a dictionary, runs in a continuous loop

responses = {
    "hi": "hey! how's it going?",
    "hello": "hey! how's it going?",
    "how are you": "i'm doing good, thanks for asking",
    "what is your name": "my name is Bot, this is DecodeLabs project 1",
    "what can you do": "right now i can only reply to a few fixed questions",
    "who made you": "i was built as part of the DecodeLabs internship training",    
    "are you human": "nope, just lines of code running in a loop",
    "help": "try things like hi, how are you, tell me a joke, who made you, or bye to quit",
    "thanks": "no problem!",
    "thank you": "no problem!",
    "good morning": "good morning! hope you have a nice day",
    "good night": "good night, sleep well",
}

exit_words = ["bye", "exit", "quit"]
default_reply = "sorry, i don't understand that"

print("Chatbot is running. Type something to chat, type 'bye' to quit")
print("-" * 50)

while True:
    user_msg = input("You: ")
    user_msg = user_msg.lower().strip()

    if user_msg in exit_words:
        print("Bot: okay, goodbye!")
        break

    elif user_msg == "":
        print("Bot: you didn't type anything")

    else:
        reply = responses.get(user_msg, default_reply)
        print("Bot:", reply)