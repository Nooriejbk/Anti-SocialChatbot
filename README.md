# Anti-SocialChatbot

Building a Chatbot

Brief Liturature Review :

Chatbots are “becoming an increasingly valuable asset for companies when dealing with customer service” (Psymbolic Staff. (2019). 7 Common ai techniques used in chatbots. Psymbolic. Available from https://www.psymbolic.com/7-common-ai-techniques-used-in-chatbots/ [Accessed 26/12/2022]) with more and more companies turning to AI chatbots and a staggering predicted chatbot market growth of 500 percent by 2026 (Iuchanka, A. (2022). How do chatbots really work. Itechart. Available from https://www.itechart.com/blog/how-do-chatbots-really-work/ [Accessed 06/01/2023]). By streamlining the process of common customer service requests, companies in the United States can save an estimated $62 billion in sales alone by implementing chatbot capabilities and having their customer service staff only deal with the high value/high risk issues and while their chatbot/s deal with the repetitive or low risk/low value issues. Common enquires such as business hours, simple account management or order tracking can save customer service representatives hours of repetitive labour by outsourcing all that work to a simple chat bot that will recognise the text requests, process it and then return the information all in a matter of seconds.

Users feel seen when an AI chatbot provides instant responses (Iuchanka, A. (2022). How do chatbots really work. Itechart. Available from https://www.itechart.com/blog/how-do-chatbots-really-work/ [Accessed 06/01/2023]) and includes the opportunity for 24/7 service. While most requests can be dealt with by a chatbot there and then, others can be offered a “call back service” while their enquiry will be processed and passed on the customer service team who will deal with it during the company’s work hours. As some of the request may have been processed already, such as customer details and category or general department this request would go to, the only thing left would be for an agent to check the details and respond. Therefore, making the representative’s job just that much easier.

This also allows for the chance to capitalise on the modern preference to online chatting over waiting on business numbers. Chatbot services allow for all round assistance throughout a customer’s journey when online. Providing warm greetings, reminders of items left in shopping baskets (Iuchanka, A. (2022). How do chatbots really work. Itechart. Available from https://www.itechart.com/blog/how-do-chatbots-really-work/ [Accessed 06/01/2023]) and quick and engaging after sale surveys without the potential judgement of the customer service representative who was with you the entire time.

But chatbots aren’t only being used in customer service/retail industries, they can also be used for healthcare and within your own home. An example of a healthcare chatbot is Woebot, “a digital mental health start-up” (Iuchanka, A. (2022). How do chatbots really work. Itechart. Available from https://www.itechart.com/blog/how-do-chatbots-really-work/ [Accessed 06/01/2023]) bringing together Cognitive Behavioural Therapy, Interpersonal Psychology and Dialectical Behaviour Therapy with AI, or more specifically Natural language processing. By using research from certified psychologists, Woebot is designed to aid users when and where therapy can’t but not to replace it. The chatbot format allows for those who don’t have access to therapy at all or for those who need to talk to someone at unsociable hours to have a place to talk, even just until they can. (Woebot Health. (2018). Why we need mental health chatbots. Woebot health. Available from https://woebothealth.com/why-we-need-mental-health-chatbots/ [Accessed 06/01/2023])

Prominent chatbots such as Alexa and Siri who deal with speech requests and commands such as “set a timer for 15 minutes” or “open YouTube” and then process speech into a request into the outcome. These chatbots don’t have the option to connect user to an agent and therefore must rely on their AI to be able to complete the task. With commands like “set timer for 15 minutes” or “turn the lights on”, these AI chatbots are controlled by voice (Freed, A. (2021). Conversational AI : Chatbots that work. Shelter Island, New York : Manning) Using data from the customer service representatives and pattern matching, chatbots can become more sophisticated and human-like by using deep learning, machine learning and natural language processing (NLP) (Iuchanka, A. (2022). How do chatbots really work. Itechart. Available from https://www.itechart.com/blog/how-do-chatbots-really-work/ [Accessed 06/01/2023]).

AI Techniques :

The most fundamental part of a chatbot is the user is communicating with the AI and the AI to respond, I have found 3 ways of doing this : Natural language processing, machine learning and deep learning. The examples I've see of natural language processing include functions or data sets that help the program understand each word or the structure of the text inputed. This is the most complex option I found as it includes teaching the AI how langauge works. Machine learning is the greater subset that includes natural language processing and deep learning so a machine learning chatbot would be a chatbot doesn't delve into the subsets of deep learning and natural language processing and so is a lot simpler than the 2. Then there is deep learning with includes the AI learning from it's past experiences.

In this specific chat bot, I wanted to do something different. My inital thought was to give the chat bot a super random unimportant topic but I realised that would include me also learning about that topic to some extent and so thought maybe not. Whilst looking for datasets to inspire me I found a chatbot idea from the nltk modules website and decided that was it. I'm going to make a rude chatbot, that doesn't want to chat.

Machine learning is when a computer system (AI) can learn and adapt using algorithms and statistics using patterns in data. For strengths, I will be giving the AI all possible inputs and appropriate responses. With that I am also able to make sure the responses aren't just contexually appropriate but societally appropriate in the sense that the AI is not learning any phrases that it shouldn't be saying and aren't being seen by me first. This means that the chatbot will be limited in what it can say and will only be able to learn from the dataset and not from it's previous iterations. TBy limiting this, I am making sure the chatbot isn't learnig to be nice as well as not learning any foul language or slurs, if any users choose to use them. With this, the chatbot will only learn from the dataset once and just be stuck at that point as it doesn't learn anything past that point. An example of this could be the chat bot having a collection of greetings, questions and statements and the desired responses for each such as the user saying "hello" and the chat bot being able to choose at random the reponses of "What do you want?", "I was enjoying the silence before you arrived" and "Hi... i guess". Every time the AI decides the users input is a greeting, you will only get one of the responses I have written.

With Natural lanaguage processing, you are teaching the bot to understand how a language works and how to string sentences together. If this chatbot was intended for prolonged use, maybe have lot of people interact with it, then I would need to incorporate a system of learning everything that users are saying to the chatbot to improve its responses. Doing so would make the chatbots responses more unique and may possibly teach my chat bot to be kind if the chat bot only had nice chat users and was learning from that. On the other hand, if a chatbot is mean and is also learning how to be more mean, where's the limit? I would also have to create a limit and I think I'm happy with my chatbot not knowing too much and not having to worry about the chatbot potentially using offensive language it's learnt from other users. Natural language processing would be a great tehcnique to explore if the chatbot was made for other purposes, such as medical enquiries or even just general chat but because were teaching it to be negative, I don't want to see my creation become a monster. Natural langauge processing all the inputs would studied and the AI's word bank (words that the AI knows) would constantly be growing and it would have it's own ways of distingusihing whether words are verbs or adjectives. It would sting sentences together on its own without the need for human interferance. This could result in the AI learning profanity and using it as a verb or an adjective. I like to use the example of a child, if a child suddenly learned a swear word, their parent or a grown up could tell them the context of the word, the moral inplications of using that word and whether or not the child should be using that word at all. I would either have to prepare the AI before hand and teach it all the swear words I know (unfortunately, my knowledge is limited) and then tell it to not use it or find a way for the AI to understand suitable times to use it and I don't my own ability of teaching a child that, let alone an AI.

Deep learning I feel is a mid point between natural language processing and just general machine learning. It requires the AI to practice it's predicting skills and then start stringing sentences together using what it's already learned. Strengths, the AI would learn now ways to express its amguish or displeasure. I could give the AI a dataset with all the dialogue of a popular angry character and essentially make an AI version of this character that you could talk to. However, it might go rouge. In movies, all the dialogue is written by writers. Its well thoughout, they're thinking about their target audience, they could slip a few adult jokes in there but there is a human writer supervising everything that is said and making sure they don't go too far. I can't predict what an AI strings together and loading it with mildly rude responses, could not only get the AI in trouble but me as well and I don't want the AI to string any sentences itself.

The technique I have chosen to explore in depth for my prototype (you can probaly guess) is machine learning as I still feel this technique is the most relevant to the task and my task doesn't really need too much learning with it, just a lot of rude responses it can scroll through and that I can pre-approve. I think with the concept of a rude chatbot, there is the potential that the chatbot learns things it shouldn't say, much like a child hearing a word and repeating it without the word being taught to them with context and approriate times to use it and with that I don't want my chatbot to become too rude and so not learnanything except for what i've told it exactly. I think thats best for everyone.

Building the chatbot

As the program is a chatbot, I will need to train the AI on typical phrases and words used in a conversation as well as apropriate responses. To do this I will need to create a file filled with greetings, typical questions and responses and maybe a few other conversation info or giving the AI a topic to know a lot about so the ai can deal with more converstaion scenarios.

The input and output is text, the ai needs to recognise the text inputed and work out appropriate responces to the users input. I found it quite hard to find a chatbot dataset that was organised in the way I wanted, a lot of them were just big blocks of text from whatsapp conversations or other chatbot receipts and so I had to first work out what would be the best format to structure my dataset. I found a chatbot dataset that seperated the data into intents, inputs and responses(Tech with Tim. (?). Simple AI Chat Bot. Available from https://www.techwithtim.net/tutorials/ai-chatbot/part-1/ [Accessed 31/12/2022]. In other examples of chatbots, they would've required me to program the bot to learn from the chat dataset but I couldn't find an example of that along with them turning it in to a interactive chat bot.

When it comes to the chat bot, it will need to have a clear set of different types of conversation inputs as well as out puts. The first example of a chatbot I found was a medical chatbot and had the typical greetings at the top of the dataset but then went into things a person looking for medical assisstance would ask for or questions about medicine or gp opening times.

#imports
import nltk
from nltk.stem.lancaster import LancasterStemmer 
stemmer = LancasterStemmer()
import os.path
from os import path
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
C:\Users\queen\anaconda3\lib\site-packages\scipy\__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.24.1
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
WARNING:tensorflow:From C:\Users\queen\AppData\Roaming\Python\Python39\site-packages\tensorflow\python\compat\v2_compat.py:107: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
curses is not supported on this machine (please install/reinstall curses for an optimal experience)
#import data file
with open("intents.json") as file:
    data = json.load(file)
    
#Pre-process data files
​
try: #if already processed - don't do it again
    with open("data.pickle", "rb") as f:
        prossWords, prossTags, training, output = pickle.load(f)
​
except:
    prossWords = []
    prossTags = []
    docs_x = []
    docs_y = []
​
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wordToke = nltk.word_tokenize(pattern)
            prossWords.extend(wordToke)
            docs_x.append(wordToke)
            docs_y.append(intent["tag"])
​
        if intent["tag"] not in prossTags:
            prossTags.append(intent["tag"])
​
    prossWords = [stemmer.stem(w.lower()) for w in prossWords if w not in "?"]
    prossWords = sorted(list(set(words)))
​
    prossTags = sorted(prossTags)
​
#Prepare training data
    training = []
    output = []
​
    out_empty = [0 for _ in range(len(prossTags))]
​
    for x, doc in enumerate(docs_x):
        bag = []
​
        wordToke = [stemmer.stem(w) for w in doc]
​
        for w in prossWords:
            if w in wordToke:
                bag.append(1)
            else:
                bag.append(0)
​
        output_row = out_empty[:]
        output_row[prossTags.index(docs_y[x])] = 1
​
        training.append(bag)
        output.append(output_row)
​
    training = numpy.array(training)
    output = numpy.array(output)
    
    with open("data.pickle", "wb") as f:
        pickle.dump((prosswords, prossTags, training, output),f)
​
#Training Neural Network
​
tensorflow.compat.v1.reset_default_graph()
​
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, 24)
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
                    
model = tflearn.DNN(net)   
​
​
if path.exists('checkpoint'):
    model.load('model.tflearn')
else:
    model.fit(training, output, n_epoch=1000, batch_size=9, show_metric=True)
    model.save('model.tflearn')
    
def bag_of_words(s, prossWords):
    bag = [0 for _ in range(len(prossWords))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(prossWords):
            if w == se :
                bag[i] = (1)
            
    return numpy.array(bag)
​
def chat():
    print ("Start talking with the bot, type quit to stop")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        results = model.predict([bag_of_words(inp, prossWords)])[0]
        results_index = numpy.argmax(results)
        tag = prossTags[results_index]
        
        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                
            print("Chatb*t**: " + random.choice(responses)) 
        else:
            print("No idea what you're talking about")
        
chat()
INFO:tensorflow:Restoring parameters from C:\Users\queen\ai\Chatbot\model.tflearn
Start talking with the bot, type quit to stop
You:  hi
Chatb*t**: 'Hello'? How original...
You:  how are you
Chatb*t**: not very good at conversation starters, are you?
You:  whats your name
Testing:

The appropriate testing methods for my project is to test if the ai can recognise a input that was already in the training data, one thats similar (not exact wording) and one in completely different wording but same meaning. I went through and did one of each field above for each intent tag and got a correct response roughly 66% of the time (16/24). The ai was able to recognise all intents from the training set, a few from the similar and a few from the different set. I think that is because some intents do have a wide range of options (for example please phrases that don't use please were unrecogniseable but fancy ways to say hello were easy to pick up) and some intents only have a limited range of options.

I expected the bot to have very obvious faults, by that I meant it being obvious when it chose the wrong intent but because the general tone of the chatbot is negative and dismissive, most of them worked even in the wrong context. I think because of the style of bot I chose, I probably didn't even need to make so many intent categories and could've just filled the bot with insults. As the chatbot was made with the intention of being rude and a good percentage of the outputs make sense in the context of the user's input, i would say I've found a good way to hide that the chatbot is 66% correct with responses.

Evaluate results:

With an accuracy rate of 66% from testing but an even higher percentage of responses still making sense even when wrong, I'd say the chat bot was a success. Considering I did more or less make the data set myself, I was expecting there to be faults to the accuracy regardless as I didn't entirely know what a dataset was supposed to look like and only found 2 partial examples of organised datasets and full chatbot code. I think if I had a prebuilt data set and trained the chatbot for longer, the accuracy rate would be higher and it could potentially be unnoticeable when the chat bot was wrong.

With expected results, I wanted the chatbot to express its anguish with being forced to chat, refuse to tell you it's name and recluctantly tell you it's age whilst explaining how asking a chatbot for its age is weird. I had created about 8 categories of user text to responses and a option for if the chatbot couldn't work out which response to use. I feel like if you didn't know what resonse was in what category of input, you'd probaly assume the chatbot understood what you were saying about 80% of the time. As someone who does know whether the response was correct, its about 50% correct. Also the bot confuses a lot of unique responses with what could be a goodbye and will just randomly tell users to leave and never come back, which is quite obviously the wrong response but on the goal of being a rude chat bot, it works.

The method I chose was to just have a couple conversations with the chat bot and see how much of the chat was coherant and made sense on a call and response level. A lot of the responses work with other categories but there is a few cases that don't work at all and the chatbot will say "I love it when you beg" which is supposed to be a response for and phrase that includes "please" but will say it if you say anything that sounds like youre agreeing like yes, probably or thank you. I think in terms of the overall goal for the bot, it's pretty rude and it's managed to get the responses about 80% coherant on a test of 20 user inputs, 8 similar (not identical) to the ones given in each intent category and 12 random.
