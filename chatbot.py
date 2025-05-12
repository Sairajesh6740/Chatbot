from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

bot = ChatBot("mybot",
storage_adapter='chatterbot.storage.SQLStorageAdapter',
logic_adapters=[
'chatterbot.logic.BestMatch',
'chatterbot.logic.TimeLogicAdapter'],
database_uri='sqlite:///database.db'
)
trainer = ListTrainer(bot)
trainer.train([
"hello",
"hi,how can i help you",
"tell me a joke",
"here is a joke for you what do you call a train carrying bubble gum? a chew-chew train"
])
while True:
    query = input("you : ")
    response = bot.get_response(query)
    print('bot :',response)