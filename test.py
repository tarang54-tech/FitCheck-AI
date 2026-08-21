from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=10000)

print(client.admin.command("ping"))