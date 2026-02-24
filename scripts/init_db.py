import chromadb

# Initialize the persistent memory in the 'db' folder
client = chromadb.PersistentClient(path="db/chromadb")

# Create a collection called 'students'
# If it already exists, this will just load it
collection = client.get_or_create_collection(name="students")

print(f"📦 Database Initialized. Current Student Count: {collection.count()}")