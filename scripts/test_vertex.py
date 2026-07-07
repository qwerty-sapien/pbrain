from pb.llm.gemini import get_client

client = get_client()
print("Available:", client.is_available())
if client.is_available():
    res = client.generate("Hello, are you there?")
    print(res)
