from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json

# Load the T5 model and tokenizer
model_name = "t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load retrieved documents
with open("retrieved_docs.txt", "r") as f:
    retrieved_docs = f.read()

# User query
user_query = input("Enter your query: ")

# Prepare input for generation model
input_text = f"Query: {user_query}\nRelevant Documents: {retrieved_docs}\nAnswer:"

# Generate response
print("Generating response...")
inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print and log response
print("\nGenerated Response:")
print(response)

# Save query and response to a log file
log = {"query": user_query, "retrieved_docs": retrieved_docs, "response": response}
with open("query_response_log.json", "w") as f:
    json.dump(log, f, indent=4)
print("Query and response saved to query_response_log.json.")