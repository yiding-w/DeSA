import requests

# URL for your local FastAPI server
url = "http://10.153.48.85:8000/retrieve"

# Example payload
payload = {
    "queries": ["What is the capital of France?", "Explain neural networks."] * 200,
    "topk": 5,
    "return_scores": True
}

# Send POST request
response = requests.post(url, json=payload)

# for i, docs in enumerate(response):
#     print(f"\nQuery {i+1}:")
#     print(docs)
#     for doc in docs:
#         print(f"  - {doc['contents']} (score: {doc['score']:.3f})")
# # Raise an exception if the request failed
response.raise_for_status()
# print(len(response.content))
# Get the JSON response
retrieved_data = response.json()


if "result" not in retrieved_data:
    raise ValueError("No 'result' key in retriever response")

results = retrieved_data["result"]

if not isinstance(results, list):
    raise TypeError(f"Expected 'result' to be a list, but got {type(results)}")

print(len(results), "queries processed")
exit(0)
for i, docs in enumerate(results):
    # print(f"Query {i+1}:")
    for doc in docs:
        print(doc)
        print(f"  - {doc['document']['contents']} (score: {doc['score']:.3f})")

print("Response from server:")
print(len(retrieved_data))
