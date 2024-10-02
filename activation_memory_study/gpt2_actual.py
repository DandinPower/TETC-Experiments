from transformers import AutoModel, AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B"
context_length = 1024


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print(tokenizer.bos_token)
text = "<|begin_of_text|>" * context_length
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)
output = model(**encoded_input)
