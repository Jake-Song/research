from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = "Qwen/Qwen3.5-2B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
text = "I hate this movie, it was terrible."
inputs = tokenizer(text, return_tensors="pt")
print(tokenizer.decode(inputs['input_ids'][0][-1]))