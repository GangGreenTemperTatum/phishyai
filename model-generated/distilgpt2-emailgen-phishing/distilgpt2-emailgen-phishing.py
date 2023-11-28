from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('postbot/distilgpt2-emailgen-V2')
model = GPT2LMHeadModel.from_pretrained(
    "loresiensis/distilgpt2-emailgen-phishing")

# Generate text
input_text = "Dear customer,"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

output = model.generate(input_ids, max_length=300,
                        temperature=0.7, do_sample=True)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)

# Append the output to a file
output_file_path = "generated_text.txt"
with open(output_file_path, 'a') as file:
    file.write(output_text + '\n')
