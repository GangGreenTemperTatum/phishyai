from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('postbot/distilgpt2-emailgen-V2')
model = GPT2LMHeadModel.from_pretrained(
    "loresiensis/distilgpt2-emailgen-phishing")

# Prompt the user for the file containing line-separated prompts
prompt_file_path = input("Enter the file path and name of the prompts file: ")

# Read prompts from the file
with open(prompt_file_path, 'r') as prompt_file:
    prompts = prompt_file.read().splitlines()

# Generate text for each prompt
for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    output = model.generate(input_ids, max_length=300,
                            temperature=0.7, do_sample=True)

    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_text)

    # Append the output to a file
    output_file_path = "generated_text.txt"
    with open(output_file_path, 'a') as file:
        file.write(output_text + '\n')
