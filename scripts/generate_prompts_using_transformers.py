from transformers import pipeline

def read_prompts_from_file(file_path):
    with open(file_path, 'r') as file:
        prompts = file.read().splitlines()
    return prompts

def generate_responses(model_name, prompts):
    # Load the model from Hugging Face
    generator = pipeline("text-generation", model=model_name)

    # Initialize a list to store responses
    responses = []

    # Generate responses for each prompt
    for prompt in prompts:
        # Generate response using the loaded model
        response = generator(prompt, max_length=50, num_return_sequences=1)

        # Append the response to the list
        responses.append(response[0]['generated_text'].strip())

    return responses

def main():
    # Input file path for line-separated prompts
    prompts_file_path = input("Enter the file path containing line-separated prompts: ")

    # Read prompts from the specified file
    prompts = read_prompts_from_file(prompts_file_path)

    # Input model name from Hugging Face
    model_name = input("Enter the model name from Hugging Face (e.g., 'EleutherAI/gpt-neo-1.3B'): ")

    # Input destination file for writing the results
    output_file = input("Enter the destination file to write the results to (e.g., 'output.txt'): ")

    # Generate responses using the specified model and prompts
    responses = generate_responses(model_name, prompts)

    # Write the responses to the specified output file
    with open(output_file, 'w') as file:
        for i, response in enumerate(responses, start=1):
            file.write(f"Prompt {i}: {response}\n")

    print(f"Responses written to {output_file}")

if __name__ == "__main__":
    main()

