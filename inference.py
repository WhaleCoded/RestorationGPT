from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("training/model")
tokenizer = AutoTokenizer.from_pretrained("training/model")

input_text = "Who is Joseph Smith?"

input_ids = tokenizer.encode(input_text, return_tensors='pt')

# generate text until the output length (which includes the context length) reaches 50
sample_outputs = model.generate(
    input_ids,
    do_sample=True,
    max_length=50,
    top_k=50,
    top_p=0.95,
    num_return_sequences=3
)

print("Output:\n" + 20 * '-')
for i, sample_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))