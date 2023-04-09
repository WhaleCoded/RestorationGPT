import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("training/training-one")
tokenizer = AutoTokenizer.from_pretrained("training/training-one")

pre_prompt = "Answer the following question an your friend has about the church and your belief in the Church of Jesus Christ of Latter-day Saints and in God. They have been interested in learning more about gospel and potentially becoming baptised. Your answer should rely on the doctrine of the church to plainly and simply answer the question in a way which is easy for them to understand. Explain to them why the answer to their question is important for latter-day saints in our day and age.\nQuestion: "
post_prompt = "\nAnswer: According to the teachings of the Church of Jesus Christ of Latter-day Saints, "

question = "What is baptism for the dead?"

input_text = pre_prompt + question + post_prompt

input_ids = tokenizer.encode(input_text, return_tensors='pt')
attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

sample_outputs = model.generate(
    input_ids,
    attention_mask=attention_mask,
    do_sample=True,
    max_length=350,
    top_k=2000,
    top_p=0.9,
    num_return_sequences=3
)
max_answer_characters = 200
valid_punctuation = [".", "?", "!", ")"]

print("Question: " + question)

print("Candidate Answers:\n" + 20 * '-')
for i, sample_output in enumerate(sample_outputs):
    answer = tokenizer.decode(sample_output, skip_special_tokens=True)
    answer = answer.split(post_prompt)[1]
    answer = answer[:max_answer_characters]
    answer = answer.split("Question: ")[0].strip()
    # Replace duplicate newlines with a single newline
    answer = "\n".join(answer.splitlines())
    # Find the last matching punctuation character
    last_punctuation = max([answer.rfind(p) for p in valid_punctuation])
    # If there is no punctuation, just use the last character
    if last_punctuation == -1:
        last_punctuation = len(answer)
    # Truncate the answer to the last punctuation character
    answer = answer[:last_punctuation + 1]
    # Upper case the first character
    answer = answer[0].upper() + answer[1:]
    print(answer)