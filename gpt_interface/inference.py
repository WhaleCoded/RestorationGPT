import torch

pre_prompt = "Answer the following question your friend has about the church and your belief in the Church of Jesus Christ of Latter-day Saints and in God. They have been interested in learning more about gospel and potentially becoming baptised. Your answer should rely on the doctrine of the church to plainly and simply answer the question in a way which is easy for them to understand. Explain to them why the answer to their question is important for latter-day saints in our day and age.\nQuestion: "
post_prompt = "\nAnswer: According to the teachings of the Church of Jesus Christ of Latter-day Saints, "


def get_answer(prompt, model, tokenizer, model_version):
    if model_version == 1:
        return get_answer_v1(prompt, model, tokenizer)
    elif model_version == 2:
        return get_answer_v2(prompt, model, tokenizer)
    else:
        return get_answer_v3(prompt, model, tokenizer)


def get_answer_v3(prompt, model, tokenizer):
    pre_prompt = "The Church places great emphasis on knowledge and on the importance of being well informed about Church history, doctrine, and practices. Ongoing historical research, revisions of the Church’s curriculum, and the use of new technologies allowing a more systematic and thorough study of scriptures have all been pursued by the Church to that end. We again encourage members to study the Gospel Topics essays below. Study the scriptures and pray to know if they are true.\nSection One\n"
    post_prompt = None

    question = prompt

    if post_prompt is not None:
        input_text = pre_prompt + question + post_prompt
    else:
        input_text = pre_prompt + question
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    sample_outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        max_length=200,
        top_k=100,
        top_p=0.95,
        temperature=0.6,
        num_return_sequences=2,
    )
    max_answer_characters = 10000
    valid_punctuation = [".", "?", "!", ")", "'", '"', "”"]

    outputs = []
    for i, sample_output in enumerate(sample_outputs):
        answer = tokenizer.decode(sample_output, skip_special_tokens=True)

        if post_prompt is not None:
            answer = answer.split(post_prompt)[-1]
        else:
            answer = answer.split(question)[-1]
        answer = answer[:max_answer_characters]
        answer = answer.split("Question: ")[0].strip()
        answer = answer.split("User: ")[0].strip()
        # Replace duplicate newlines with a single newline
        answer = "\n".join(answer.splitlines())
        # Find the last matching punctuation character
        last_punctuation = max([answer.rfind(p) for p in valid_punctuation])
        # If there is no punctuation, just use the last character
        if last_punctuation == -1:
            last_punctuation = len(answer)
        # Truncate the answer to the last punctuation character
        answer = answer[: last_punctuation + 1]
        # Upper case the first character
        answer = answer[0].upper() + answer[1:]
        # Add the answer to the list of outputs
        outputs.append(answer)

    print("Question: " + question)
    # Grab the first answer without underscores
    got_answer = False
    for answer in outputs:
        if "_" not in answer:
            got_answer = True
            print("Answer: " + answer)
            break
    if not got_answer:
        print("Answer: " + outputs[0])


def get_answer_v2(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    sample_outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        max_length=250,
        top_k=175,
        top_p=0.88,
        temperature=0.7,
        num_return_sequences=3,
    )
    max_answer_characters = 200
    valid_punctuation = [".", "?", "!", ")"]

    outputs = []
    for i, sample_output in enumerate(sample_outputs):
        answer = tokenizer.decode(sample_output, skip_special_tokens=True)
        if post_prompt is not None:
            answer = answer.split(post_prompt)[-1]
        else:
            answer = answer.split(question)[-1]
        answer = answer[:max_answer_characters]
        answer = answer.split("Question: ")[0].strip()
        answer = answer.split("User: ")[0].strip()
        # Replace duplicate newlines with a single newline
        answer = "\n".join(answer.splitlines())
        # Find the last matching punctuation character
        last_punctuation = max([answer.rfind(p) for p in valid_punctuation])
        # If there is no punctuation, just use the last character
        if last_punctuation == -1:
            last_punctuation = len(answer)
        # Truncate the answer to the last punctuation character
        answer = answer[: last_punctuation + 1]
        # Upper case the first character
        answer = answer[0].upper() + answer[1:]
        # Add the answer to the list of outputs
        outputs.append(answer)

    # Grab the first answer without underscores
    for answer in outputs:
        if "_" not in answer:
            return answer

    return outputs[0]


def get_answer_v1(prompt, model, tokenizer):
    input_text = pre_prompt + prompt + post_prompt
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    sample_outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        max_length=350,
        top_k=2000,
        top_p=0.9,
        num_return_sequences=3,
    )
    max_answer_characters = 200
    valid_punctuation = [".", "?", "!", ")"]

    sample_output = sample_outputs[0]
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
    answer = answer[: last_punctuation + 1]
    # Upper case the first character
    answer = answer[0].upper() + answer[1:]

    return answer
