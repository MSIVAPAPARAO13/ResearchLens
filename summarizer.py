from transformers import BartForConditionalGeneration, BartTokenizer

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)


def summarize_text(text, max_length, min_length):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=6,
        length_penalty=2.0,
        max_length=max_length,
        min_length=min_length,
        early_stopping=True,
        no_repeat_ngram_size=3
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
