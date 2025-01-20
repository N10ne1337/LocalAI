from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загрузка модели
model_name = "ai-forever/rugpt3small_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Введение текста
input_text = "Привет! Как дела сегодня?"
inputs = tokenizer(input_text, return_tensors='pt')

# Получение результата
output = model.generate(**inputs)
result = tokenizer.decode(output[0], skip_special_tokens=True)

print(result)
