from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загрузка модели и токенизатора
model_name = "sberbank-ai/rugpt3small_based_on_gpt2"  
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Функция для генерации ответа
def generate_response(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Основной цикл общения
print("Привет! Я твой локальный ИИ. Напиши что-нибудь, и я отвечу.")
while True:
    user_input = input("Ты: ")
    if user_input.lower() in ["выход", "exit", "quit"]:
        print("ИИ: Пока! Было приятно пообщаться.")
        break
    response = generate_response(user_input)
    print(f"ИИ: {response}")
