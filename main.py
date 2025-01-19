from transformers import pipeline

   # Загрузка модели для генерации текста
   generator = pipeline("text-generation", model="gpt2")

   # Основной цикл общения
   print("Привет! Я твой локальный ИИ. Напиши что-нибудь, и я отвечу.")
   while True:
       user_input = input("Ты: ")
       if user_input.lower() in ["выход", "exit", "quit"]:
           print("ИИ: Пока! Было приятно пообщаться.")
           break
       response = generator(user_input, max_length=50, num_return_sequences=1)
       print(f"ИИ: {response[0]['generated_text']}")
