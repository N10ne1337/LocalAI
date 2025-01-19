from transformers import GPT2LMHeadModel, GPT2Tokenizer

   # Укажи путь для сохранения модели
   model_path = "./LocalAI"

   # Загрузка модели и токенизатора
   model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
   tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")

   # Сохранение модели и токенизатора
   model.save_pretrained(model_path)
   tokenizer.save_pretrained(model_path)

   print(f"Модель и токенизатор сохранены в {model_path}")
