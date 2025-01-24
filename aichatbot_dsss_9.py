#!pip install python-telegram-bot --upgrade
#!pip install huggingface_hub

import nest_asyncio
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, pipeline
import torch

# Start more than one Event-Loop -> google colab problem
nest_asyncio.apply()

# Proof if a GPU is available
colab_gpu = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {colab_gpu}")

# Tokenizer and model (GPTNeo)
try:
    model_path = "EleutherAI/gpt-neo-1.3B"

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPTNeoForCausalLM.from_pretrained(model_path).to(colab_gpu)

    # Pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if colab_gpu == "cuda" else -1)


    print("GPTNeo loaded successfully!")
except Exception as e:
    print(f"Error: {e}")

# start: how bot is replying
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I am your fancy AI chat bot. How can I help you?")


async def response_with_gptneo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_user = update.message.text.strip()  # user message

    try:
        response = pipe(text_user, max_length=150, num_return_sequences=1, truncation=True,
                        temperature=0.7, top_p=0.9,
                        no_repeat_ngram_size=2, do_sample=True)[0]['generated_text'].split(text_user, 1)[-1].strip()

    except Exception as e:
        print(f"Error during generation: {e}")

    #answer to user
    await update.message.reply_text(response)

def main():
    print("Bot is starting!")
    API_TOKEN = "7793354822:AAHJJSxlKclkn-KiC63vqqDbjp4LuZItOWY"
    application = Application.builder().token(API_TOKEN).build()

    # Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, response_with_gptneo))

    print("Bot is running!")

    # Starting bot
    application.run_polling()

if __name__ == "__main__":
    main()
