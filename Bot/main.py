from aiogram import Bot, Dispatcher, executor, types
from io import BytesIO
from PIL import Image
import torch
from torchvision import models
import torchvision.transforms as transforms
import numpy as np

import csv

# функция соответствия индексов названиям машин
def get_car_name(index):
    with open('names.csv') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == index:
                return row[0]

# Функция для классификации автомобиля
def classify_car(image):
    # Преобразование фото в тензор
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    img_tensor = transform(image).unsqueeze_(0)

    # Классификация автомобиля
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        class_idx = np.argmax(probabilities.numpy())

    return class_idx

# Загружаем датасет
model = torch.load('resnet50.h5')
model.eval()

# Подключаем бота
with open('token.txt') as tg_api:
    TELEGRAM_API_TOKEN = tg_api.read()
bot = Bot(token=TELEGRAM_API_TOKEN)
dp = Dispatcher(bot)

# handlers
@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    await message.answer('Привет!'+\
                         ' Я телеграмм-бот для классификации автомобилей' +\
                         ' по фото. Просто пришли мне фото автомобиля и я' +\
                         ' скажу тебе, какой это автомобиль.')


@dp.message_handler(content_types=['document'])
async def process_file(message: types.Message):

    if message.document.mime_type.startswith('image'):
        file = BytesIO()
        await bot.download_file_by_id(message.document.file_id, file)
        image = Image.open(file)
        
        class_idx = classify_car(image)
        await message.answer('Это ' + str(get_car_name(class_idx)))

    else:
        await message.answer('Пришли фотографию.')


@dp.message_handler(content_types=['photo'])
async def false_photo(message: types.Message):

    photo = message.photo[-1]

    file = BytesIO()
    await bot.download_file_by_id(photo.file_id, file)
    image = Image.open(file)

    class_idx = classify_car(image)
    await message.answer('Это ' + str(get_car_name(class_idx)))


@dp.message_handler()
async def false_message(message: types.Message):
    await message.answer('пришли фото')



if __name__ == '__main__':
    executor.start_polling(dp)
