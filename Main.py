import cv2
import face_recognition
import os
import asyncio
from aiogram import Bot

# Настройка Telegram-бота
TELEGRAM_TOKEN = '7629553408:AAFuzylq9iazvCVsqwX4RDQDbmzIeGnO8lU'  # Вставьте токен вашего бота
CHAT_ID = '1175927389'  # Вставьте ваш ID чата

bot = Bot(token=TELEGRAM_TOKEN)

# Путь к папке с изображениями
path_to_images = '/Users/andrejskripnikov/Desktop/MTUCI Face/registered_face'

# Списки для хранения кодировок и имён известных лиц
known_face_encodings = []
known_face_names = []

# Загрузка изображений и имен из папки
for image_name in os.listdir(path_to_images):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    img_path = os.path.join(path_to_images, image_name)
    img = face_recognition.load_image_file(img_path)
    
    encodings = face_recognition.face_encodings(img)
    if encodings:
        face_encoding = encodings[0]
        known_face_encodings.append(face_encoding)
        name = os.path.splitext(image_name)[0]
        known_face_names.append(name)
    else:
        print(f"Лицо не обнаружено на изображении {image_name}. Оно будет пропущено.")

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Не удалось открыть камеру.")
    exit()

process_every_n_frames = 2
frame_count = 0

previous_face_locations = []
previous_face_names = []

# Словарь для отслеживания состояния обнаружения лиц
detected_faces = {}

# Функция для отправки сообщения в Telegram
async def send_telegram_message(name):
    await bot.send_message(chat_id=CHAT_ID, text=f"Обнаружено лицо: {name}")

# Функция для обработки сообщений
async def process_message(name):
    await send_telegram_message(name)

async def main_loop():
    global frame_count, previous_face_locations, previous_face_names, detected_faces

    while True:
        ret, frame = video_capture.read()
        
        if not ret:
            print("Не удалось получить кадр с камеры.")
            break
        
        frame_count += 1

        if frame_count % process_every_n_frames == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown" #Начально
                
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding) #Расстояние
                best_match_index = face_distances.argmin() if face_distances.size > 0 else None
                
                if best_match_index is not None and matches[best_match_index]:
                    name = known_face_names[best_match_index]

                    # Проверка, было ли это лицо уже обнаружено
                    if name not in detected_faces or not detected_faces[name]:
                        # Отправка сообщения в Telegram при обнаружении лица
                        await process_message(name)  # Запуск асинхронной функции
                        detected_faces[name] = True  # Устанавливаем флаг, что лицо обнаружено
                    
                face_names.append(name)

            # Сброс флага для лиц, которых больше нет в кадре
            for name in detected_faces.keys():
                if name not in face_names:
                    detected_faces[name] = False
            
            previous_face_locations = face_locations
            previous_face_names = face_names
        else:
            face_locations = previous_face_locations
            face_names = previous_face_names
        
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 5)
            
            font = cv2.FONT_HERSHEY_DUPLEX
            text_size = cv2.getTextSize(name, font, 1.0, 2)[0]
            text_x = left + (right - left) // 2 - text_size[0] // 2
            text_y = top - 10
            
            cv2.putText(frame, name, (text_x, text_y), font, 1.0, (0, 255, 0), 2)

        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Запуск основного цикла с использованием asyncio
asyncio.run(main_loop())

video_capture.release()
cv2.destroyAllWindows()