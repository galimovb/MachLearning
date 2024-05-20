import cv2
import mediapipe as mp

# Инициализация MediaPipe для распознавания лиц и рук
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
face_detector = mp_face.FaceDetection()
hand_detector = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.7, min_tracking_confidence=0.5
)

# Функция, возвращающая имя в зависимости от числа поднятых пальцев
def identify_name_by_fingers(fingers_count):
    names = {1: "Bulat", 2: "Galimov"}
    return names.get(fingers_count, "Show anymore")

# Захват видео с камеры
camera = cv2.VideoCapture(0)

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        continue

    # Преобразование изображения в формат RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Детекция лиц
    faces = face_detector.process(rgb_frame).detections
    if faces:
        for face in faces:
            box = face.location_data.relative_bounding_box
            height, width, _ = frame.shape
            start_point = (int(box.xmin * width), int(box.ymin * height))
            end_point = (int((box.xmin + box.width) * width), int((box.ymin + box.height) * height))
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

    # Детекция рук и подсчет поднятых пальцев
    hands = hand_detector.process(rgb_frame).multi_hand_landmarks
    fingers_up = 0
    if hands:
        for hand in hands:
            tip_ids = [4, 8, 12, 16, 20]#индексы кончиков пальцев. большой->мизинец
            up_status = [
                hand.landmark[tip_ids[0]].y < hand.landmark[tip_ids[0] - 1].y, #кончик каждого пальца сравнивается с его ключевой точкой которая находится ниже к ладони,
                hand.landmark[tip_ids[1]].y < hand.landmark[tip_ids[1] - 2].y,# если кончик выше в изображении, то палец поднят
                hand.landmark[tip_ids[2]].y < hand.landmark[tip_ids[2] - 2].y,
                hand.landmark[tip_ids[3]].y < hand.landmark[tip_ids[3] - 2].y,
                hand.landmark[tip_ids[4]].y < hand.landmark[tip_ids[4] - 2].y
            ]
            fingers_up = up_status.count(True) #кол-во поднятых

    # Вывод имени по количеству пальцев
    name = identify_name_by_fingers(fingers_up)
    if faces:
        cv2.putText(frame, name, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Отображение кадра
    cv2.imshow('Camera Output', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
camera.release()
cv2.destroyAllWindows()
