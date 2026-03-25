import cv2
import mediapipe as mp
import math

#MediaPipe 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# img 
cat_img = cv2.imread("img/cat.png", cv2.IMREAD_UNCHANGED)
dog_img = cv2.imread("img/dog.png", cv2.IMREAD_UNCHANGED)

# overlay function 
def overlay_image(bg, overlay, x, y):
    h, w = overlay.shape[:2]

    # set up boarding 
    if x < 0 or y < 0 or x + w > bg.shape[1] or y + h > bg.shape[0]:
        return

    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y+h, x:x+w, c] = (
                alpha * overlay[:, :, c] +
                (1 - alpha) * bg[y:y+h, x:x+w, c]
            )
    else:
        bg[y:y+h, x:x+w] = overlay

# background + animation
def draw_text_box(img, text, cx, cy, offset=180, color=(255,255,255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2

    # animation 
    t = cv2.getTickCount() / cv2.getTickFrequency()
    bounce = int(10 * math.sin(t * 3))

    y = cy - offset + bounce

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    x = cx - tw // 2

    #set up boarding 
    if y < 30:
        y = 30

    # background
    cv2.rectangle(img, (x-10, y-th-10), (x+tw+10, y+10), (0,0,0), -1)

    # shadow
    cv2.putText(img, text, (x+2, y+2), font, scale, (0,0,0), thickness+2)

    # textsize
    cv2.putText(img, text, (x, y), font, scale, color, thickness)

# camera control
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            label = hand_info.classification[0].label

            # head control
            cx = int(sum([lm.x for lm in hand_landmarks.landmark]) / 21 * w)
            cy = int(sum([lm.y for lm in hand_landmarks.landmark]) / 21 * h)

            # letf hand
            if label == "Left":
                draw_text_box(image, "CAT DETECTED!", cx, cy, offset=220)

                if cat_img is not None:
                    cat = cv2.resize(cat_img, (240, 240))  #  size
                    overlay_image(image, cat, cx - 120, cy - 120)

            # right head
            if label == "Right":
                draw_text_box(image, "DOG DETECTED!", cx, cy, offset=220, color=(0,255,255))

                if dog_img is not None:
                    dog = cv2.resize(dog_img, (240, 240))  #  size
                    overlay_image(image, dog, cx - 120, cy - 120)

            # head
            mp_draw.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("AR Hand Animal ", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
