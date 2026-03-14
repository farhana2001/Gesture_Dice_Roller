import cv2
import mediapipe as mp
import random
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2)   # Allow 2 hands

cap = cv2.VideoCapture(0)

# ------------------ GAME VARIABLES ------------------
user_guess = None
dice_result = None
message = "Show 1-6 fingers to guess"
game_state = "WAITING"  
previous_fist = False
roll_start_time = 0
final_number = None


# ------------------ MAIN LOOP ------------------
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    finger_count = 0
    current_fist = True   # assume fist until fingers found

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            tip_ids = [4, 8, 12, 16, 20]
            lm_list = []


            for id, lm in enumerate(handLms.landmark):
                lm_list.append([id, lm.x, lm.y])

            if len(lm_list) == 21:
                fingers = []

                # THUMB
                if lm_list[8][1] < lm_list[20][1]:
                    fingers.append(1 if lm_list[4][1] < lm_list[3][1] else 0)
                else:
                    fingers.append(1 if lm_list[4][1] > lm_list[3][1] else 0)

                # OTHER FINGERS
                for i in range(1, 5):
                    fingers.append(1 if lm_list[tip_ids[i]][2] < lm_list[tip_ids[i]-2][2] else 0)

                finger_count += fingers.count(1)

                if fingers.count(1) > 0:
                    current_fist = False

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # ------------------ GAME LOGIC ------------------

    # -------- 1️⃣ WAITING: Ask for guess (1-6) --------
    if game_state == "WAITING":
        if 1 <= finger_count <= 6:
            user_guess = finger_count
            message = "Make a fist to roll"
            game_state = "READY"

    # -------- 2️⃣ READY: Wait for fist --------
    elif game_state == "READY":
        if current_fist :
            roll_start_time = time.time()
            final_number = random.randint(1, 6)
            game_state = "ROLLING"

    # -------- 3️⃣ ROLLING: Dice animation --------
    elif game_state == "ROLLING":
        dice_result = random.randint(1, 6)

        if time.time() - roll_start_time > 1:
            dice_result = final_number

            if dice_result == user_guess:
                message = "YOU WIN! Open hand to restart"
            else:
                message = "TRY AGAIN! Open hand to restart"

            game_state = "RESULT"

    # # -------- 4️⃣ RESULT: Restart --------
    elif game_state == "RESULT":
        if finger_count > 0 and not current_fist:
            user_guess = None
            dice_result = None
            final_number = None
            message = "Show 1-6 fingers to guess"
            game_state = "WAITING"

    previous_fist = current_fist

    # ------------------ DISPLAY ------------------
    cv2.putText(img, f"Your Guess: {user_guess}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(img, f"Dice: {dice_result}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(img, message, (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Gesture Dice Roller (1-6)", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
