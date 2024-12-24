import cv2
import numpy as np
import time
import os
import Cards
import VideoStream
from collections import defaultdict

def convert_rank(rank):
    """Convert card rank to simplified format"""
    rank_map = {
        'Ace': 'a',
        'Two': '2',
        'Three': '3',
        'Four': '4',
        'Five': '5',
        'Six': '6',
        'Seven': '7',
        'Eight': '8',
        'Nine': '9',
        'Ten': '10',
        'Jack': 'j',
        'Queen': 'q',
        'King': 'k'
    }
    return rank_map.get(rank, rank)

def get_best_cards(cards_confidence):
    """
    Get the ranks with highest confidence scores
    
    Parameters:
    cards_confidence (dict): Dictionary mapping ranks to their confidence scores
    
    Returns:
    list: List of ranks with highest confidence scores
    """
    best_cards = []
    for rank, confidences in cards_confidence.items():
        if confidences and rank != "Unknown":  # If we have predictions and rank is not Unknown
            # Get the highest confidence for this rank
            best_confidence = max(confidences)
            best_cards.append((rank, best_confidence))
    
    return [convert_rank(rank) for rank, conf in best_cards]

def get_dealer_ranks():
    """
    Return the ranks of the dealer's cards in simplified format.

    Returns:
    list: List of dealer's card ranks.
    """
    return get_best_cards(dealer_confidence)

def get_player_ranks():
    """
    Return the ranks of the player's cards in simplified format.

    Returns:
    list: List of player's card ranks.
    """
    return get_best_cards(player_confidence)

### ---- INITIALIZATION ---- ###
IM_WIDTH = 1280
IM_HEIGHT = 720 
FRAME_RATE = 10

frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera object and video feed
videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE,2,0).start()
time.sleep(1)

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks( path + '/Card_Imgs/')
train_suits = Cards.load_suits( path + '/Card_Imgs/')

### ---- MAIN LOOP ---- ###
cam_quit = 0
is_scanning = False
scan_start_time = 0
scan_duration = 10  # seconds

# Dictionaries to store confidence scores for upper and lower halves
dealer_confidence = defaultdict(list)
player_confidence = defaultdict(list)

print("Press 's' to start a 10-second scan or 'q' to quit")

while not cam_quit:
    # Grab frame from video stream
    image = videostream.read()
    t1 = cv2.getTickCount()

    if is_scanning:
        current_time = time.time()
        elapsed_time = current_time - scan_start_time
        
        if elapsed_time >= scan_duration:
            # Scanning period ended
            is_scanning = False

            # Get and print the best predictions for each half
            dealer_cards = get_dealer_ranks()
            player_cards = get_player_ranks()

            print("\nDealer's cards:")
            print(' '.join(sorted(dealer_cards)))

            print("\nPlayer's cards:")
            print(' '.join(sorted(player_cards)))

            # Reset confidence tracking for next scan
            dealer_confidence.clear()
            player_confidence.clear()
            print("\nPress 's' to start another scan or 'q' to quit")
        else:
            # Still scanning - process frame
            pre_proc = Cards.preprocess_image(image)
            cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

            if len(cnts_sort) != 0:
                cards = []
                k = 0

                for i in range(len(cnts_sort)):
                    if (cnt_is_card[i] == 1):
                        card = Cards.preprocess_card(cnts_sort[i], image)
                        card.best_rank_match, card.best_suit_match, card.rank_diff, card.suit_diff = Cards.match_card(card, train_ranks, train_suits)

                        # Calculate confidence score
                        confidence = 1.0 / (card.rank_diff + 1)  # Add 1 to avoid division by zero

                        # Determine if card is in upper or lower half
                        centroid_y = np.mean(card.contour[:, :, 1])
                        if centroid_y < IM_HEIGHT / 2:
                            dealer_confidence[card.best_rank_match].append(confidence)
                        else:
                            player_confidence[card.best_rank_match].append(confidence)

                        image = Cards.draw_results(image, card)
                        cards.append(card)
                        k += 1

                if len(cards) != 0:
                    temp_cnts = [card.contour for card in cards]
                    cv2.drawContours(image, temp_cnts, -1, (255, 0, 0), 2)

            # Display remaining time
            remaining_time = int(scan_duration - elapsed_time)
            cv2.putText(image, f"Scanning... {remaining_time}s", (10, 26), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    else:
        # Not scanning - display waiting message
        cv2.putText(image, "Press 's' to start scanning", (10, 26), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    # Display the image
    cv2.imshow("Card Detector", image)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1
    
    # Check for keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s") and not is_scanning:
        is_scanning = True
        scan_start_time = time.time()
        print("\nStarting 10-second scan...")
    elif key == ord("q"):
        cam_quit = 1

# Clean up
cv2.destroyAllWindows()
videostream.stop()