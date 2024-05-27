import random
import time
import numpy as np
import os

class QLearningAgent:
    def __init__(self, min_value, max_value, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99, q_table_file="q_table.npy"):
        self.min_value = min_value
        self.max_value = max_value
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table_file = q_table_file
        self.q_table = np.zeros((250, 250))  # Q-table size increased to 250 x 250
        self.load_q_table()

    def choose_action(self, low, high):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(low, high)
        else:
            return (low + high) // 2  # Use binary search strategy

    def update_q_table(self, state, action, reward, next_state):
        state_index = state - self.min_value
        action_index = action - self.min_value
        next_state_index = next_state - self.min_value

        if next_state_index >= len(self.q_table):
            next_state_index = len(self.q_table) - 1

        best_next_action = np.argmax(self.q_table[next_state_index])
        td_target = reward + self.discount_factor * self.q_table[next_state_index][best_next_action]
        td_error = td_target - self.q_table[state_index][action_index]
        self.q_table[state_index][action_index] += self.learning_rate * td_error

    def decay_exploration(self):
        self.exploration_rate *= self.exploration_decay

    def save_q_table(self):
        np.save(self.q_table_file, self.q_table)
        print(f"Q-table saved to {self.q_table_file}")

    def load_q_table(self):
        if os.path.exists(self.q_table_file):
            self.q_table = np.load(self.q_table_file)
            print(f"Q-table loaded from {self.q_table_file}")

def read_records():
    try:
        with open("records.txt", "r", encoding="utf-8") as f:
            content = f.read()
            records = {"easy": {}, "medium": {}, "hard": {}}
            for line in content.splitlines():
                parts = line.split(": ")
                if len(parts) == 3:
                    difficulty, name, highest_attempts = parts
                    records[difficulty][name] = int(highest_attempts)
            return records
    except FileNotFoundError:
        return {"easy": {}, "medium": {}, "hard": {}}

def write_record(name, attempts, difficulty):
    records = read_records()
    records[difficulty][name] = attempts

    # Sort records by number of attempts in ascending order
    content = []
    for difficulty, recs in records.items():
        sorted_records = sorted(recs.items(), key=lambda x: x[1])
        for name, highest_attempts in sorted_records:
            content.append(f"{difficulty}: {name}: {highest_attempts}")
    
    with open("records.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(content))

def choose_difficulty():
    print("Choose difficulty:")
    print("1. Easy (1-50)")
    print("2. Medium (1-100)")
    print("3. Hard (1-200)")
    difficulty = input("Enter your choice (1, 2, or 3): ")
    if difficulty == '1':
        return 'easy', 50
    elif difficulty == '2':
        return 'medium', 100
    elif difficulty == '3':
        return 'hard', 200
    else:
        print("Invalid choice, defaulting to medium difficulty.")
        return 'medium', 100

def choose_mode():
    print("Choose game mode:")
    print("1. Single player")
    print("2. Versus AI")
    mode = input("Enter your choice (1 or 2): ")
    if mode == '1':
        return False
    elif mode == '2':
        return True
    else:
        print("Invalid choice, defaulting to single player.")
        return False

def main():
    # Choose difficulty
    difficulty, range_value = choose_difficulty()

    # Choose game mode
    versus_ai = choose_mode()

    player_name = input("Enter player name: ")

    # Check if player uses cheat code
    if player_name.lower() == "terminal":
        print("**Congratulations! You've used the cheat code and won instantly!**")
        return

    # Choose a random number within the selected difficulty range
    secret_number = random.randint(1, range_value)

    # Initialize AI
    agent = QLearningAgent(1, range_value)

    # Start the game
    attempts = 0
    ai_attempts = 0
    low, high = 1, range_value
    while True:
        attempts += 1

        # Enter player's guess
        guess = input(f"Attempt {attempts}: Enter your guess (1 - {range_value}): ")

        # Easter Egg: if "python" is entered, display a surprise message
        if guess.lower() == "python":
            for char in "PYTHON":
                print(char, end=" ", flush=True)
                time.sleep(0.5)
            print("\nYou've discovered an Easter Egg! Try guessing again!")
            continue

        guess = int(guess)

        # Check player's guess
        if guess == secret_number:
            # Print congratulatory message character by character
            for char in f"**Congratulations {player_name}! You've guessed the secret number in {attempts} attempts!** ":
                print(char, end="", flush=True)
                time.sleep(0.1)

            # Write record
            write_record(player_name, attempts, difficulty)
            break
        elif guess < secret_number:
            print("Your guess is lower than the secret number.")
            low = guess + 1
        else:
            print("Your guess is higher than the secret number.")
            high = guess - 1
        
        # If versus AI, the AI will also guess
        if versus_ai:
            ai_attempts += 1
            action = agent.choose_action(low, high)
            print(f"AI's attempt {ai_attempts}: AI guesses {action}")

            if action == secret_number:
                print(f"**AI guessed the secret number in {ai_attempts} attempts! You lost the game.**")
                return
            elif action < secret_number:
                print("AI's guess is lower than the secret number.")
                low = action + 1
            else:
                print("AI's guess is higher than the secret number.")
                high = action - 1

            reward = -1
            if action == secret_number:
                reward = 100

            # Update Q-table with correct state and action
            state = action
            next_state = state
            agent.update_q_table(state, action, reward, next_state)
            agent.decay_exploration()

        # Provide a hint after 5 attempts for higher difficulty
        if attempts == 5 and difficulty == 'hard':
            if secret_number % 2 == 0:
                print("Hint: The secret number is even.")
            else:
                print("Hint: The secret number is odd.")
            if secret_number <= range_value // 2:
                print(f"Hint: The secret number is less than or equal to {range_value // 2}.")
            else:
                print(f"Hint: The secret number is greater than {range_value // 2}.")

    # Print records after the game ends
    records = read_records()
    records_sorted = sorted(records[difficulty].items(), key=lambda x: x[1])
    print("\n--- Records ---")
    for name, highest_attempts in records_sorted:
        print(f"{name}: {highest_attempts}")

if __name__ == "__main__":
    main()
