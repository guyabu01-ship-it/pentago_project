import matplotlib.pyplot as plt
import json
import os
import numpy as np



class Data_cleaning():
    def __init__(self):
        self.X = []
        self.Y = []
        self.dict = {}
        # המילונים יאותחלו בטעינה
        self.dict1 = {}
        self.dict2 = {}
        self.dict3 = {}
        self.dict4 = {}
        self.dict5 = {}
        self.dict6 = {}
        self.dict7 = {}

    def process_pentago_board(self, board_str):
        decimal_value = int(board_str)

        flat_board = []
        temp_n = decimal_value
        for _ in range(36):
            digit = temp_n % 3

            if digit == 2:
                digit = -1

            flat_board.append(digit)
            temp_n //= 3

        flat_board.reverse()

        matrix_board = [flat_board[i:i + 6] for i in range(0, 36, 6)]
        return matrix_board

    def reload_dict_from_json(self,
                              filename1='dict1_pentago.json', filename2='dict2_pentago.json',
                              filename3='dict3_pentago.json', filename4='dict4_pentago.json',
                              filename5='dict5_pentago.json', filename6='dict6_pentago.json',
                              filename7='dict7_pentago.json'):
        print(" --- loading date --- ")

        filenames = [filename1, filename2, filename3, filename4, filename5, filename6, filename7]
        loaded_dicts = []

        for fname in filenames:
            if not os.path.exists(fname):
                print(f"Warning: File {fname} not found. Skipping...")
                loaded_dicts.append({})
                continue

            try:
                with open(fname, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                    d = {str(k): tuple(v) for k, v in raw.items()}
                    loaded_dicts.append(d)
            except Exception as e:
                print(f"Error parsing {fname}: {e}")
                loaded_dicts.append({})

        self.dict1, self.dict2, self.dict3, self.dict4, self.dict5, self.dict6, self.dict7 = loaded_dicts
        print("\n --- data is loaded ---")

    def merge_dicts(self):
        # מיזוג כל המילונים לאחד
        self.dict = self.dict1 | self.dict2 | self.dict3 | self.dict4 | self.dict5 | self.dict6 | self.dict7
        print(f"Total unique boards in merged dictionary: {len(self.dict)}")

    def dict_to_list(self):
        self.X = []
        self.Y = []

        print("\n--- Processing Data ---")
        for board_key, data in self.dict.items():
            # total_score הוא הציון המצטבר, count הוא מספר המופעים
            total_score, count = data

            if count > 0:
                avg_score = total_score / count

               
                if count >= 3 or avg_score >= 0 or avg_score <= -0.75:
                    try:
                        matrix = self.process_pentago_board(str(board_key))
                        self.X.append(matrix)
                        self.Y.append(avg_score)
                    except Exception as e:
                        # מדפיס שגיאה רק אם יש בעיה בעיבוד הלוח עצמו
                        print(f"Error processing board {board_key}: {e}")
                        continue

        print(f"--- Success! Items in Y: {len(self.Y)} ---")

    def y_distribution(self):
        if not self.Y:
            print("No data in Y to plot! Check your filtering conditions (count and score).")
            return

        plt.figure(figsize=(10, 6))
        plt.hist(self.Y, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f'Distribution of Board Scores (Samples: {len(self.Y)})')
        plt.xlabel('Average Score')
        plt.ylabel('Number of Boards')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.show()



    def safety_check(self):
        print("\n--- Safety Check ---")
        if len(self.X) != len(self.Y):
            print("Board count and grades count are mismatch. ")
            return False
        for grade in self.Y:
            if grade == None:
                print("there is a missing score.")
                return False

        for board in self.X:
            if board == 0:
                print("there is a missing board.")
                return False

        print("Board count and grades count are correct, data is ready to be saved.")
        return True

    def save_as_numpy(self):
        print("\n--- Saving Data as NumPy ---")
        # המרה למערכי NumPy
        x_array = np.array(self.X, dtype='int8')
        y_array = np.array(self.Y, dtype='float32')

        # שמירה כקובץ אחד דחוס
        np.savez_compressed('processed_data_np3.npz', X=x_array, Y=y_array)
        print(f"--- Success! Saved to processed_data_np3.npz ---")

    def run(self):
        self.reload_dict_from_json()
        self.merge_dicts()
        self.dict_to_list()
        self.safety_check()
        self.y_distribution()
        # הוספת הקריאה לשמירה החדשה
        self.save_as_numpy()



if __name__ == '__main__':
    cleaner = Data_cleaning()
    cleaner.run()
