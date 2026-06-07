import numpy as np
import random
import json
import os
from smart_agent import SmartAgent

class Pentago_game:
    def __init__(self, dict1, dict2, dict3, dict4, dict5, dict6, dict7, smart_agent):
        """
        מגדירה משחק חדש ומאפס אותו
        """
        self.board = np.zeros((6, 6), dtype=int)
        self.agent_sign = 1
        self.player_sign = -1
        self.tie_sign = 0
        self.boards_history = []
        self.grades = []
        self.gamma = 0.9
        self.winner = self.tie_sign
        self.dict1 = dict1
        self.dict2 = dict2
        self.dict3 = dict3
        self.dict4 = dict4
        self.dict5 = dict5
        self.dict6 = dict6
        self.dict7 = dict7
        self.smart_agent = smart_agent


    def player_win(self, i,j,p):
        b = self.board

        # אופקי
        if np.sum(b[i, 0:5] == p) == 5 or np.sum(b[i, 1:6] == p) == 5:
            return True

        # אנכי
        if np.sum(b[0:5, j] == p) == 5 or np.sum(b[1:6, j] == p) == 5:
            return True

        # אלכסון ראשי
        if np.sum(b.diagonal(j - i) == p) >= 5:
            return True

        # אלכסון משני
        if np.sum(np.fliplr(b).diagonal((5 - j) - i) == p) >= 5:
            return True

        return False

    def check_win(self, i,j):
        if self.player_win(i,j,self.agent_sign):
            return self.agent_sign
        if self.player_win(i,j,self.player_sign):
            return self.player_sign
        return self.tie_sign

    def possible_boards(self):
        empty_places = self.empty_places()
        boards = []
        for (i, j) in empty_places:
            for quarter in range(1, 5):
                for side in [1, 2]:
                    back_up_board = self.board.copy()
                    self.board[i, j] = self.agent_sign
                    self.turn_quarter(quarter, side)
                    boards.append(self.board_to_base3(self.board))
                    self.board = back_up_board
        return boards

    def count_possible_boards_in_dictionary(self,dict):
        boards = self.possible_boards()
        count = 0
        for board in boards:
            if board in dict:
                count += 1
        return count/len(boards)

    def exploit(self, index, default_score=0.0):
        """
        gets index and finds the right dictionary to use and finds the best option to play from the possible moves,
        otherwise it plays by block() or by win_turn() or randomly.
        :param index: represents the number of the turn from 1-36
        :param default_score: reference value for the scores
        :return: coordination of the best option to play from the possible moves
        """
        best_score = -float('inf')
        best_moves = []
        score_dict = self.choose_right_dictionary(index)
        if score_dict is None:
            score_dict = self.dict7

        empty_places = self.empty_places()
        precentage_possible_by_dict = self.count_possible_boards_in_dictionary(score_dict)
        count = 0

        if precentage_possible_by_dict >= 0.65:
            for (r, c) in empty_places:
                for quarter in range(1, 5):
                    for side in (1, 2):
                        board_backup = self.board.copy()
                        self.board[r, c] = self.agent_sign
                        self.turn_quarter(quarter, side)
                        key = self.board_to_base3(self.board)
                        val = score_dict.get(key)
                        score = val[0] if val is not None else default_score
                        if score > best_score:
                            best_score = score
                            best_moves = [(r, c, quarter, side)]
                        elif score == best_score:
                            best_moves.append((r, c, quarter, side))
                        self.board = board_backup

            # אם מצאנו מהלכים במילון
            if best_moves:
                r, c, quarter, side = random.choice(best_moves)
                self.board[r, c] = self.agent_sign
                self.turn_quarter(quarter, side)
                count = 1
                return r, c, count

        # אם לא עמדנו בתנאי ה-65% או שלא נמצאו מהלכים במילון
        move = self.win_turn(self.agent_sign)
        if move:
            i, j = move
            return i, j, count

        move = self.block(self.agent_sign, self.player_sign)
        if move:
            i, j = move
            return i, j, count

        # מוצאים מהלך רנדומלי (הפונקציה random_turn כבר מעדכנת את הלוח ומחזירה i, j)
        i, j = self.random_turn(self.agent_sign)
        return i, j, count

    def turn_quarter(self, number, side):
        """
        turn the side of the quarter
        :param number: number of quarter
        :param side: side of turning, 1 is right 2 is left
        :return: void
        """
        if number == 1:
            r, c = slice(0, 3), slice(0, 3)
        elif number == 2:
            r, c = slice(0, 3), slice(3, 6)
        elif number == 3:
            r, c = slice(3, 6), slice(0, 3)
        elif number == 4:
            r, c = slice(3, 6), slice(3, 6)
        else:
            return

        sub = self.board[r, c].copy()

        k = -1 if side == 1 else 1
        self.board[r, c] = np.rot90(sub, k=k)

    def empty_places(self):
        """
        gets the empty places
        :return: list of tuples with coordinates of empty places
        """
        r,c = np.where(self.board == 0)
        return list(zip(r,c))

    def random_turn(self, sign):
        i, j = random.choice(self.empty_places())
        quarter = random.randint(1, 4)
        side = random.choice([1, 2])
        self.board[i, j] = sign
        self.turn_quarter(quarter, side)
        return i, j

    def agent_turn(self):
        return self.random_turn(self.agent_sign)

    def win_turn(self,sign):
        """
        identifying win in the next round
        :param sign: the player's sign
        :return: coordinates if winning is possible, else False
        """
        result = self.rotate_to_win(sign)
        if result is None:
            return False
        i, j, quarter, side = result
        new_side = 2 if side == 1 else 1
        r, c = self.rotate_point(i, j, quarter, new_side)
        self.board[r, c] = sign
        self.turn_quarter(quarter, side)
        return r,c

    def player_turn(self):
        return self.random_turn(self.player_sign)

    def init_board(self):
        """
        valuing each place in the board with zero
        :return:
        """
        self.board = np.zeros((6, 6), dtype=int)
        self.boards_history = []
        self.eq_boards = []
        self.grades = []
        self.winner = self.tie_sign


    def board_to_base3(self, board):
        """
            Converts a 6x6 Pentago board into a single integer using base-3 encoding.
            Cell mapping:
                0  -> 0
                1  -> 1
                -1 -> 2
            The board is read row by row (row-major order).
        """
        key = 0
        power = 1
        for i in range(6):
            for j in range(6):
                cell = board[i, j]
                if cell == 0:
                    digit = 0
                elif cell == 1:
                    digit = 1
                else:  # cell == -1
                    digit = 2
                key += digit * power
                power *= 3
        return key

    def is_tie (self):
        """
        checks if the game board is tied
        :return:
            boolean True or False whether the game board is tied
        """
        return not np.any(self.board == 0)

    def apply_full_move(self, i, j, quarter, side, player):
        self.board[i][j] = player
        self.turn_quarter(quarter, side)
        return i, j

    def play_one_game(self):
        """
        plays one game of pentago
        :return:
        returns 1 if agent won, -1 if player won or 0 if the game board is a tied
        """
        self.init_board()
        turn = random.choice([self.agent_sign, self.player_sign])
        i = j = 0
        index = 0
        count = 0

        smart_agent = self.smart_agent
        smart_agent.board = self.board

        while True:
            index += 1
            if self.is_tie():
                return self.tie_sign, count
            alfa = random.random()
            if turn == self.agent_sign:
                smart_agent.board = self.board
                if alfa < 0.5:
                    i, j, quarter, side = smart_agent.choose_move_with_cnn()
                    i, j = self.apply_full_move(
                        i, j, quarter, side, self.agent_sign
                    )
                    smart_agent.board = self.board
                else:
                    i, j, move_count = self.exploit(index)
                    if move_count > 0:
                        count = 1
                turn = self.player_sign
                self.record_board()
            else:

                i, j = self.player_turn()
                # רק כאן צריך לעדכן לוח!
                self.board[i][j] = self.player_sign
                turn = self.agent_sign
                self.record_board()

            winner = self.check_win(i, j)

            if winner != 0:
                self.winner = winner
                return winner, count

    def record_board(self):
        self.boards_history.append(self.board_to_base3(self.board))

    def choose_right_dictionary(self, i):
        if i < 3:
            return self.dict1
        elif i < 6:
            return self.dict2
        elif i < 9:
            return self.dict3
        elif i < 12:
            return self.dict4
        elif i < 25:
            return self.dict5
        elif i < 31:
            return self.dict6
        else:
            return self.dict7

    def will_win(self, p):
        """
        checks if player p has a chance to win according to the game board
        :param p: the player's symbol (agent or player)
        :return: coordinates of winning condition
        """
        for i, j in self.empty_places():
            self.board[i, j] = p
            if self.player_win(i, j, p):
                self.board[i, j] = 0
                return (i, j)
            self.board[i, j] = 0
        return None

    def rotate_to_win (self,p):
        """
        finds the quarter player p should move in order to win in the next turn
        :param p: the player's symbol (agent or player)
        :return: coordinates of winning condition and quarter number and side to rotate
        """
        for quarter in range(1, 5):
            for side in (1, 2):
                self.turn_quarter(quarter, side)
                move = self.will_win(p)
                opposite = 2 if side == 1 else 1
                self.turn_quarter(quarter, opposite)
                if move is not None:
                    i, j = move
                    return i, j, quarter, side
        return None

    def block(self,p1,p2):
        """
        blocks the p2 from winning in the next turn
        :return: coordinates of winning condition
        """
        result = self.rotate_to_win(p2)
        if result is None:
            return False
        i, j, quarter, side = result
        new_side = 2 if side == 1 else 1
        r, c = self.rotate_point(i, j, quarter, new_side)
        self.board[r, c] = p1
        self.turn_quarter(quarter, side)
        return r,c

    def rotate_point(self, i, j, quarter, direction):
        """
        i, j – קואורדינטות המקור
        quarter – 1 עד 4
        direction – 1 = ימינה, 2 = שמאלה
        """
        # קביעת גבולות הרבע
        if quarter == 1:
            base_row, base_col = 0, 0
        elif quarter == 2:
            base_row, base_col = 0, 3
        elif quarter == 3:
            base_row, base_col = 3, 0
        elif quarter == 4:
            base_row, base_col = 3, 3
        else:
            return None  # רבע לא חוקי

        # אם הנקודה לא בתוך הרבע — אין שינוי
        if not (base_row <= i <= base_row + 2 and base_col <= j <= base_col + 2):
            return (i, j)

        # קואורדינטות יחסיות בתוך הרבע
        ri = i - base_row
        rj = j - base_col

        # חישוב סיבוב
        if direction == 1:  # ימינה
            new_ri = rj
            new_rj = 2 - ri
        else:  # שמאלה
            new_ri = 2 - rj
            new_rj = ri

        # החזרה לקואורדינטות עולמיות
        return (new_ri + base_row, new_rj + base_col)

    def grading_boards (self):
        score = self.winner
        for board in reversed(self.boards_history):
            self.grades.append(score)
            score *= self.gamma
        self.grades.reverse()


class Pentago_run_games():
    def __init__(self, num_games):
        self.X = []
        self.Y = []
        self.num_games = num_games#how many times to run the ame
        self.agent_wins = 0#amount of agent winnings
        self.player_wins = 0#amount of player winnings
        self.ties = 0#amount of ties
        self.gamma = 0.9
        self.dict1 = {}
        self.dict2 = {}
        self.dict3 = {}
        self.dict4 = {}
        self.dict5 = {}
        self.dict6 = {}
        self.dict7 = {}
        self.smart_agent = SmartAgent(np.zeros((6, 6)))

        self.pentago = Pentago_game(
            self.dict1, self.dict2, self.dict3,
            self.dict4, self.dict5, self.dict6, self.dict7,
            self.smart_agent
        )

    def run(self):
        sum_count = 0
        for i in range(self.num_games):
            count = 0
            win, count = self.pentago.play_one_game()
            sum_count+=count
            if win == 1:
                self.agent_wins += 1
            elif win == -1:
                self.player_wins += 1
            else:
                self.ties += 1
            print(i," ", len(self.pentago.boards_history))

            self.pentago.grading_boards()
            self.divide_to_dicts()
        print("used the dictionaries ", (sum_count/self.num_games)*100)


    def reload_dict_from_json(
            self,
            filename1, filename2, filename3,
            filename4, filename5, filename6, filename7
    ):
        print(" --- loading data from json files ---")
        # רשימת שמות הקבצים שקיבלנו כארגומנטים
        filenames = [filename1, filename2, filename3, filename4, filename5, filename6, filename7]

        # נעבור על כל קובץ ונשמור את המילון שנוצר ברשימה זמנית
        loaded_dicts = []

        for i, fname in enumerate(filenames, 1):
            if not os.path.exists(fname):
                print(f"Warning: File {fname} not found. Skipping...")
                loaded_dicts.append({})  # או ערך ברירת מחדל אחר
                continue

            try:
                with open(fname, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                    # המרה של המפתחות ל-int והערכים ל-tuple
                    d = {int(k): tuple(v) for k, v in raw.items()}
                    loaded_dicts.append(d)
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                print(f"Error parsing {fname}: {e}")
                loaded_dicts.append({})
        print(" --- loaded dicts --- ")

        # עדכון המילונים באובייקט הנוכחי (self)
        self.dict1, self.dict2, self.dict3, self.dict4, self.dict5, self.dict6, self.dict7 = loaded_dicts

        # טעינת equivalent_set
        if os.path.exists("equivalent_set.json"):
            with open("equivalent_set.json", 'r', encoding='utf-8') as f:
                self.equivalent_set = set(json.load(f))
        else:
            self.equivalent_set = set()

        # עדכון האובייקט pentago - שימוש בלולאה כדי לחסוך שורות
        for i in range(1, 8):
            setattr(self.pentago, f'dict{i}', getattr(self, f'dict{i}'))

        self.pentago.eq_boards = set(self.equivalent_set)

    def divide_to_dicts(self):
        for i in range(1, len(self.pentago.boards_history)):
            board = self.pentago.boards_history[i]
            score = self.pentago.grades[i]



            # מוצאים את המילון המתאים לפי האינדקס
            target_dict = self.pentago.choose_right_dictionary(i)

            if board in target_dict:
                old_avg, old_count = target_dict[board]
                new_count = old_count + 1
                new_avg = (old_avg * old_count + score) / new_count
                target_dict[board] = (new_avg, new_count)
            else:
                target_dict[board] = (score, 1)

    def save_dicts_to_json(self, filename1,filename2,filename3,filename4):
        with open(filename1, 'w', encoding='utf-8') as f:
            json.dump(self.serialize_dict(self.dict1), f, indent=4)
        with open(filename2, 'w', encoding='utf-8') as f:
            json.dump(self.serialize_dict(self.dict2), f, indent=4)
        with open(filename3, 'w', encoding='utf-8') as f:
            json.dump(self.serialize_dict(self.dict3), f, indent=4)
        with open(filename4, 'w', encoding='utf-8') as f:
            json.dump(self.serialize_dict(self.dict4), f, indent=4)
        with open("dict5_pentago.json", "w", encoding="utf-8") as f:
            json.dump(self.serialize_dict(self.dict5), f, indent=4)
        with open("dict6_pentago.json", "w", encoding="utf-8") as f:
            json.dump(self.serialize_dict(self.dict6), f, indent=4)
        with open("dict7_pentago.json", "w", encoding="utf-8") as f:
            json.dump(self.serialize_dict(self.dict7), f, indent=4)


    def serialize_dict(self, d):
        return {str(k): [v[0], v[1]] for k, v in d.items()}

    def winnings_precentage(self):
        return "Agent: ",self.agent_wins / self.num_games*100,"%, Player: ",self.player_wins / self.num_games*100,"%"

if __name__ == "__main__":
    games = Pentago_run_games(5000)


    games.reload_dict_from_json("dict1_pentago.json", "dict2_pentago.json", "dict3_pentago.json",
                                    "dict4_pentago.json", "dict5_pentago.json", "dict6_pentago.json",
                                    "dict7_pentago.json")
    games.run()
    print(games.winnings_precentage())
    games.save_dicts_to_json("dict1_pentago.json", "dict2_pentago.json", "dict3_pentago.json", "dict4_pentago.json")

