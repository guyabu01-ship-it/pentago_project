from keras.models import load_model
import numpy as np


class SmartAgent:

    model = None  

    def __init__(self, board):

        if SmartAgent.model is None:
            SmartAgent.model = load_model("pentago_cnn4_model.keras")

        self.model = SmartAgent.model
        self.board = np.array(board)



    def turn_quarter(self, board, number, side):
        """
        number: 1–4
        side: 1=right , 2=left
        """

        board = board.copy()

        if number == 1:
            r, c = slice(0, 3), slice(0, 3)

        elif number == 2:
            r, c = slice(0, 3), slice(3, 6)

        elif number == 3:
            r, c = slice(3, 6), slice(0, 3)

        elif number == 4:
            r, c = slice(3, 6), slice(3, 6)

        else:
            return board


        sub = board[r, c]

        k = -1 if side == 1 else 1

        board[r, c] = np.rot90(sub, k=k)

        return board


    # יצירת כל המהלכים האפשריים

    def get_possible_moves(self, player=1):

        moves = []

        board = self.board

        for row in range(6):
            for col in range(6):

                if board[row][col] == 0:

                    for quadrant in range(1, 5):

                        for direction in [1, 2]:

                            moves.append(
                                (row, col, quadrant, direction)
                            )

        return moves


    # הפעלת מהלך

    def apply_move(self, move, player=1):

        row, col, quadrant, direction = move

        new_board = self.board.copy()

        # הצבת כדור
        new_board[row][col] = player

        # סיבוב רבע
        new_board = self.turn_quarter(
            new_board,
            quadrant,
            direction
        )

        return new_board


    # הערכת לוח עם CNN

    def evaluate_boards_batch(self, boards):

        boards_array = np.array(boards)

        boards_array = boards_array.reshape(-1, 6, 6, 1)

        scores = self.model.predict(
            boards_array,
            verbose=0
        )

        return scores.flatten()


    # בחירת מהלך ע"י CNN

    def choose_move_with_cnn(self, player=1):

        legal_moves = self.get_possible_moves(player)

        boards_after_moves = []

        # יצירת כל הלוחות
        for move in legal_moves:

            new_board = self.apply_move(
                move,
                player
            )

            boards_after_moves.append(new_board)


        # חיזוי לכל הלוחות בבת אחת 
        scores = self.evaluate_boards_batch(
            boards_after_moves
        )


        # בחירת הטוב ביותר
        best_index = np.argmax(scores)

        best_move = legal_moves[best_index]


        return best_move

# בדיקה

if __name__ == "__main__":

    board = np.zeros((6, 6), dtype=int)

    agent = SmartAgent(board)

    move = agent.choose_move_with_cnn(player=1)

    print("Best move:", move)
