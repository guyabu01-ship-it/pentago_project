from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.popup import Popup
from kivy.graphics import Color, Ellipse
from kivy.core.window import Window
from tensorflow.keras.models import load_model
import numpy as np
import random
from pentago import *

Window.clearcolor = (0.95, 0.95, 0.95, 1)


# ---------------- MENU ---------------- #

class MenuScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=50, spacing=20)

        layout.add_widget(Label(text="PENTAGO AI",
                                font_size='40sp',
                                color=(0.2, 0.4, 0.6, 1)))

        rules_btn = Button(text="How to Play",
                           size_hint_y=None,
                           height=50,
                           background_color=(0.4, 0.4, 0.4, 1))
        rules_btn.bind(on_release=self.show_rules)
        layout.add_widget(rules_btn)

        layout.add_widget(Label(text="Select Opponent Mode:",
                                font_size='20sp',
                                color=(0, 0, 0, 1)))

        modes = [("Random Agent", "RANDOM"),
                 ("Q-Learning", "Q_LEARNING"),
                 ("CNN Model", "CNN"),
                 ("ANN Model", "ANN")]

        for text, mode in modes:
            btn = Button(text=text,
                         size_hint_y=None,
                         height=60,
                         background_color=(0.2, 0.4, 0.6, 1))
            btn.bind(on_release=lambda instance, m=mode: self.start_game(m))
            layout.add_widget(btn)

        self.add_widget(layout)

    def show_rules(self, instance):
        content = BoxLayout(orientation='vertical', padding=10)
        rules_text = (
            "1. Place a marble on any empty spot.\n"
            "2. Rotate one of the four quadrants 90 degrees (L/R).\n"
            "3. First to get 5 marbles in a row (horizontally,\n"
            "   vertically, or diagonally) wins.\n"
            "4. If a rotation creates a win for both, it's a tie!"
        )
        content.add_widget(Label(text=rules_text, halign='center', valign='middle'))
        close_btn = Button(text="Got it!", size_hint_y=None, height=50)
        popup = Popup(title='Pentago Rules', content=content, size_hint=(0.8, 0.5))
        close_btn.bind(on_release=popup.dismiss)
        content.add_widget(close_btn)
        popup.open()

    def start_game(self, mode):
        app = App.get_running_app()
        app.game_mode = mode
        game_screen = self.manager.get_screen('game')
        interface = game_screen.children[0]
        interface.ctrl.reset_game()
        interface.refresh_ui()
        interface.update_status("Your turn!")
        self.manager.current = 'game'


# ---------------- CELL ---------------- #

class CellButton(Button):
    def __init__(self, row, col, **kwargs):
        super().__init__(**kwargs)
        self.row = row
        self.col = col
        self.value = 0
        self.background_normal = ""
        self.background_color = (0.7, 0.7, 0.7, 1)

    def on_size(self, *args):
        self.update_canvas()

    def update_canvas(self):
        self.canvas.after.clear()
        if self.value == 0:
            return
        with self.canvas.after:
            if self.value == 1:
                Color(1, 0.84, 0, 1)
            else:
                Color(0.1, 0.1, 0.1, 1)
            size = min(self.width, self.height) * 0.8
            Ellipse(
                pos=(self.center_x - size / 2,
                     self.center_y - size / 2),
                size=(size, size)
            )


# ---------------- CONTROLLER ---------------- #

class PentagoController:
    def __init__(self, model, interface):
        self.model = model
        self.interface = interface
        self.state = "PLACE_MARBLE"
        self.turn_counter = 1

    def reset_game(self):
        self.model.init_board()
        self.state = "PLACE_MARBLE"
        self.turn_counter = 1
        self.interface.update_status("Your turn!")

    def check_game_end(self, last_i=None, last_j=None):
        # פונקציית עזר לבדיקת ניצחון של שחקן ספציפי על כל הלוח ללא פקודת any
        def has_player_won(player_sign):
            for r in range(6):
                for c in range(6):
                    if self.model.board[r, c] == player_sign:
                        # שימוש ב-check_win המקורי מהמודל שלך עם הקואורדינטות הנכונות
                        if self.model.check_win(r, c):
                            return True
            return False

        player_won = has_player_won(self.model.player_sign)
        agent_won = has_player_won(self.model.agent_sign)

        if player_won and agent_won:
            self.interface.update_status("It's a Tie! (Both win)")
            self.state = "GAME_OVER"
            return True
        elif player_won:
            self.interface.update_status("You win!")
            self.state = "GAME_OVER"
            return True
        elif agent_won:
            self.interface.update_status("Agent wins!")
            self.state = "GAME_OVER"
            return True
        elif self.model.is_tie():
            self.interface.update_status("Tie game!")
            self.state = "GAME_OVER"
            return True
        return False

    def handle_marble_click(self, btn):
        if self.state != "PLACE_MARBLE":
            return
        if self.model.board[btn.row, btn.col] == 0:
            self.model.board[btn.row, btn.col] = self.model.player_sign
            btn.value = self.model.player_sign
            btn.update_canvas()

            # שליחת הקואורדינטות לבדיקת סוף משחק
            if self.check_game_end(btn.row, btn.col):
                return
            self.state = "ROTATE_QUADRANT"
            self.interface.update_status("Select a quadrant to rotate")

    def handle_rotate(self, q_id, direction):
        if self.state != "ROTATE_QUADRANT":
            return
        self.model.turn_quarter(q_id, direction)
        self.interface.refresh_ui()

        # בדיקה אחרי סיבוב - כאן חשוב לבדוק את כל הלוח כי ניצחון יכול להופיע בכל מקום
        if self.check_game_end():
            return
        self.state = "AGENT_TURN"
        self.interface.update_status("Agent is thinking...")
        self.agent_move()

    def agent_move(self):
        app = App.get_running_app()
        mode = app.game_mode
        if mode == "RANDOM":
            i, j = self.model.random_turn(self.model.agent_sign)
        elif mode == "Q_LEARNING":
            i, j, _ = self.model.exploit(self.turn_counter)
        elif mode == "CNN":
            i, j, q, s = self.cnn_move()
            self.model.apply_full_move(i, j, q, s, self.model.agent_sign)
        elif mode == "ANN":
            i, j, q, s = self.ann_move()
            self.model.apply_full_move(i, j, q, s, self.model.agent_sign)

        self.interface.refresh_ui()
        # בדיקה אחרי תור המחשב
        if self.check_game_end():
            return
        self.turn_counter += 1
        self.state = "PLACE_MARBLE"
        self.interface.update_status("Your turn!")

    def ann_move(self):
        model_nn = App.get_running_app().ann_model
        move = self.model.win_turn(self.model.agent_sign)
        if move: return move[0], move[1], None, None
        move = self.model.block(self.model.agent_sign, self.model.player_sign)
        if move: return move[0], move[1], None, None

        best_score = -float('inf')
        best_move = None
        empty = np.argwhere(self.model.board == 0)
        np.random.shuffle(empty)
        for (i, j) in empty[:10]:
            input_board = self.model.board.reshape(1, 6, 6, 1)
            score = model_nn.predict(input_board, verbose=0)[0][0]
            if score > best_score:
                best_score = score
                best_move = (i, j, random.randint(1, 4), random.randint(1, 2))
        return best_move if best_move else (empty[0][0], empty[0][1], 1, 1)

    def cnn_move(self):
        model_nn = App.get_running_app().cnn_model
        move = self.model.win_turn(self.model.agent_sign)
        if move: return move[0], move[1], None, None
        move = self.model.block(self.model.agent_sign, self.model.player_sign)
        if move: return move[0], move[1], None, None

        best_score = -float('inf')
        best_move = None
        empty = np.argwhere(self.model.board == 0)
        np.random.shuffle(empty)
        for (i, j) in empty[:10]:
            input_board = self.model.board.reshape(1, 6, 6, 1)
            score = model_nn.predict(input_board, verbose=0)[0][0]
            if score > best_score:
                best_score = score
                best_move = (i, j, random.randint(1, 4), random.randint(1, 2))
        return best_move if best_move else (empty[0][0], empty[0][1], 1, 1)


# ---------------- QUADRANT & INTERFACE ---------------- #

class Quadrant(GridLayout):
    def __init__(self, quad_id, **kwargs):
        super().__init__(**kwargs)
        self.cols = 3
        self.rows = 3
        self.spacing = 3
        self.marbles = []
        r_off = 0 if quad_id in [1, 2] else 3
        c_off = 0 if quad_id in [1, 3] else 3
        for r in range(3):
            for c in range(3):
                m = CellButton(row=r + r_off, col=c + c_off)
                self.add_widget(m)
                self.marbles.append(m)


class PentagoInterface(FloatLayout):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.ctrl = PentagoController(model, self)
        self.board_grid = GridLayout(cols=2, size_hint=(0.7, 0.7), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        self.quadrants = []
        for q_id in range(1, 5):
            q = Quadrant(q_id)
            for m in q.marbles:
                m.bind(on_release=self.ctrl.handle_marble_click)
            box = BoxLayout(orientation='vertical')
            box.add_widget(q)
            btns = BoxLayout(size_hint_y=None, height=50)
            btns.add_widget(Button(text="L", on_release=lambda x, q=q_id: self.ctrl.handle_rotate(q, 2)))
            btns.add_widget(Button(text="R", on_release=lambda x, q=q_id: self.ctrl.handle_rotate(q, 1)))
            box.add_widget(btns)
            self.board_grid.add_widget(box)
            self.quadrants.append(q)
        self.add_widget(self.board_grid)
        self.status_label = Label(text="Welcome!", font_size='35sp', color=(0, 0, 0, 1), bold=True,
                                  pos_hint={'center_x': 0.5, 'y': 0.05})
        self.add_widget(self.status_label)
        self.back_btn = Button(text="Back", size_hint=(0.1, 0.05), pos_hint={'x': 0.02, 'top': 0.98},
                               background_color=(0.8, 0.3, 0.3, 1))
        self.back_btn.bind(on_release=self.go_back)
        self.add_widget(self.back_btn)

    def go_back(self, instance):
        self.ctrl.reset_game()
        app = App.get_running_app()
        app.root.current = 'menu'

    def refresh_ui(self):
        for q in self.quadrants:
            for m in q.marbles:
                m.value = self.model.board[m.row, m.col]
                m.update_canvas()

    def update_status(self, text):
        self.status_label.text = text


# ---------------- APP ---------------- #

class FinalPentagoApp(App):
    game_mode = "RANDOM"

    def build(self):
        self.runner = Pentago_run_games(num_games=1)
        self.model = self.runner.pentago
        self.model.init_board()
        self.ann_model = load_model("pentago_ann_model.keras")
        self.cnn_model = load_model("pentago_cnn_model.keras")
        sm = ScreenManager()
        sm.add_widget(MenuScreen(name='menu'))
        game_screen = Screen(name='game')
        game_screen.add_widget(PentagoInterface(model=self.model))
        sm.add_widget(game_screen)
        return sm


if __name__ == "__main__":
    FinalPentagoApp().run()