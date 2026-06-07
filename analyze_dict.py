import numpy as np
import json
from scipy.signal import convolve2d

# הגדרת קרנלים לזיהוי רצפים של 5
# שורה, עמודה ושני אלכסונים
HORIZONTAL = np.ones((1, 5), dtype=int)
VERTICAL = np.ones((5, 1), dtype=int)
DIAG1 = np.eye(5, dtype=int)
DIAG2 = np.fliplr(DIAG1)
KERNELS = [HORIZONTAL, VERTICAL, DIAG1, DIAG2]


def base3_to_board_fast(key):
    """המרה מהירה של בסיס 3 למערך numpy"""
    temp_key = int(key)
    # יצירת מערך שטוח והמרה לדו-מימד בסוף מהירה יותר
    flat = np.zeros(36, dtype=int)
    for i in range(36):
        digit = temp_key % 3
        if digit == 1:
            flat[i] = 1
        elif digit == 2:
            flat[i] = -1
        temp_key //= 3
    return flat.reshape(6, 6)


def check_fast(board, player_val):
    """בדיקת ניצחון או איום בשיטה מתמטית (קונבולוציה) במקום מחרוזות"""
    p_board = (board == player_val).astype(int)
    empty_board = (board == 0).astype(int)

    has_5 = False
    has_4_open = False

    for kernel in KERNELS:
        # סכימת כמות הכלים בכל חלון של 5
        counts = convolve2d(p_board, kernel, mode='valid')

        if (counts == 5).any():
            return True, True  # אם יש 5, יש גם "איום" טכני, נצא מיד

        if (counts == 4).any():
            # בדיקה אם ה-4 הזה מלווה במשבצת ריקה באותו חלון
            empty_counts = convolve2d(empty_board, kernel, mode='valid')
            if ((counts == 4) & (empty_counts == 1)).any():
                has_4_open = True

    return has_5, has_4_open


def get_status_optimized(board, player):
    """סריקה אופטימלית של כל הסיבובים עם יציאה מוקדמת"""
    # בדיקה לפני סיבוב
    w, t = check_fast(board, player)
    if w: return True, True

    # בדיקה אחרי סיבובים
    for q in range(4):  # 4 רבעים
        r, c = (slice(0, 3), slice(0, 3)) if q == 0 else \
            (slice(0, 3), slice(3, 6)) if q == 1 else \
                (slice(3, 6), slice(0, 3)) if q == 2 else \
                    (slice(3, 6), slice(3, 6))

        orig_q = board[r, c]
        # סיבוב ימינה
        board[r, c] = np.rot90(orig_q, k=-1)
        w_post, t_post = check_fast(board, player)
        board[r, c] = orig_q  # החזרה למצב קודם (חוסך copy)

        if w_post: return True, True
        if t_post: t = True

        # סיבוב שמאלה
        board[r, c] = np.rot90(orig_q, k=1)
        w_post, t_post = check_fast(board, player)
        board[r, c] = orig_q

        if w_post: return True, True
        if t_post: t = True

    return w, t


def run_stats_fast(file_paths):
    categories = {
        "Agent_Win_Next": [], "Opponent_Win_Next": [],
        "Agent_4_Open_Safe": [], "Opponent_4_Open_Safe": []
    }

    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except:
            continue

        print(f"Analyzing {file_path}...")
        for key, val in data.items():
            score = val[0]
            board = base3_to_board_fast(key)

            # בדיקת סוכן
            a_win, a_4 = get_status_optimized(board, 1)
            # בדיקת יריב
            o_win, o_4 = get_status_optimized(board, -1)

            if a_win: categories["Agent_Win_Next"].append(score)
            if o_win: categories["Opponent_Win_Next"].append(score)

            if a_4 and not o_win and not a_win:
                categories["Agent_4_Open_Safe"].append(score)
            if o_4 and not a_win and not o_win:
                categories["Opponent_4_Open_Safe"].append(score)

    # הדפסה (זהה למקור)
    print(f"\n{'Category':<25} | {'Count':<10} | {'Avg Score'}")
    print("-" * 55)
    for cat, scores in categories.items():
        avg = sum(scores) / len(scores) if scores else 0
        print(f"{cat:<25} | {len(scores):<10} | {avg:.5f}")


if __name__ == "__main__":
    files = [f'dict{i}_pentago.json' for i in range(1, 8)]
    run_stats_fast(files)
