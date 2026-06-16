# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.3.5",
#     "jax==0.7.2",
#     "einops==0.8.1",
#     "tqdm==4.67.1",
#     "matplotlib==3.10.8",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    # IMPORTS
    import marimo as mo
    import sys
    import os
    import time
    import numpy as np
    import jax
    import jax.numpy as jnp
    import html


    # Sanity check that Jax is working
    #print(f"JAX version: {jax.__version__}")
    #print(f"Devices: {jax.devices()}")


    @jax.jit
    def test_jit(x):
        return x**2


    # The first call triggers the Metal compilation
    assert jnp.all(test_jit(jnp.array([1, 2, 3])) == jnp.array([1, 4, 9]))

    from tqdm import tqdm
    from einops import reduce, rearrange, repeat
    #import optax
    #import flax.linen as nn
    from functools import partial
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.transforms as transforms

    # jaxtyping:
    # This makes the inputs/outputs for each function much more explicit
    # but its optional, so if it doesn't work just use orindarly typing
    #try:
        # 1. Try to import everything normally
        #import jaxtyping
        #from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
    #except ImportError:
        #import warnings

        #warnings.warn(
        #    "jaxtyping is not installed. Type hints will be ignored.",
        #    ImportWarning,
        #)

    # 2. Create a dummy object that allows bracket notation like Float[Array, "N"]
    class DummyType:
        def __getitem__(self, item):
            return None

    # 3. Fallback all your specific types to the dummy object
    Int = Bool = Float = PRNGKeyArray = Array = DummyType()

    #from typing import NamedTuple, Callable


@app.cell(hide_code=True)
def _():
    mo.vstack(
        [
            mo.md(f"# Notebook Setup Instructions"),
            mo.md(
                f"On bottom right control panel, press {mo.icon('lucide:layout-template')} to hide code and enter app mode, and press {mo.icon('lucide:play')} to run everything!"
            ),
            mo.md("# Table of Contents"),
            mo.outline(label=""),
        ]
    )
    return


@app.cell(hide_code=True)
def gamerule_ui_elements():
    # UI elements fstor game rules
    # must be defined before they are used in Marimo
    forced_move_rule_checkbox = mo.ui.checkbox(
        label='<div data-tooltip="Forces taking immediate wins and blocking opponent wins (like check rule in chess). Greatly improves rollout quality!"> Forced Move Rule </div>',
        value=True,
    )
    gravity_checkbox = mo.ui.checkbox(
        label='<div data-tooltip="If enabled, then pieces fall down to the bottom (like in Connect4)"> Gravity </div>',
        value=True,
    )
    diagonals_count_checkbox = mo.ui.checkbox(
        label='<div data-tooltip="If enabled, then diagonals count for getting k in a row. Otherwise, they must be either up/down or left/right"> Diagonals count </div>',
        value=True,
    )
    n_rows_slider = mo.ui.slider(
        label="Rows",
        start=3,
        stop=10,
        value=6,
        show_value=True,
        debounce=True,
    )
    n_cols_slider = mo.ui.slider(
        label="Columns",
        start=3,
        stop=10,
        value=7,
        show_value=True,
        debounce=True,
    )
    k_target_slider = mo.ui.slider(
        label="Number in a line needed to win",
        start=3,
        stop=5,
        value=4,
        show_value=True,
        debounce=True,
    )
    random_start_turn_slider = mo.ui.slider(
        label='<div data-tooltip="Start all games from a random starting position with a number of random moves played. This makes for a more dynamic opennings and/or can out the balance Player 1 advangage"> Random Start Moves </div> ',
        start=0,
        stop=5,
        value=0,
        show_value=True,
        debounce=True,
    )
    random_seed_checkbox = mo.ui.checkbox(
        label='<div data-tooltip="Used for random initial turns: The seed tells you the columns of the moves, e.g. 112 means plays in columns 1, 1 and 2 to start"> Use seed </div>',
        value=False,
    )
    random_seed_num = mo.ui.number(
        label='<div data-tooltip="Enter the seed number for the starting moves here , e.g. 112 means plays in columns 1, 1 and 2 to start"> Seed</div>',
        value=0,
    )


    GLOBAL_BACKGROUND_COLOR = "#faf8f5"
    DARK_BACKGROUND_COLOR = "#ede9e2"  # EDE9E2
    # off white grey #f4f7f6
    # off white yellow #f7f6f4
    # warm paper #faf8f5 FAF8F5
    # linen cream #f5f0e8 F5F0E8

    # P1_COLOR = #D62728 / #1F77B4
    # P2_COLOR = #FF7F0E / #2CA02C

    mo.Html(
        f"""
        <style>
            :root {{
                /* Your chosen background color (e.g., Vanilla Cream) */
                --background: {GLOBAL_BACKGROUND_COLOR} !important;
                --background-darker: {GLOBAL_BACKGROUND_COLOR} !important;

                /* Make the code cells completely see-through */
                --card: transparent !important; 

                /* Make the borders transparent so there are no boxes around the code */
                --border: transparent !important;

                /* If the actual typing area (CodeMirror) still has a slight tint, make it transparent too */
                --input: transparent !important;
            }}
            /* * TARGET CODEMIRROR DIRECTLY
             * This forces the actual text editor to drop its white background 
             */
            .cm-editor, 
            .cm-scroller, 
            .cm-content {{
                background-color: transparent !important;
            }}

            /* Optional: Remove the background from the active line highlight if it looks blocky */
            .cm-activeLine, 
            .cm-activeLineGutter {{
                background-color: transparent !important;
            }}
        </style>
        </style>
        """
    )
    return (
        DARK_BACKGROUND_COLOR,
        GLOBAL_BACKGROUND_COLOR,
        diagonals_count_checkbox,
        forced_move_rule_checkbox,
        gravity_checkbox,
        k_target_slider,
        n_cols_slider,
        n_rows_slider,
        random_seed_checkbox,
        random_seed_num,
        random_start_turn_slider,
    )


@app.cell(hide_code=True)
def _():
    # mo.md(r"""
    # Learning about AlphaZero with k-in-a-line games : Monte Carlo Tree Search

    # This interactive notebook teaches you about the AlphaZero algorithm using k-in-a-line games (like Connect 4 or Tic-Tac-Toe). This particular notebook is about **Monte Carlo Tree Search**, which is a way of exploring the search space of a game. There is a previous notebook about the **Upper Confidence Bound Algorithm** and a future notebook about using **Neural Networks**.
    # """)
    return


@app.cell(hide_code=True)
def gamerule_settings(
    diagonals_count_checkbox,
    forced_move_rule_checkbox,
    gravity_checkbox,
    k_target_slider,
    n_cols_slider,
    n_rows_slider,
    random_seed_checkbox,
    random_seed_num,
    random_start_turn_slider,
):
    ### GLOBAL GAME PARAMETERS
    # UI To set some game paramters


    # -------------------------------------------
    # WARNING: If you try to change these after you have jax.jit compiled
    # the functions, the comiled function might not see the change. You
    # may need to rerun in!

    # Size of the board
    N_ROWS = n_rows_slider.value
    N_COLS = n_cols_slider.value

    # The maximum possible turns in a game
    MAX_TURNS = n_rows_slider.value * n_cols_slider.value

    # Goal of the game is to get this many in a line!
    # You instantly win if you get this many!
    K_TARGET = min(n_rows_slider.value, n_cols_slider.value, k_target_slider.value)  # 4

    # Are we playing like Connect 4 with gravity or like tic-tac-toe where you can play anywhere?
    # NOTE: While all the core game mechanics shoudl work with GRAVITY = False, I only made the UI for GRAVITY=True since I wanted to make connect 4 the main example
    GRAVITY = gravity_checkbox.value


    # Do diagonals count for getting your k-in-a-row?
    DIAGONALS_COUNT = diagonals_count_checkbox.value

    # If playing a gravity game, then one possible action per column
    # If no gravvity, then one possible action for each possible square
    N_ACTIONS = N_COLS if GRAVITY else N_COLS * N_ROWS

    # Note that this is the *maximum* number of actions available to you at any time,
    # we will use boolean "legal move" masks to handle cases where available actions become
    # restricted by the board state

    # The "forced move rule" says that if you are about to lose the game, you are "in check" and it is illegal to NOT block the move. Similarly, if you are about to win, you MUST win. IN other words, it is illegal to make a move that 1. misses an instant win or 2. misses an opportunity to block an opponents instant win
    FORCED_MOVE_RULE = forced_move_rule_checkbox.value

    # How many random moves to play before the players get to start playing. Makes for more varied opennings! 1 is nice to offset first player advantage. 3 is nice to really mix things up. Classic Connect 4 simply has 0 random starting moves
    RANDOM_START_TURNS = random_start_turn_slider.value

    modify_game_rules = mo.accordion(
        {
            "Modify game rules": mo.callout(
                mo.vstack(
                    [
                        mo.hstack([n_rows_slider, n_cols_slider], justify="start"),
                        k_target_slider,
                        mo.hstack(
                            [
                                diagonals_count_checkbox,
                                gravity_checkbox,
                                forced_move_rule_checkbox,
                            ],
                            justify="start",
                        ),
                    ]
                )
            ),
        }
    )
    modify_game_start = mo.accordion(
        {
            "Random opening rule": mo.callout(
                mo.vstack(
                    [
                        random_start_turn_slider,
                        mo.hstack(
                            [
                                random_seed_checkbox,
                                random_seed_num,
                            ],
                            justify="start",
                        ),
                    ]
                )
            )
        }
    )
    # modify_game_rules
    return (
        DIAGONALS_COUNT,
        FORCED_MOVE_RULE,
        GRAVITY,
        K_TARGET,
        MAX_TURNS,
        N_ACTIONS,
        N_COLS,
        N_ROWS,
        RANDOM_START_TURNS,
        modify_game_rules,
        modify_game_start,
    )


@app.cell(hide_code=True)
def _game_class(
    DIAGONALS_COUNT,
    FORCED_MOVE_RULE,
    GRAVITY,
    K_TARGET,
    MAX_TURNS,
    N_ACTIONS,
    N_COLS,
    N_ROWS,
    RANDOM_START_TURNS,
):
    class Game:
        """
        A vectorized game environment for k in a line games (like tic-tac-toe, connect 4).

        BOARD ORIENTATION & GRAVITY:
        We use arrays of shape (N_ROWS, N_COLS) of Int to represent the board.
        0 = Empty square, 1 = Player 1's piece, 2 = Player 2's piece.
        (and we can switch player using opponent = 3 - player since 3-1=2 and 3-2=1)

        For rows, Index 0 is the TOP of board, Index N_ROWS-1 is the BOTTOM.

              Col0 Col1 Col2
        Row0 [ .    .    . ] (Top)
        Row1 [ .    .    . ]
        Row2 [ .    X    . ] (Bottom)

        Why this orientation? It makes checking if a column is empty easy
        (just check row 0). If a column has `c` pieces in it, the next piece
        simply lands at index `N_ROWS - 1 - c`. Also easier to print to text

        BATCHING & EINOPS:
        This class is highly optimized using JAX and Einops to support batching.
        The input boards can be of shape (*batch, N_ROWS, N_COLS) where
        the `*batch` dimensions in the type hints and `...` in einops strings
        allow these functions to run on a single board OR millions simultaneously.
        """

        @staticmethod
        def create_k_in_a_lines() -> Bool[
            Array, "num_winning_lines N_ROWS N_COLS"
        ]:
            """
            Creates a boolean array of all possible winning k-in-a-row lines
            based on the global board dimensions. Each slice along the zeroth-dimension is a
            2D board where `True` indicates a piece in a specific winning line.
            num_winning_lines is the number of possible k-in-a-lines that fit
            on the board, for example classic tic-tac-toe, num_winning_lines=8.

            Notes
            -----
            This function uses nested loops and is very slow!,
            but we only have to run it *once* to create a constant mask.
            """
            lines = []

            # We only need forward-facing directions to avoid duplicate lines:
            # (Right, Down, Down-Right, Up-Right)
            directions = [(0, 1), (1, 0)]
            if DIAGONALS_COUNT == True:
                directions.extend([(1, 1), (-1, 1)])

            for r in range(N_ROWS):
                for c in range(N_COLS):
                    for dr, dc in directions:
                        # Cast a ray from the current cell (r, c) in the given direction
                        # to find where the k-in-a-row line would end.
                        end_r = r + (K_TARGET - 1) * dr
                        end_c = c + (K_TARGET - 1) * dc

                        # If the end of the ray is still on the board, it's a valid win condition
                        if 0 <= end_r < N_ROWS and 0 <= end_c < N_COLS:
                            line = np.zeros((N_ROWS, N_COLS), dtype=bool)

                            # Use numpy advanced indexing to set all elements along the ray to True
                            line[
                                r + np.arange(K_TARGET) * dr,
                                c + np.arange(K_TARGET) * dc,
                            ] = True
                            lines.append(line)

            return jnp.stack(lines)

        # =====================================================================
        # Sanity Check (Using Tic-Tac-Toe dimensions as a mental model)
        # =====================================================================
        # For classic tic-tac-toe, N_ROWS=3, N_COLS=3, K_TARGET=3.
        # There are exactly 8 winning lines: 3 horizontal, 3 vertical, 2 diagonal.
        #
        # If you temporarily change the globals above to N_ROWS=3, N_COLS=3, K_TARGET=3, this assertion will pass:
        # assert create_k_in_a_lines().shape == (8, 3, 3)

        # We invert the mask (False for winning spots, True elsewhere) for a logical trick,
        # that lets us easily search for k-in-a-lines using "or"s later on.
        K_IN_A_LINE_MASK = jnp.logical_not(create_k_in_a_lines.__func__())

        @staticmethod
        @jax.jit
        def has_k_in_a_line(
            board_bool: Bool[Array, "*batch N_ROWS N_COLS"],
        ) -> Bool[Array, "*batch"]:
            """Checks if a boolean board mask contains a winning line."""
            # Add a '1' dim for broadcasting against the K_IN_A_LINE_MASK
            broadcastable_board = rearrange(
                board_bool, "... row col -> ... 1 row col"
            )

            # THE LOGICAL_OR TRICK:
            # K_IN_A_LINE_MASK is False at target spots and True elsewhere.
            # If board_bool fills the False gaps with Trues, the line is complete.
            line_check_board = jnp.logical_or(
                Game.K_IN_A_LINE_MASK, broadcastable_board
            )

            # Collapse spatial dims: 'all' requires the whole line to be True
            which_k_in_a_lines = reduce(
                line_check_board, "... n_line row col -> ... n_line", "all"
            )

            # Collapse lines dim: 'any' means if ANY winning line exists, it's a win
            return reduce(which_k_in_a_lines, "... n_line -> ...", "any")

        @staticmethod
        @jax.jit
        def standard_legal_move_mask(
            state: Int[Array, "*batch N_ROWS N_COLS"],
        ) -> Bool[Array, "*batch N_ACTIONS"]:
            """Returns a boolean mask of size N_ACTIONS telling us which actions are legal, assuming there is no check rule in play (i.e. classic, you are allowed to make mistakes rules).
            If GRAVITY==True, A move is legal if the top row (index 0) of that column is empty.
            If GRAVITY==False, A move is legal if that spot is empty."""
            if GRAVITY == True:
                # Action is a column. Legal if the top row is empty.
                return state[..., 0, :] == 0
            elif GRAVITY == False:
                # Action is a specific cell. Legal if the cell is 0.
                # We flatten the 2D board to match the 1D N_ACTIONS array.
                return rearrange(state == 0, "... row col -> ... (row col)")

        @staticmethod
        @jax.jit
        def forced_move_rule_legal_move_mask(
            state: Int[Array, "*batch N_ROWS N_COLS"],
            player_turn: int | Int[Array, "*batch"],
        ) -> Bool[Array, "*batch N_ACTIONS"]:
            """
            Returns a mask of legal moves that follow the 'forced check rule'.
            The forced check rule means you are not allowed to miss an instant win or miss blocking an opponents instant win.
            Priorities: 1. Take an instant win. 2. Block an instant loss. 3. Any legal move.
            """

            # This flag is used to indicate a hypothetical move by either us or the opponet
            HYPOTHETICAL_FLAG = 999
            legal_moves = Game.standard_legal_move_mask(state)
            opponent = 3 - player_turn

            # 1. Lookahead by forcing a move into all possible columns
            hypothetical_states = Game.step_all_actions(
                state, fill_value=HYPOTHETICAL_FLAG
            )

            # 2. Identify Tactical Masks
            # Does this move create a win for us?
            win_mask = Game.has_k_in_a_line(
                (hypothetical_states == player_turn)
                | (hypothetical_states == HYPOTHETICAL_FLAG)
            )
            # Would this move create a win for the opponent?
            # If so we need to block it!
            block_mask = Game.has_k_in_a_line(
                (hypothetical_states == opponent)
                | (hypothetical_states == HYPOTHETICAL_FLAG)
            )

            # 3. Check if a win or block exists anywhere in the action space
            has_win = reduce(win_mask, "... action -> ... 1", "any")
            has_block = reduce(block_mask, "... action -> ... 1", "any")

            # 4. The Priority Cascade: Win > Block > default to just Legal
            return jnp.where(
                has_win, win_mask, jnp.where(has_block, block_mask, legal_moves)
            )

        @staticmethod
        @jax.jit
        def legal_move_mask(
            state: Int[Array, "*batch N_ROWS N_COLS"],
            player_turn: int | Int[Array, "*batch"],
        ) -> Bool[Array, "*batch N_ACTIONS"]:
            """Returns the legal moves according to the rule set chosen"""
            if FORCED_MOVE_RULE == False:
                return Game.standard_legal_move_mask(state)
            elif FORCED_MOVE_RULE == True:
                return Game.forced_move_rule_legal_move_mask(state, player_turn)

        @staticmethod
        @jax.jit
        def reward(
            state: Int[Array, "*batch N_ROWS N_COLS"], player_turn: int = 1
        ) -> Float[Array, "*batch"]:
            """
            Returns +1.0 if player_pov wins, -1.0 if opponent wins, 0.0 otherwise.
            Note: Assumes the game stops immediately upon a win.
            """
            player_points = 1.0 * Game.has_k_in_a_line(state == player_turn)
            opponets_points = 1.0 * Game.has_k_in_a_line(
                state == (3 - player_turn)
            )
            return player_points - opponets_points

        @staticmethod
        @jax.jit
        def get_winner(
            state: Int[Array, "*batch N_ROWS N_COLS"], player_turn: int = 1
        ) -> Int[Array, "*batch"]:
            """
            Returns 1 if Player 1 wins, 2 if Player 2 wins, 0 otherwise.
            """
            return jnp.where(
                Game.has_k_in_a_line(state == 1),
                1,
                jnp.where(Game.has_k_in_a_line(state == 2), 2, 0),
            )

        @staticmethod
        @jax.jit
        def is_terminal(
            state: Int[Array, "*batch N_ROWS N_COLS"],
        ) -> Bool[Array, "*batch"]:
            """A state is terminal if someone won, or if no legal moves remain."""

            # Condition 1: Check explicitly if Player 1 or Player 2 has a winning line
            p1_won = Game.has_k_in_a_line(state == 1)
            p2_won = Game.has_k_in_a_line(state == 2)

            # Condition 2: Are there ANY legal moves left on the board?
            any_legal_moves = reduce(
                Game.standard_legal_move_mask(state), "... action -> ...", "any"
            )

            # Terminal if someone won, or if the board is completely full
            return p1_won | p2_won | ~any_legal_moves

        @staticmethod
        def to_string(state: Int[Array, "N_ROWS N_COLS"]) -> str:
            """Converts a single board array to a readable string (Not batched)."""
            symbols = {0: ".", 1: "X", 2: "O"}
            # symbols = {0: "⬜", 1: "🔴, 2: "🔵"}
            board_str = ""
            for row in range(N_ROWS):
                row_str = "".join(
                    symbols[int(state[row, col])] for col in range(N_COLS)
                )
                board_str += row_str + "\n"
            return board_str.strip()

        @staticmethod
        @jax.jit
        def step_all_actions(
            state: Int[Array, "*batch N_ROWS N_COLS"],
            fill_value: int | Int[Array, "*batch"] = 1,
        ) -> Int[Array, "*batch N_ACTIONS N_ROWS N_COLS"]:
            """
            Returns the next board state of ALL possible moves (even illegal ones),
            which are calculated simulataneously. This is a lot more efficent than
            applying the actions one by one. To do this, we add an extra axis of size
            N_ACTIONS, which indicates what action was taken.
            """
            # 1. Calculate landing spots for every possible action
            if GRAVITY:
                # In gravity mode, N_ACTIONS == N_COLS.
                col_ix_all = jnp.arange(N_COLS)
                # Find which row each column would play in by counting pieces
                pieces_in_cols = reduce(
                    state != 0, "... row col -> ... col", "sum"
                )
                row_ix_all = N_ROWS - 1 - pieces_in_cols
            else:
                # In no-gravity mode, coordinates are a static mapping.
                all_actions = jnp.arange(N_ACTIONS)
                row_ix_all = all_actions // N_COLS
                col_ix_all = all_actions % N_COLS

            # 2. Branch the universe: create N_ACTIONS copies of the current state
            repeated_state = repeat(
                state, "... row col -> ... action row col", action=N_ACTIONS
            )

            # 3. Align coordinates for a batched scatter-add
            # batch_action_indices handles the (*batch, N_ACTIONS) part of the target tensor
            batch_action_indices = jnp.indices(repeated_state.shape[:-2])

            # 4. Drop the pieces into all boards at once
            return repeated_state.at[
                (*batch_action_indices, row_ix_all, col_ix_all)
            ].add(fill_value)

        @staticmethod
        @jax.jit
        def step(
            state: Int[Array, "*batch N_ROWS N_COLS"],
            action: Int[Array, "*batch"],
            fill_value: int | Int[Array, "*batch"] = 1,
        ) -> Int[Array, "*batch N_ROWS N_COLS"]:
            """Applies a single specific action, and returns the resulting board."""
            # 1. Determine the landing coordinates for the specific action
            if GRAVITY:
                # If GRAVITY, then count pieces in the column to see where to put it
                pieces_in_cols = reduce(
                    state != 0, "... row col -> ... col", "sum"
                )

                # Extract the count for the chosen column (action)
                action_as_idx = rearrange(action, "... -> ... 1")
                col_count = jnp.take_along_axis(
                    pieces_in_cols, action_as_idx, axis=-1
                )
                col_count = rearrange(col_count, "... 1 -> ...")

                row_ix = N_ROWS - 1 - col_count
                col_ix = action
            else:
                row_ix = action // N_COLS
                col_ix = action % N_COLS

            # 2. Apply the single move to each board
            batch_indices = jnp.indices(action.shape)
            return state.at[(*batch_indices, row_ix, col_ix)].add(fill_value)

        @staticmethod
        @jax.jit
        def start_board(_key=None) -> Int[Array, "N_ROWS N_COLS"]:
            """Returns the starting board state."""
            if _key is None:
                _key = jax.random.PRNGKey(int(time.time()))
            return Game.random_rollout(
                jnp.zeros((N_ROWS, N_COLS), dtype=jnp.int32),
                player_turn=(RANDOM_START_TURNS % 2) + 1,
                key=_key,
                max_rollout_steps=RANDOM_START_TURNS,
            )[0]  # only return the board, not the key for simplicty

        @staticmethod
        @partial(jax.jit, static_argnames=("max_rollout_steps",))
        def random_rollout(
            board: Int[Array, "*batch N_ROWS N_COLS"],
            player_turn: int,
            key: PRNGKeyArray,
            max_rollout_steps: int = MAX_TURNS,
        ) -> tuple[Int[Array, "*batch N_ROWS N_COLS"], PRNGKeyArray]:
            """
            Fast JAX-native random rollout using jax.lax.scan over MAX_TURNS steps.
            Because the game is guaranteed to terminate within MAX_TURNS moves, we
            replace the dynamic while_loop with a fixed-length scan, which XLA can
            fully unroll and optimize at compile time.

            =====================================================================
            OUR RUN MANY GAMES IN A BATCH SYNCHRONIZATION STRATEGY
            =====================================================================
            Because JAX forces all games in a batch to execute in parallel lockstep,
            shorter games must wait for longer games to finish. We handle this with
            the following "Ghost Game" strategy:

            1. The loop runs as long as ANY game in the batch is still active.
            2. For finished games, we pretend all moves are legal to prevent NaNs
               during the math/sampling phase.
            3. We apply the sampled moves to the entire batch.
            4. We overwrite the board state of the finished games with their frozen,
               completed state, effectively discarding the "ghost" moves.
            """

            def step_fn(carry, _):
                current_board, current_player, rng_key, is_game_over = carry

                # determine possible actions
                possible_action_mask = Game.legal_move_mask(
                    current_board, current_player
                )

                # 2. Ghost game fix: finished games get all-True mask to avoid
                #    sampling from a zero-probability distribution
                game_over_action_bcast = rearrange(is_game_over, "... -> ... 1")
                safe_possible_action_mask = jnp.where(
                    game_over_action_bcast, True, possible_action_mask
                )

                # 3. Sample a random action
                action_logits = jnp.where(safe_possible_action_mask, 0.0, -jnp.inf)
                rng_key, subkey = jax.random.split(rng_key)
                action = jax.random.categorical(subkey, action_logits, axis=-1)

                # 4. Apply the move
                next_board = Game.step(
                    current_board, action, fill_value=current_player
                )

                # 5. Freeze finished boards — discard the ghost move
                game_over_board_bcast = rearrange(is_game_over, "... -> ... 1 1")
                updated_board = jnp.where(
                    game_over_board_bcast, current_board, next_board
                )

                # 6. Prepare next turn
                next_player = 3 - current_player
                next_is_game_over = Game.is_terminal(updated_board)

                carry = (updated_board, next_player, rng_key, next_is_game_over)
                return (
                    carry,
                    None,
                )  # None = we don't need to record intermediate states

            player_turn = jnp.asarray(player_turn, dtype=jnp.int32)
            initial_is_game_over = Game.is_terminal(board)

            init_carry = (board, player_turn, key, initial_is_game_over)

            (final_board, _, final_key, _), _ = jax.lax.scan(
                step_fn,
                init_carry,
                xs=None,
                length=max_rollout_steps,
            )

            return final_board, final_key

    return (Game,)


@app.cell(hide_code=True)
def tree_class(Game, N_ACTIONS, N_COLS, N_ROWS):
    class Tree:
        """
        An array-based tree structure for Monte Carlo Tree Search.

        Why Numpy AND Jax?
        - The tree variables (wins, N, parent_ix) use standard NumPy (np). This is because
          NumPy allows in-place updates (e.g., self.N_sims[idx] += 1), which is crucial for
          speed when building a tree. JAX arrays are immutable and would be too slow here.
        - We use JAX (jnp) for the board rules/simulations because it is faster
          at heavy math and vectorization.

        Note: A full MCTS loop involves 4 steps: Selection, Expansion, Simulation, Backpropagation.
        This Tree object handles Expansion, Simulation, and Backpropagation. The "Selection"
        logic (calculating UCB scores to pick the next node) is left to the external game loop.
        """

        ILLEGAL_CHILD_FLAG = -999
        ROOT_NODE_IX = 0

        def __init__(
            self,
            max_nodes=50_000,
            max_children=N_ACTIONS,
            root_board=None,
            root_player_turn=2,
        ):
            # maximum children for any given node
            self.max_children = max_children

            # maximum number of nodes in our tree; will be full once we get here!
            self.max_nodes = max_nodes

            # Pre-allocate large arrays for all the tree nodes now
            # (instead of making thousands of slow Python objects)

            # Every node has an ix in the range 0 to max_nodes.

            # The largest used node index, as we add new nodes this gets updated
            self.largest_used_node_ix = 0

            # All information about the nodes is stored in arrays
            # indexed by the node's ix. For example, self.board[ix] is the
            # board state of the node.
            self.board = np.zeros((max_nodes, N_ROWS, N_COLS), dtype=int)

            # The parent_ix and children_ixs
            self.parent_ix = np.zeros(max_nodes, dtype=int)
            self.children_ixs = np.zeros((max_nodes, max_children), dtype=int)

            # whose turn is it at this node? Is it a leaf? Is it terinal?
            self.player_turn = np.ones(max_nodes, dtype=int)
            self.is_leaf = np.ones(max_nodes, dtype=bool)
            self.is_terminal = np.zeros(max_nodes, dtype=bool)

            # Keeping track of simulation results for each node:
            self.total_sim_reward = np.zeros(
                max_nodes, dtype=float
            )  # the total reward over all the simulations we've done involving this ndoe

            self.ucb = np.full(max_nodes, np.nan, dtype=float)

            self.N_sims = np.zeros(
                max_nodes, dtype=int
            )  # Number of times we've simulated this node

            ## Add the root node to the tree
            if root_board is not None:
                self.board[Tree.ROOT_NODE_IX] = root_board
                self.player_turn[Tree.ROOT_NODE_IX] = root_player_turn
                self.is_terminal[Tree.ROOT_NODE_IX] = bool(
                    Game.is_terminal(self.board[Tree.ROOT_NODE_IX])
                )

            ####
            # Variables to keep track of simulation outcomes
            self.simulation_ix = (
                None  # the ix of the node we have simulated (usually a leaf node)
            )
            self.simulation_board = jnp.zeros(
                (N_ROWS, N_COLS), dtype=int
            )  # the outcome of the simulation (what the final board becamse)

            self.highlight_path = []  # this is the list of all the nodes from the simulation board, back up to the root.

            self.key = jax.random.PRNGKey(
                int(time.time())
            )  # random key passed into simulations

        def clear_sim(self):
            self.simulation_ix = (
                None  # the ix of the node we have simulated (usually a leaf node)
            )
            self.simulation_board = jnp.zeros(
                (N_ROWS, N_COLS), dtype=int
            )  # the outcome of the simulation (what the final board becamse)

        def expand_node(self, node_ix):
            """
            MCTS Step: EXPANSION.
            Evaluates all actions, claims N_ACTIONS array spaces, and logs the legal moves.
            """
            possible_move_mask = Game.legal_move_mask(
                self.board[node_ix], self.player_turn[node_ix]
            )

            # Check capacity. We now allocate exactly N_ACTIONS nodes every time,
            # trading memory efficiency for faster, constant-sized array insertions.
            if (
                self.largest_used_node_ix + N_ACTIONS < self.max_nodes
                and not self.is_terminal[node_ix]
            ):
                # 1. Get ALL hypothetical next boards at once using our highly JIT-able function
                all_next_boards = Game.step_all_actions(
                    self.board[node_ix], fill_value=self.player_turn[node_ix]
                )
                all_terminals = Game.is_terminal(all_next_boards)

                # 2. Claim the next N_ACTIONS indices in our pre-allocated arrays
                start_ix = self.largest_used_node_ix + 1
                child_ixs = np.arange(start_ix, start_ix + N_ACTIONS)

                # 3. Update the parent's children pointers.
                # If a move is illegal, point to ILLEGAL_CHILD_FLAG instead of the claimed index
                # so the selection phase knows it's a dead end.
                self.children_ixs[node_ix] = np.where(
                    np.asarray(possible_move_mask),
                    child_ixs,
                    self.ILLEGAL_CHILD_FLAG,
                )

                # 4. Write data for ALL N_ACTIONS children into the memory banks.
                # We store the illegal ones too because doing a straight block assignment
                # is significantly faster than masking and dynamically sizing arrays in Python.
                self.board[child_ixs] = np.asarray(all_next_boards)
                self.is_terminal[child_ixs] = np.asarray(all_terminals)
                self.parent_ix[child_ixs] = node_ix
                self.player_turn[child_ixs] = 3 - self.player_turn[node_ix]

                # 5. Update leaf status
                self.is_leaf[child_ixs] = True
                self.is_leaf[node_ix] = False

                # 6. Increment the frontier
                self.largest_used_node_ix += N_ACTIONS

        def simulate_node(self, node_ix):
            """
            MCTS Step: SIMULATION (Rollout).
            Takes a node and plays random (or blunder-free) moves until the game ends.
            """
            self.simulation_ix = node_ix
            self.simulation_board, self.key = Game.random_rollout(
                self.board[node_ix],
                self.player_turn[node_ix],
                self.key,
            )
            self.highlight_path = []

        def backup(self, c=1.414):
            """
            MCTS Step: BACKUP TREE AND RECORD RESULT.
            Walks backward up to the root, updating visit counts, win scores,
            and caching the updated UCB values for the next selection phase.
            """
            if self.simulation_ix is not None:
                # Get the reward from the final simulated board
                reward = float(Game.reward(self.simulation_board))
                temp_ix = self.simulation_ix
                self.highlight_path = [self.simulation_ix]

                # Climb the tree back to the root
                while True:
                    self.highlight_path.append(temp_ix)
                    self.N_sims[temp_ix] += 1
                    self.total_sim_reward[temp_ix] += reward

                    # --- RECALCULATE UCB FOR THIS NODE'S CHILDREN ---
                    # Only expanded nodes have children to update
                    if not self.is_leaf[temp_ix]:
                        child_ixs = self.children_ixs[temp_ix]
                        valid_mask = child_ixs != self.ILLEGAL_CHILD_FLAG

                        # 1. Gather stats
                        safe_child_ixs = np.where(valid_mask, child_ixs, 0)
                        n_child = self.N_sims[safe_child_ixs]
                        w_child = self.total_sim_reward[safe_child_ixs]
                        n_parent = self.N_sims[temp_ix]

                        # 2. Raw Win Rate
                        safe_n_child = np.where(n_child == 0, 1, n_child)
                        raw_win_rate = w_child / safe_n_child

                        # 3. Optimism Bonus
                        player_mult = (
                            1.0 if self.player_turn[temp_ix] == 1 else -1.0
                        )
                        optimism_bonus = (
                            c
                            * np.sqrt(np.log(n_parent + 1.0) / safe_n_child)
                            * player_mult
                        )

                        # 4. Calculate UCB and handle unvisited nodes
                        ucb = raw_win_rate + optimism_bonus
                        unexplored_bonus = np.inf * player_mult
                        ucb = np.where(n_child == 0, unexplored_bonus, ucb)

                        # 5. Write the updated UCBs back to the main array!
                        valid_indices = child_ixs[valid_mask]
                        valid_ucbs = ucb[valid_mask]
                        self.ucb[valid_indices] = valid_ucbs

                    # Move up the tree
                    if temp_ix == Tree.ROOT_NODE_IX:
                        break
                    else:
                        temp_ix = int(self.parent_ix[temp_ix])

        def select_leaf(self):
            """MCTS Step: SELECTION. Rapidly traverses the tree by looking up pre-calculated UCB values."""
            node_ix = Tree.ROOT_NODE_IX
            while not self.is_leaf[node_ix] and not self.is_terminal[node_ix]:
                child_ixs = self.children_ixs[node_ix]
                valid_mask = child_ixs != Tree.ILLEGAL_CHILD_FLAG

                safe_child_ixs = np.where(valid_mask, child_ixs, 0)

                # -------------------------------------------------------
                # NEW: use N_sims == 0 for unvisited detection
                # -------------------------------------------------------
                child_N = self.N_sims[safe_child_ixs]
                unvisited = valid_mask & (child_N == 0)

                if np.any(unvisited):
                    # select uniformly among unvisited children
                    candidates = np.flatnonzero(unvisited)
                    best_action = np.random.choice(candidates)

                else:
                    child_ucbs = self.ucb[safe_child_ixs]

                    player_mult = 1.0 if self.player_turn[node_ix] == 1 else -1.0
                    invalid_penalty = -np.inf * player_mult
                    child_ucbs = np.where(valid_mask, child_ucbs, invalid_penalty)

                    if self.player_turn[node_ix] == 1:
                        best_action = np.argmax(child_ucbs)
                    else:
                        best_action = np.argmin(child_ucbs)

                node_ix = int(child_ixs[best_action])

            return node_ix

        def run_mcts(self, num_iterations=50, c=1.414, show_progress=True):
            """
            Executes the full Monte Carlo Tree Search loop internally, by automatically selecting with UCB, exploring, and simulating with rollouts.
            """

            iterator = (
                # tqdm.tqdm(range(num_iterations), desc="MCTS Iterations")
                mo.status.progress_bar(
                    range(num_iterations),
                    title="MCTS Running ...",
                    subtitle="Please wait",
                    show_eta=True,
                    show_rate=True,
                )
                if show_progress
                else range(num_iterations)
            )

            for _ in iterator:
                # 1) SELECTION (Pure NumPy, inside the tree)
                current_ix = self.select_leaf()

                # 2) EXPANSION
                # If we selected a leaf that has already been visited, expand it.
                # Note that otherwise, we are at a leaf that has never been visited, so we can just run a rollout from there.
                if (
                    self.N_sims[current_ix] > 0
                    and not self.is_terminal[current_ix]
                ):
                    self.expand_node(current_ix)

                    # After expansion, pick a random valid child to drop down to simulate
                    child_ixs = self.children_ixs[current_ix]
                    valid_mask = child_ixs != self.ILLEGAL_CHILD_FLAG

                    valid_children = child_ixs[valid_mask]

                    if valid_children.size > 0:
                        current_ix = int(np.random.choice(valid_children))

                # 3) SIMULATION (Batched JAX) & 4) BACKPROPAGATION
                self.simulate_node(current_ix)
                self.backup()

            return

        def get_child_visits(self, node_ix=None):
            """
            Returns an array representing the number of visits each child node received.
            The index of the array corresponds to the action taken.
            Illegal actions will have a visit count of 0.
            """
            if node_ix is None:
                node_ix = Tree.ROOT_NODE_IX

            # 1. Grab the child IDs for all possible actions
            child_ixs = self.children_ixs[node_ix]

            # 2. Identify which actions are legal
            valid_mask = child_ixs != self.ILLEGAL_CHILD_FLAG

            # 3. Safely look up the visit counts
            # We temporarily point invalid children to node 0 so numpy doesn't crash
            safe_child_ixs = np.where(valid_mask, child_ixs, 0)
            visits = self.N_sims[safe_child_ixs]

            # 4. Mask the illegal moves with -1
            visits = np.where(valid_mask, visits, 0)

            return visits

    return (Tree,)


@app.cell(hide_code=True)
def _():
    # mo.md(r"""
    # How does the Monte Carlo Tree Search AI think?
    # """)
    return


@app.cell
def _():
    # mo.vstack(
    #    [
    ##        mo.md(
    #            "This section will walk you through how Monte Carlo Tree Search (MCTS) thinks about the game and chooses an action to play."
    ##        ),
    #        mo.md(
    #            "- You can create a custom board state to think about (or just stick with the default empty starting board). \n - MCTS will create a tree to 'think' about this state. The chosen state is set as the root of the tree, and random rollout simulations are used to gain knowledge about good/bad actions (see previous video/notebook for information on what a random rollout is!) \n - The board state and MCTS choices for rollouts are shown in the bar chart below. This starts with 0 rollouts and increases as the AI 'thinks'. \n - The full tree is visible in the tree explorer below. There are several buttons to show you how MCTS grows the tree. These walk you through the steps that are used to expand the tree and choose which actions to rollout. \n - In the next section, you can try playing against a MCTS AI that chooses its moves according to this algorithm."
    #        ),
    #    ]
    # )
    return


@app.cell
def _(Game, N_ACTIONS, N_COLS, N_ROWS, Tree, ai_think_slider):
    # 2. State triggers
    get_mcts_tick, set_mcts_tick = mo.state(0)
    get_original_node, set_original_node = mo.state(0)  # Tracks Step 1
    get_expanded_node, set_expanded_node = mo.state(0)  # Tracks Step 2b
    if "tutorial_board_start_position" not in globals():
        tutorial_board_start_position = jnp.zeros((N_ROWS, N_COLS), dtype=int)


    def load_manual_board(value):
        """Triggered when the user submits the manual matrix form."""
        global tutorial_board_start_position

        # Convert the 2D list from the UI matrix directly into a JAX array
        tutorial_board_start_position = jnp.array(value, dtype=int)
        my_reset_tree(None)

        # Optional: You could trigger a marimo UI toast here to let the user know it was staged!
        # mo.status.toast("Manual board staged! Click Reset Sandbox Tree to load it.")


    def load_random_board(value):
        """Triggered when the user submits the random slider form."""
        global tutorial_board_start_position

        # 1. Start with a blank board
        empty_board = jnp.zeros((N_ROWS, N_COLS), dtype=int)

        # 2. Create a fresh random key
        rng_key = jax.random.PRNGKey(int(time.time() * 1000))

        # 3. Use your rollout function to play 'value' number of moves
        # We unpack [0] because rollout returns (final_board, final_key)
        # Try up to 10 times to find a board that isn't already game-over
        for _ in range(10):  # try 10 times then give up
            random_board, rng_key = Game.random_rollout(
                board=empty_board,
                player_turn=1,
                key=rng_key,
                max_rollout_steps=value,
            )

            # If the game is NOT over, we found a good board! Break the loop.
            if not bool(jnp.any(Game.is_terminal(random_board))):
                break

        # 4. Update the state
        tutorial_board_start_position = random_board
        my_reset_tree(None)


    # 4. Helper to trace a path back to the root
    def set_highlight_path(target_ix):
        path = [target_ix]
        temp = target_ix
        while temp != 0:
            temp = int(tutorial_tree.parent_ix[temp])
            path.append(temp)
        tutorial_tree.highlight_path = path


    # 5. Button Callbacks
    def my_manual_select(_):
        val = target_node_input.value
        # Reset both to the manually selected node
        set_original_node(val)
        set_expanded_node(val)
        set_highlight_path(val)
        set_mcts_tick(lambda v: v + 1)


    def my_auto_select(_):
        global tutorial_tree
        ix = int(tutorial_tree.select_leaf())

        # Reset both to the auto-selected node
        set_original_node(ix)
        set_expanded_node(ix)
        set_highlight_path(ix)
        tutorial_tree.clear_sim()
        set_mcts_tick(lambda v: v + 1)


    def my_expand_node(_):
        orig_ix = get_original_node()

        # 1. Expand the original leaf
        tutorial_tree.expand_node(orig_ix)

        # Now move to child

        # 2. Step 2b Logic: Step down to a random valid child

        child_ixs = tutorial_tree.children_ixs[orig_ix]
        valid_mask = np.logical_and(
            child_ixs != tutorial_tree.ILLEGAL_CHILD_FLAG, child_ixs != 0
        )

        if np.any(valid_mask):
            valid_children = child_ixs[valid_mask]
            child_ix = int(
                np.random.choice(valid_children)
            )  # choose a random child
        else:
            child_ix = (
                orig_ix  # Fallback just in case it was a terminal game-over node
            )

        # 3. Update state and visually step the highlight down!
        set_expanded_node(child_ix)
        set_highlight_path(child_ix)
        set_mcts_tick(lambda v: v + 1)


    def my_simulate_node(_):
        # CRITICAL: We simulate from the EXPANDED child, not the original parent!
        ix = get_expanded_node()
        tutorial_tree.simulate_node(ix)

        # Re-apply the highlight path because simulate_node wiped it
        set_highlight_path(ix)
        set_mcts_tick(lambda v: v + 1)


    def my_backup(_):
        tutorial_tree.backup()
        set_mcts_tick(lambda v: v + 1)


    def my_full_mcts_step(n_times):
        def _callback(_):
            global tutorial_tree
            for _i in range(n_times):
                my_auto_select(_)
                orig_ix = get_original_node()
                if tutorial_tree.N_sims[orig_ix] > 0:
                    my_expand_node(_)
                else:
                    set_expanded_node(orig_ix)
                my_simulate_node(_)
                my_backup(_)
                set_mcts_tick(lambda v: v + 1)

        return _callback


    def my_reset_tree(_):
        global tutorial_tree
        _tutorial_root = tutorial_board_start_position  #
        tutorial_tree = Tree(root_board=_tutorial_root, root_player_turn=1)
        tutorial_tree.expand_node(0)
        for ix in range(1, N_ACTIONS + 1):  # simulate all the children once!
            tutorial_tree.simulate_node(ix)
            tutorial_tree.backup()

        tutorial_tree.simulation_ix = None  # turn off the simulation
        set_original_node(0)
        set_expanded_node(0)
        tutorial_tree.highlight_path = []

        set_mcts_tick(lambda v: v + 1)


    if "tutorial_tree" not in globals():
        _tutorial_root = jnp.zeros((N_ROWS, N_COLS), dtype=int)
        tutorial_tree = Tree(root_board=_tutorial_root, root_player_turn=1)
        my_reset_tree(None)

    # tutorial_tree.expand_node(0)  # Start with root expanded
    #    for ix in range(1, N_ACTIONS+1): # simulate all the children once!
    #        tutorial_tree.simulate_node(ix)
    #        tutorial_tree.backup()

    #    set_original_node(0)
    #    set_expanded_node(0)
    #    tutorial_tree.highlight_path = []
    #    set_mcts_tick(lambda v: v + 1)

    # Starting board inputs
    start_board_input = mo.ui.matrix(
        min_value=0,
        max_value=2,
        step=1,
        debounce=True,
        value=[[0] * N_COLS] * N_ROWS,
        label="Input a custom position to examine (*1 = Player 1 piece, 2=Player 2 piece, 0=empty*)",
    ).form(show_clear_button=True, on_change=load_manual_board)

    slider_random_position = mo.ui.slider(
        label="Create a random position from how many random moves?",
        start=0,
        stop=N_ROWS * N_COLS,
        value=0,
        show_value=True,
    ).form(on_change=load_random_board, clear_on_submit=True)


    # 3. UI Inputs


    # THE MANUAL SELECT IS NOT CURRENTLY USED!
    target_node_input = mo.ui.number(
        value=0, step=1, label="🎯 Manual Select Node #"
    ).form(on_change=my_manual_select)

    show_ucb_tutorial = mo.ui.checkbox(label="Show UCB Scores", value=False)
    hide_unvisited_tutorial = mo.ui.checkbox(
        label="Hide Unvisited Nodes", value=False
    )
    display_board_tutorial = mo.ui.checkbox(
        label="Display Board State in Tree", value=True
    )
    minimalist_tutorial = mo.ui.checkbox(label="Minimalist Nodes", value=False)
    # 2. New Checkbox for Value Chart
    show_value_tutorial_checkbox = mo.ui.checkbox(
        label="Show average win values (*P1 wins=+1, P2 wins=-1, Tie=0*)",
        value=False,
    )

    # 6. Buttons
    btn_auto_select = mo.ui.button(
        label='1\. 🤖 <div data-tooltip="Starting from the root, select a child using UCB at each level until you reach a leaf.">Select</div> leaf by UCB Algo at each level',
        on_click=my_auto_select,
    )
    btn_expand = mo.ui.button(
        label='2\. 🌱 <div data-tooltip="Add new nodes to the tree for all the possible moves that could be played from here. \n Then move down to a random newly created leaf.">Expand</div> leaf & move down',
        on_click=my_expand_node,
    )
    btn_simulate = mo.ui.button(
        label='3\. 🎲 <div data-tooltip="Play random moves from this leaf state until game ends.">Simulate</div> rollout from leaf',
        on_click=my_simulate_node,
    )
    btn_backup = mo.ui.button(
        label='4\. ⬆️ <div data-tooltip="Save the results of the rollout (win/loss/draw) at this leaf node *and* to all its ancestors!">Backup</div> results up tree',
        on_click=my_backup,
    )

    max_nodes_tutorial_slider = mo.ui.slider(
        start=0,
        stop=ai_think_slider.value,
        step=10,
        value=200,
        debounce=True,
        show_value=True,
        label="🌳 Max Nodes:",
    )


    # 8. Button & UI Setup
    N_times_ui = mo.ui.number(
        start=1,
        stop=1000,
        label='<div data-tooltip="Number of times to run MCTS algo">Num times</div>',
    )

    btn_reset_tree = mo.ui.button(label="🔄 Reset", on_click=my_reset_tree)
    return (
        N_times_ui,
        btn_auto_select,
        btn_backup,
        btn_expand,
        btn_reset_tree,
        btn_simulate,
        get_expanded_node,
        get_mcts_tick,
        get_original_node,
        my_full_mcts_step,
        slider_random_position,
        start_board_input,
        tutorial_tree,
    )


@app.cell
def _(N_times_ui, my_full_mcts_step):
    btn_full_step = mo.ui.button(
        label=f'⏭️ Run Full MCTS Step (<div data-tooltip="To avoid having too many unvisited leaves, we actually *skip* the expand step for leaves with N=0 rollouts, and just go straight to simulate.">all*</div> 4 Steps!) ×{N_times_ui.value}',
        on_click=my_full_mcts_step(N_times_ui.value),
    )
    return (btn_full_step,)


@app.cell
def _(create_tree_ui):
    # ---------------------------------------------------------
    # UNPACK FOR TREE 1
    # ---------------------------------------------------------
    (
        tut_start_node_input,
        tut_max_nodes_slider,
        tut_max_children_slider,
        tut_zoom_slider,
        tut_show_ucb_checkbox,
        tut_show_board_checkbox,
        tut_minimalist_nodes_checkbox,
        tut_hide_unvisited_checkbox,
        tut_tree_controls,
    ) = create_tree_ui()
    return (
        tut_hide_unvisited_checkbox,
        tut_max_children_slider,
        tut_max_nodes_slider,
        tut_minimalist_nodes_checkbox,
        tut_show_board_checkbox,
        tut_show_ucb_checkbox,
        tut_start_node_input,
        tut_tree_controls,
        tut_zoom_slider,
    )


@app.cell
def _(
    GLOBAL_BACKGROUND_COLOR,
    MCTSMermaidVisualizer,
    N_COLS,
    N_ROWS,
    N_times_ui,
    NumpyTreeAdapter,
    btn_auto_select,
    btn_backup,
    btn_expand,
    btn_full_step,
    btn_reset_tree,
    btn_simulate,
    get_expanded_node,
    get_mcts_tick,
    get_original_node,
    interactive_mermaid,
    modify_game_rules,
    slider_random_position,
    start_board_input,
    tut_hide_unvisited_checkbox,
    tut_max_children_slider,
    tut_max_nodes_slider,
    tut_minimalist_nodes_checkbox,
    tut_show_board_checkbox,
    tut_show_ucb_checkbox,
    tut_start_node_input,
    tut_tree_controls,
    tut_zoom_slider,
    tutorial_tree,
):
    _tick = get_mcts_tick()
    _original_node = get_original_node()
    _expanded_node = get_expanded_node()

    _total_sims = (
        int(tutorial_tree.N_sims[0]) if tutorial_tree.N_sims[0] > 0 else 0
    )

    # 1. Wrap the tutorial tree in the Adapter
    _tut_adapter = NumpyTreeAdapter(
        tutorial_tree,
        display_ucb=tut_show_ucb_checkbox.value,
        enable_highlighting=True,
        hide_unvisited=tut_hide_unvisited_checkbox.value,
        display_board=tut_show_board_checkbox.value,
        minimal_display=tut_minimalist_nodes_checkbox.value,
    )

    # 2. Instantiate the Visualizer
    _tut_visualizer = MCTSMermaidVisualizer(_tut_adapter)

    # 3. Generate the HTML string and render it
    _tut_html = interactive_mermaid(
        _tut_visualizer.to_mermaid(
            start_ix=tut_start_node_input.value,
            max_total_node_weight=tut_max_nodes_slider.value,
            max_total_children_weight=tut_max_children_slider.value,
        ),
        initial_zoom=tut_zoom_slider.value,
    )

    #### Display of board and bar chart
    # ==========================================================
    # A. RENDER THE ROOT BOARD
    # ==========================================================
    _SB_CELL_SIZE = "3.0rem"
    _SB_FONT_SIZE = "2.5rem"


    def _sb_static_cell(_display_char, action_idx=None, opacity=1.0):
        _action_html = ""
        if action_idx is not None:
            _action_html = f"<div style='position: absolute; top: 0.1rem; left: 50%; transform: translateX(-50%); font-size: 0.7rem; font-family: monospace; font-weight: bold; color: #888; pointer-events: none; line-height: 1;'>A{action_idx}</div>"

        # Apply opacity to the whole cell container
        return mo.Html(
            f"<div style='width: {_SB_CELL_SIZE}; height: {_SB_CELL_SIZE}; position: relative; display: grid; place-items: center; font-size: {_SB_FONT_SIZE}; line-height: 1; opacity: {opacity};'>"
            f"{_display_char}{_action_html}"
            f"</div>"
        )


    # Fetch the boards for comparison
    _root_board = tutorial_tree.board[0]
    _node_board = tutorial_tree.board[_expanded_node]
    _sim_board = (
        tutorial_tree.simulation_board
        if tutorial_tree.simulation_ix is not None
        else tutorial_tree.board[_expanded_node]
    )

    _sb_rows = []
    # Cache the GRAVITY check outside the loop for speed
    _has_gravity = globals().get("GRAVITY", False)

    for _r in range(N_ROWS):
        _row = []
        for _c in range(N_COLS):
            # 1. Determine which board state defines this cell's presence
            if _sim_board[_r, _c] != 0 and _node_board[_r, _c] == 0:
                _opacity = 0.25
                _val = _sim_board[_r, _c]
                _display_char = "🔴" if _val == 1 else "🔵"
            elif _node_board[_r, _c] != 0 and _root_board[_r, _c] == 0:
                _opacity = 1.0
                _val = _node_board[_r, _c]
                _display_char = "⭕" if _val == 1 else "Ⓜ️"
            else:
                _val, _opacity = int(_root_board[_r, _c]), 1.0
                if _val == 0:
                    _display_char = "⬜"
                else:
                    _display_char = "🔴" if _val == 1 else "🔵"

            # Calculate the correct action index based on your game type
            if _has_gravity:
                # Connect 4: Only label the top row!
                _action_idx = _c if _r == 0 else None
            else:
                # Tic-Tac-Toe: Label every cell
                _action_idx = _r * N_COLS + _c

            # 3. Create cell
            _row.append(
                _sb_static_cell(_display_char, _action_idx, opacity=_opacity)
            )

        _sb_rows.append(mo.hstack(_row, gap=0, justify="center"))

    sandbox_board_ui = mo.vstack(_sb_rows, gap=0)


    # ==========================================================
    # B. RENDER THE BAR CHART
    # ==========================================================
    _fig, _ax_top = plt.subplots(
        figsize=(10, 7), facecolor=GLOBAL_BACKGROUND_COLOR
    )
    _ax_top.set_facecolor(GLOBAL_BACKGROUND_COLOR)
    _ax_top.tick_params(axis="both", labelsize=18)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(
            facecolor="#d62728",
            alpha=0.9,
            label="Number of simulations",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            label="0.xy = Average score \n (P1=+1, P2=-1, Tie=0)",
        ),
    ]

    leg = _ax_top.legend(
        handles=legend_elements,
        fontsize=14,
        loc="upper right",
        handlelength=1.5,  # keeps spacing consistent
        handletextpad=0.8,
    )

    leg.legend_handles[1].set_visible(False)

    # Hide the second legend handle
    # leg.legend_handles[1].set_visible(False)

    # Get visits and valid children
    _visits = tutorial_tree.get_child_visits(0)
    _child_ixs = tutorial_tree.children_ixs[0]

    # 1. Apply the clever bar_offset trick!
    _bar_offset = 0.05
    _plot_visits = _visits + _bar_offset

    _max_v = max(_visits) if len(_visits) > 0 else 0
    _y_lim_top = max(1, _max_v) * 1.2 + _bar_offset

    _x = np.arange(len(_visits))
    _bars = _ax_top.bar(_x, _plot_visits, color="#d62728", alpha=0.9)

    # 2. Draw the Average Win Value on top of the bars
    if True:  # show_value_tutorial_checkbox.value:
        for _idx, _bar in enumerate(_bars):
            _c_ix = _child_ixs[_idx]

            # Only draw text for legal moves
            if _c_ix != tutorial_tree.ILLEGAL_CHILD_FLAG:
                _n = tutorial_tree.N_sims[_c_ix]

                # Show the calculated average, or "nan" if unvisited
                if _n > 0:
                    _val = tutorial_tree.total_sim_reward[_c_ix] / _n
                    _val_str = f"{_val:.2f}"
                else:
                    _val_str = "nan"

                # Restored your working text logic!
                _ax_top.text(
                    _bar.get_x() + _bar.get_width() / 2,
                    _bar.get_height()
                    + (_y_lim_top * 0.02),  # Hover slightly above the bar
                    _val_str,
                    ha="center",
                    va="bottom",
                    fontsize=20,
                    fontweight="bold",
                )

    _ax_top.set_ylim(0, _y_lim_top)
    _ax_top.set_ylabel("Number of Rollout Simulations Run", fontsize=20)
    _ax_top.set_xlabel("Action Number", fontsize=20)
    _ax_top.set_title("Which actions did we try?", fontweight="bold", fontsize=24)
    _ax_top.set_xticks(_x)

    # 3. Shift the Y-Axis Ticks
    # Force standard integer ticks (0, 1, 2, 3...)
    _ax_top.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    _fig.canvas.draw()  # Force matplotlib to generate the ticks

    # Grab the integer ticks, shift their physical placement up by the offset,
    # and explicitly label them with the original integer!
    _ticks = _ax_top.get_yticks()
    _ax_top.set_yticks(_ticks + _bar_offset)
    _ax_top.set_yticklabels([f"{int(t)}" for t in _ticks])

    _fig.tight_layout()

    sandbox_chart_ui = mo.as_html(_fig)
    # plt.close(_fig)

    # 4. Structured Layout
    mcts_learning_display = mo.vstack(
        [
            mo.accordion(
                {
                    "**Create a custom board state to think about**": mo.hstack(
                        [
                            slider_random_position,
                            mo.md("**OR**"),
                            start_board_input,
                        ]
                    ),
                }
            ),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.hstack(
                                [
                                    mo.md("**Board State MCTS is looking at**"),
                                ]
                            ),
                            sandbox_board_ui,
                            mo.hstack(
                                [
                                    modify_game_rules,
                                    mo.accordion(
                                        {
                                            "Emoji Legend": mo.vstack(
                                                [
                                                    mo.md(
                                                        "**Original State**: P1=🔴, P2=🔵"
                                                    ),
                                                    mo.md(
                                                        "**Leaf State**: P1=⭕, P2=Ⓜ️"
                                                    ),
                                                    mo.md(
                                                        "**Random Rollout**: P1=<span style='opacity: 0.5;'>🔴</span> P2=<span style='opacity: 0.5;'>🔵</span> "
                                                    ),
                                                ],
                                                justify="center",
                                            )
                                        }
                                    ),
                                ]
                            ),
                        ],
                        align="center",
                    ),
                    mo.vstack(
                        [sandbox_chart_ui]
                    ),  # mo.vstack([show_value_tutorial_checkbox, ]),
                ],
                justify="space-around",
                align="stretch",
                widths="equal",
            ),
            # mo.md("### 🧪 MCTS Step-by-Step Sandbox"),
            # mo.md("---"),
            # Step 1
            mo.vstack(
                [
                    mo.hstack(
                        [
                            mo.md(
                                f"📊 **Total Simulations Run:** {_total_sims} **Current Selection:** Node \#{_expanded_node}"
                            ),
                            btn_full_step,
                            N_times_ui,
                            btn_reset_tree,
                        ],
                        justify="space-between",
                        align="center",
                    ),
                    mo.hstack(
                        [
                            mo.md("**4 Steps of MCTS:**"),
                            btn_auto_select,
                            btn_expand,
                            btn_simulate,
                            btn_backup,
                        ],
                        justify="start",
                    ),
                    mo.vstack(
                        [
                            _tut_html,
                            tut_tree_controls,
                        ]
                    ),
                    # mo.ui.tabs(
                    ##    {
                    #        "Bob says": mo.md("Hello, Alice!"),
                    #        "Alice says": mo.md("Hello, Bob!"),
                    #    }
                    # ),
                    # mo.hstack(
                    #    [
                    #        mo.md(
                    #            f"**Step 1.**"
                    #        ),  # Select a leaf node to explore:
                    #        btn_auto_select,
                    # mo.md("**OR**"),
                    # target_node_input,
                    #        mo.md(f"▶️ Selected: **Node #{_original_node}**"),
                    #    ],
                    # gap=4,
                    #    justify="start",
                    ##    align="center",
                    # ),
                    # Step 2a
                    # mo.hstack(
                    ##    [
                    #        mo.md("**Step 2.**"),
                    #        btn_expand,
                    #        mo.md("*(Adds new children to the tree)*"),
                    #    ],
                    #    justify="start",
                    # ),
                    # Step 2b (Visual Confirmation of the jump)
                    # mo.hstack(
                    #    [
                    #        mo.md(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳**Step 2b.**"),
                    #        btn_move_to_child,
                    #        mo.md(
                    #            f"Original: **Node #{_original_node}** ➡️ After: **Node #{_expanded_node}**"
                    #        ),
                    #    ],
                    #    justify="start",
                    # ),
                    # Step 3
                    # mo.hstack(
                    #    [
                    #        mo.md("**Step 3.**"),
                    #        btn_simulate,
                    #        mo.md("*(Plays a random rollout game to the end)*"),
                    #    ],
                    #    justify="start",
                    # ),
                    # Step 4
                    ##mo.hstack(
                    #    [
                    #        mo.md("**Step 4.**"),
                    #        btn_backup,
                    #        mo.md("*(Passes the win/loss record up the tree)*"),
                    #    ],
                    #    justify="start",
                    # ),
                ],
                justify="start",
            ),
        ],
        gap=2,
    )

    # display it!
    # mcts_learning_display
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Play Connect 4 or other k-in-a-line games against the AI
    """)
    return


@app.cell
def _(
    Game,
    N_COLS,
    N_ROWS,
    random_seed_checkbox,
    random_seed_num,
    random_start_turn_slider,
):
    def seed_int_to_actions(seed_int):
        # 2. Use pure Python math to extract the actions
        actions_list = []
        temp_seed = seed_int

        # Handle the edge case if the user inputs 0
        if temp_seed == 0:
            actions_list = [-1]
        else:
            while temp_seed > 0:
                digit = temp_seed % 10
                # Insert at index 0 to keep the left-to-right order (e.g. 112 -> [0, 0, 1])
                actions_list.insert(0, digit - 1)
                temp_seed = temp_seed // 10

        # 3. Convert to a JAX array right before the JAX scanning loop needs it
        actions = jnp.array(actions_list, dtype=jnp.int32)
        return actions


    class myGameClass:  # class for playing against the AI
        def __init__(self):
            seed_int = None
            if random_seed_checkbox.value == True:
                seed_int = random_seed_num.value
            elif random_start_turn_slider.value > 0:
                _k = random_start_turn_slider.value
                _key = jax.random.PRNGKey(int(time.time()))
                actions = jax.random.randint(
                    _key, shape=(_k,), minval=0, maxval=N_COLS
                )
                # Create an array counting down to 0, e.g., if _k=3: [2, 1, 0]
                exponents = jnp.arange(_k - 1, -1, -1)

                # Raise 10 to those powers, and multiply by our 1-indexed actions
                # Using int32 to ensure it stays a clean integer
                powers_of_10 = jnp.power(10, exponents).astype(jnp.int32)
                seed_int = jnp.sum((actions + 1) * powers_of_10)

            self.seed_int = seed_int
            self.board = jnp.zeros((N_ROWS, N_COLS), dtype=jnp.int32)

            if seed_int is not None and seed_int > 0:
                _len = int(np.log10(seed_int)) + 1
                player = (_len % 2) + 1
                for _i in range(_len):
                    digit = seed_int % 10
                    # Insert at index 0 to keep the left-to-right order (e.g. 112 -> [0, 0, 1])
                    action = digit - 1
                    action = max(action, 0)  # ensure action is at least 0
                    seed_int = seed_int // 10
                    self.board = Game.step(self.board, action, player)
                    player = 3 - player

            self.tree = None

            # Time Travel & Analytics History for the graphs shown
            self.history = [self.board.copy()]
            self.player_turn_history = []
            self.mcts_policy_history = []
            self.mcts_value_history = []
            self.mcts_child_values_history = []

            # Track the actual Tree objects over time for the tree browser
            self.tree_history = [None]


    MyGame = myGameClass()
    return (MyGame,)


@app.cell
def _(GRAVITY, Game, MyGame, N_COLS, N_ROWS):
    # State bindings
    get_phase, set_phase = mo.state("human")
    get_turn, set_turn = mo.state(0)  # Tracks the currently viewed turn
    get_game_mode, set_game_mode = mo.state(
        "Human"
    )  # whether or not human is playing, can be Human or AI


    def human_play(action):
        current_time = get_turn()

        # ⏱️ TIME TRAVEL CHECK: If playing in the past, delete the future
        if current_time < len(MyGame.history) - 1:
            MyGame.history = MyGame.history[: current_time + 1]
            MyGame.player_turn_history = MyGame.player_turn_history[:current_time]
            MyGame.mcts_policy_history = MyGame.mcts_policy_history[:current_time]
            MyGame.mcts_value_history = MyGame.mcts_value_history[:current_time]
            # MyGame.nn_logits_history = MyGame.nn_logits_history[:current_time]
            MyGame.mcts_child_values_history = MyGame.mcts_child_values_history[
                :current_time
            ]
            MyGame.tree_history = MyGame.tree_history[: current_time + 1]

            MyGame.board = MyGame.history[-1].copy()

        # ---> THE FIX: Determine whose turn it is dynamically! <---
        _current_player = 1 if len(MyGame.history) % 2 == 1 else 2

        # Apply human move using the dynamic player ID
        MyGame.board = Game.step(
            MyGame.board, jnp.array(action), fill_value=_current_player
        )

        # Update History
        MyGame.history.append(MyGame.board.copy())
        MyGame.player_turn_history.append(
            _current_player
        )  # <--- Record the correct ID here too!

        # Pad stats with zeros for the human turn
        MyGame.mcts_policy_history.append(jnp.zeros(N_COLS))
        MyGame.mcts_value_history.append(0.0)
        # MyGame.nn_logits_history.append(jnp.zeros(N_COLS))
        MyGame.mcts_child_values_history.append(jnp.zeros(N_COLS))
        MyGame.tree_history.append(None)

        # Snap the slider to the new present
        set_turn(len(MyGame.history) - 1)

        if jnp.any(Game.is_terminal(MyGame.board)):
            set_phase("game_over")
        else:
            set_phase("ai_display")


    def reset_game(_):
        MyGame.__init__()
        set_turn(0)
        # Store the chosen mode in our state
        mode = start_player_dropdown.value
        set_game_mode(mode)

        if mode == "Human":
            set_phase("human")
        else:
            # Both "AI" and "AI vs AI" start with the AI thinking!
            set_phase("ai_display")


    # ==========================================================
    # UI COMPONENTS
    # ==========================================================
    # (Keep your action_buttons and turn_slider definitions here)


    # 1. New Dropdown for Starting Player
    start_player_dropdown = mo.ui.dropdown(
        options=["Human", "AI", "AI vs AI"],
        value="Human",
        label="Player 1 is (Applies on Reset)",
    )

    # 2. New Checkbox for Value Chart
    show_value_checkbox = mo.ui.checkbox(
        label="Show average win values (*P1 wins=+1, P2 wins=-1, Tie=0*)",
        value=True,
    )


    reset_button = mo.ui.button(label="🔄 Reset Game", on_click=reset_game)

    action_buttons = {}

    if GRAVITY:
        for _c in range(N_COLS):
            action_buttons[_c] = mo.ui.button(
                label="⬇️", on_click=lambda _, a=_c: human_play(a)
            )
    else:
        for _r in range(N_ROWS):
            for _c in range(N_COLS):
                _action = _r * N_COLS + _c
                action_buttons[_action] = mo.ui.button(
                    label="⬜", on_click=lambda _, a=_action: human_play(a)
                )

    ai_think_slider = mo.ui.slider(
        start=100,
        stop=10000,
        step=100,
        value=500,
        debounce=True,
        show_value=True,
        label="🤖 AI Thinking Budget (Rollouts)",
    )
    return (
        action_buttons,
        ai_think_slider,
        get_game_mode,
        get_phase,
        get_turn,
        reset_button,
        set_phase,
        set_turn,
        show_value_checkbox,
        start_player_dropdown,
    )


@app.cell
def _(
    GLOBAL_BACKGROUND_COLOR,
    GRAVITY,
    Game,
    MyGame,
    N_COLS,
    N_ROWS,
    action_buttons,
    ai_think_slider,
    get_phase,
    get_turn,
    modify_game_rules,
    modify_game_start,
    reset_button,
    set_phase,
    show_value_checkbox,
    start_player_dropdown,
    turn_slider,
):
    CELL_SIZE = "3.5rem"
    FONT_SIZE = "2.5rem"

    current_time = get_turn()
    current_phase = get_phase()
    current_board = MyGame.history[current_time]
    _is_game_over = bool(jnp.any(Game.is_terminal(current_board)))

    is_present = current_time == (len(MyGame.history) - 1)
    can_play = (current_phase == "human" and is_present) or not is_present

    if current_time > 0:
        previous_board = MyGame.history[current_time - 1]
        just_played_mask = current_board != previous_board
    else:
        just_played_mask = jnp.zeros_like(current_board, dtype=bool)


    def get_emoji(val, is_just_played):
        if val == 0:
            return "⬜"
        is_p1 = val == 1
        if is_just_played:
            return "⭕" if is_p1 else "Ⓜ️"
        return "🔴" if is_p1 else "🔵"


    def static_cell(val, is_just_played):
        display_char = get_emoji(val, is_just_played)
        return mo.Html(
            f"<div style='width: {CELL_SIZE}; height: {CELL_SIZE}; display: grid; place-items: center; font-size: {FONT_SIZE}; line-height: 1;'>{display_char}</div>"
        )


    # 3. Label update: Using absolute positioning so it hovers slightly above the emoji
    # 1. Update the active_cell function
    def active_cell(btn, action_idx):
        # pointer-events: none ensures the text doesn't block you from clicking the button!
        return mo.Html(
            f"<div class='active-btn-wrapper' style='width: {CELL_SIZE}; height: {CELL_SIZE}; position: relative; display: grid; place-items: center;'>"
            f"{btn}"
            f"<div style='position: absolute; top: 0.1rem; left: 50%; transform: translateX(-50%); font-size: 0.85rem; font-family: monospace; font-weight: bold; color: #555; pointer-events: none; line-height: 1;'>A{action_idx}</div>"
            f"</div>"
        )


    def empty_placeholder():
        return mo.Html(
            f"<div style='width: {CELL_SIZE}; height: {CELL_SIZE};'></div>"
        )


    # 2. Build the Charts
    if show_value_checkbox.value:
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=(6, 8),
            gridspec_kw={"height_ratios": [0.75, 0.5]},
            facecolor=GLOBAL_BACKGROUND_COLOR,
        )
        ax_top.set_facecolor(GLOBAL_BACKGROUND_COLOR)
        ax_bot.set_facecolor(GLOBAL_BACKGROUND_COLOR)
    else:
        fig, ax_top = plt.subplots(
            1, 1, figsize=(6, 4), facecolor=GLOBAL_BACKGROUND_COLOR
        )
        ax_bot = None
        ax_top.set_facecolor(GLOBAL_BACKGROUND_COLOR)

    # Replace the old 'if' statement with this robust check:
    # We know it was an AI turn if the policy history isn't just padded zeros
    has_ai_data = (
        current_time > 0
        and np.sum(MyGame.mcts_policy_history[current_time - 1]) > 0
    )

    GRAPH_COLOR_P1 = "#d62728"
    GRAPH_COLOR_P2 = "#1f77b4"


    if has_ai_data:
        # We can color code the bars based on which AI is playing!
        is_p1 = MyGame.player_turn_history[current_time - 1] == 1
        bar_color = GRAPH_COLOR_P1 if is_p1 else GRAPH_COLOR_P2

        idx = current_time - 1
        visits = MyGame.mcts_policy_history[idx]
        x = jnp.arange(len(visits))
        width = 0.35

        # Catch the bars in a variable so we can iterate over them
        bars = ax_top.bar(
            x,
            visits,
            2 * width,
            color=bar_color,
            label="Simulation Count",
            alpha=1.0,
        )

        # ADD THIS: Add value text on top of the bars if checkbox is active
        if show_value_checkbox.value:
            child_values = MyGame.mcts_child_values_history[idx]
            for bar, val in zip(bars, child_values):
                if (
                    bar.get_height() > 0
                ):  # Only draw text if the move was actually explored
                    ax_top.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height()
                        + (max(visits) * 0.02),  # Hover slightly above the bar
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )
        # ---> ADD THESE TWO LINES <---
        # Give the y-axis 15% headroom above the highest bar so the text fits perfectly
        max_visits = max(visits) if len(visits) > 0 else 1
        ax_top.set_ylim(0, max_visits * 1.15)

        ax_top.set_ylabel("Number of Simulations", fontsize=10)
        ax_top.set_xlabel("Action Number", fontsize=10)

        ax_top.set_title(f"Turn {current_time}: AI Evaluation", fontweight="bold")
        ax_top.set_xticks(x)
        ax_top.legend()
    else:
        ax_top.text(
            0.5,
            0.5,
            "No AI Data for this turn"
            if current_time > 0
            else "Move slider or play move to see graph...",
            ha="center",
            va="center",
        )
        ax_top.set_axis_off()

    # 3. Update the Bottom Chart logic
    # Bottom Chart (Only if ax_bot exists)
    if ax_bot is not None:
        if current_time > 0:
            x_turns = jnp.arange(1, current_time + 1)
            y_vals = jnp.array(MyGame.mcts_value_history[:current_time])

            # ---> ADD THIS: Create a mask to filter out human turns! <---
            # We know an AI actually thought on a turn if it generated policy probabilities
            ai_mask = np.array(
                [np.sum(p) > 0 for p in MyGame.mcts_policy_history[:current_time]]
            )

            # Only draw the chart if there is at least one AI evaluation to show
            if np.any(ai_mask):
                x_ai = x_turns[ai_mask]
                y_ai = y_vals[ai_mask]

                ax_bot.plot(
                    x_ai,
                    y_ai,
                    color="black",
                    marker="o",
                    markersize=3,
                    linewidth=2,
                )
                ax_bot.axhline(0, color="gray", linestyle="--", linewidth=1)
                ax_bot.fill_between(
                    x_ai,
                    y_ai,
                    0,
                    where=(y_ai > 0),
                    color=GRAPH_COLOR_P1,
                    alpha=0.5,
                )
                ax_bot.fill_between(
                    x_ai,
                    y_ai,
                    0,
                    where=(y_ai <= 0),
                    color=GRAPH_COLOR_P2,
                    alpha=0.5,
                )

                # Force the x-axis to span the whole timeline so the dots align with the top chart
                ax_bot.set_xlim(1, max(2, current_time))
                ax_bot.set_ylim(-1.1, 1.1)

                ax_bot.set_xlabel("Turn Number", fontsize=10, fontweight="bold")
                ax_bot.set_ylabel(
                    "Root Node Value", fontsize=10, fontweight="bold"
                )
            else:
                ax_bot.text(
                    0.5, 0.5, "No AI evaluations yet...", ha="center", va="center"
                )
                ax_bot.set_axis_off()
        else:
            ax_bot.set_axis_off()

    fig.tight_layout()
    html_fig = mo.as_html(fig)
    plt.close(fig)

    # 3. Build the Grid Natively (Passing action indices!)
    _rows_ui = []
    legal_actions = Game.legal_move_mask(current_board, player_turn=1)
    if GRAVITY:  # Assuming GRAVITY is defined elsewhere in your notebook
        _top_row = []
        for _c in range(N_COLS):
            if (
                can_play
                and current_board[0, _c] == 0
                and legal_actions[_c] == True
                and not _is_game_over
            ):
                _top_row.append(active_cell(action_buttons[_c], _c))
            else:
                _top_row.append(empty_placeholder())
        _rows_ui.append(mo.hstack(_top_row, gap=0, justify="center"))

        for _r in range(N_ROWS):
            _current_row = []
            for _c in range(N_COLS):
                val = current_board[_r, _c]
                is_jp = just_played_mask[_r, _c]
                _current_row.append(static_cell(val, is_jp))
            _rows_ui.append(mo.hstack(_current_row, gap=0, justify="center"))
    else:
        for _r in range(N_ROWS):
            _current_row = []
            for _c in range(N_COLS):
                val = current_board[_r, _c]
                _action = _r * N_COLS + _c
                is_jp = just_played_mask[_r, _c]
                if (
                    val == 0
                    and legal_actions[_action]
                    and can_play
                    and not _is_game_over
                ):
                    _current_row.append(
                        active_cell(action_buttons[_action], _action)
                    )
                else:
                    _current_row.append(static_cell(val, is_jp))
            _rows_ui.append(mo.hstack(_current_row, gap=0, justify="center"))

    board_grid = mo.vstack(_rows_ui, gap=0)

    # 4. Status Indicator
    if current_phase in ["ai_display", "ai_thinking"] and is_present:
        status_text = "### 🤖 Computer is thinking..."
        if current_phase == "ai_display":
            set_phase("ai_thinking")
    elif _is_game_over and is_present:
        game_reward = Game.reward(MyGame.board)
        winner = ""
        if game_reward == 1:
            winner = " Player 1 (🔴) wins!"
        elif game_reward == -1:
            winner = " Player 2 (🔵) wins!"
        elif game_reward == 0:
            winner = " Tie!"
        status_text = "### 🏆 Game Over!" + winner
    elif not is_present:
        status_text = (
            f"### 🕰️ Time Travel: Turn {current_time} (Play to overwrite future!)"
        )
    else:
        status_text = "### 👤 Your Turn!"

    status_header = mo.md(status_text)

    # 5. CSS Tweaks
    _css = mo.Html("""
    <style>
        .game-board-wrapper .active-btn-wrapper marimo-ui-element { width: 100% !important; height: 100% !important; display: grid !important; place-items: center !important; }
        .game-board-wrapper .active-btn-wrapper button { width: 100% !important; height: 100% !important; margin: 0 !important; padding: 0 !important; font-size: inherit !important; line-height: 1 !important; background: transparent !important; border: none !important; box-shadow: none !important; display: grid !important; place-items: center !important; text-align: center !important; cursor: pointer !important; transition: transform 0.1s ease-in-out !important; }
        .game-board-wrapper .active-btn-wrapper button:hover { transform: scale(1.15) !important; }
        .reset-zone button { aspect-ratio: auto !important; height: auto !important; width: auto !important; font-size: 1rem !important; padding: 0.5rem 1rem !important; display: flex !important; }
    </style>
    """)

    safe_board = mo.Html(f"<div class='game-board-wrapper'>{board_grid}</div>")
    protected_reset = mo.Html(f"<div class='reset-zone'>{reset_button}</div>")

    if MyGame.seed_int is None:
        helper_legend = mo.md(
            "Player 1: 🔴 &nbsp;|&nbsp; Player 2: 🔵 &nbsp;|&nbsp; Last Move: ⭕ / Ⓜ️"
        )
    else:
        helper_legend = mo.md(
            f"P1: 🔴 &nbsp;|&nbsp; P2: 🔵 &nbsp;|&nbsp; Last Move: ⭕ / Ⓜ️ &nbsp;|&nbsp; Start Seed {MyGame.seed_int}"
        )

    mo.vstack(
        [
            status_header,
            turn_slider,
            mo.hstack(
                [
                    mo.vstack(
                        [
                            safe_board,
                            helper_legend,
                            mo.hstack(
                                [
                                    protected_reset,
                                    start_player_dropdown,
                                ]
                            ),
                            ai_think_slider,
                            mo.hstack([modify_game_start, modify_game_rules]),
                            _css,
                        ],
                        gap=1,
                        align="center",
                    ),
                    mo.vstack([show_value_checkbox, html_fig]),
                ],
                gap=6,
                align="center",
            ),
        ],
        gap=1,
        align="center",
    )
    return


@app.cell
def _():
    return


@app.cell
def _(
    Game,
    MyGame,
    N_COLS,
    Tree,
    ai_think_slider,
    get_game_mode,
    get_phase,
    get_turn,
    set_phase,
    set_turn,
):
    _trigger = get_phase()

    if _trigger == "ai_thinking":
        # 1. Determine whose turn it is dynamically!
        # If history length is odd (1, 3, 5...), it's Player 1's turn. If even, Player 2.
        _current_player = 1 if len(MyGame.history) % 2 == 1 else 2

        # Initialize Tree with the correct player
        MyGame.tree = Tree(
            root_board=MyGame.board, root_player_turn=_current_player
        )

        # 2. Think!
        MyGame.tree.run_mcts(num_iterations=int(ai_think_slider.value), c=1.414)

        # 3. Grab best move & data
        _num_visits = MyGame.tree.get_child_visits()
        _best_action = int(np.argmax(_num_visits))

        _root_ix = MyGame.tree.ROOT_NODE_IX
        _total_reward = MyGame.tree.total_sim_reward[_root_ix]
        _total_visits = max(1, MyGame.tree.N_sims[_root_ix])
        _mcts_value = float(_total_reward / _total_visits)
        _nn_logits = jnp.zeros(N_COLS)

        # Extract values for the root's children
        _child_ixs = MyGame.tree.children_ixs[_root_ix]
        _valid_mask = _child_ixs != MyGame.tree.ILLEGAL_CHILD_FLAG
        _safe_child_ixs = np.where(_valid_mask, _child_ixs, 0)

        _child_rewards = MyGame.tree.total_sim_reward[_safe_child_ixs]
        _child_visits = MyGame.tree.N_sims[_safe_child_ixs]
        _safe_visits = np.where(_child_visits == 0, 1, _child_visits)

        _child_values = _child_rewards / _safe_visits
        _child_values = np.where(_valid_mask, _child_values, 0.0)

        # 4. Apply action using the CURRENT player
        MyGame.board = Game.step(
            MyGame.board, _best_action, fill_value=_current_player
        )

        MyGame.tree.simulation_ix = None  # disable the rollout display

        # 5. Append to History (Record the CURRENT player)
        MyGame.history.append(MyGame.board.copy())
        MyGame.player_turn_history.append(_current_player)
        MyGame.mcts_policy_history.append(_num_visits)
        MyGame.mcts_value_history.append(_mcts_value)
        # MyGame.nn_logits_history.append(_nn_logits)
        MyGame.mcts_child_values_history.append(_child_values)

        # record the entire tree
        MyGame.tree_history.append(MyGame.tree)

        # 6. Snap the slider to the new present
        set_turn(len(MyGame.history) - 1)

        # 7. Check Game Over or Loop Next Phase
        if jnp.any(Game.is_terminal(MyGame.board)):
            set_phase("game_over")
        else:
            # THE MAGIC LOOP: If AI vs AI, bounce back to display to trigger the next AI turn
            if get_game_mode() == "AI vs AI":
                set_phase("ai_display")
            else:
                set_phase("human")

    # the turn_slider has to be defined here so that after the AI plays it can be changed
    turn_slider = mo.ui.slider(
        start=-1 if get_game_mode() == "AI" else 0,
        stop=get_turn(),
        step=1
        if get_game_mode() == "AI vs AI"
        else 2,  # can only go back to human turns if a human is playing
        value=get_turn(),
        on_change=set_turn,
        label="🕰️ Game Timeline",
        show_value=True,
        full_width=True,
    )
    return (turn_slider,)


@app.cell
def _():
    show_tree_checkbox = mo.ui.checkbox(
        value=True, label="🌲 Show MCTS Tree Visualization"
    )


    def create_tree_ui():
        start_node = mo.ui.number(
            start=0,
            step=1,
            value=0,
            debounce=True,
            label='<div data-tooltip="Recenter the tree to start from this node #.">📍 Start Node#</div>',
        )

        max_nodes = mo.ui.slider(
            start=10,
            stop=1000,
            step=10,
            value=200,
            debounce=True,
            show_value=True,
            label='<div data-tooltip="Limits the maximum number of nodes drawn. Shows the most visited first.">🪓 Max Nodes</div>',
        )

        max_children = mo.ui.slider(
            start=5,
            stop=100,
            step=5,
            value=100,
            show_value=True,
            debounce=True,
            label='<div data-tooltip="Hides the least-visited children. Keeps the minimum number of top children required to capture this percentage of the rollouts.">✂️ Prune Children %</div>',
        )

        zoom = mo.ui.number(
            start=50,
            step=50,
            value=100,
            label='<div data-tooltip="Sets the initial starting zoom of the diagram.">🔎 Default Zoom%</div>',
        )

        show_ucb = mo.ui.checkbox(
            value=False,
            label='<div data-tooltip="Displays UCB score and optimism bonus.">📊 Show UCB Scores</div>',
        )

        show_board = mo.ui.checkbox(
            value=True,
            label='<div data-tooltip="Renders the current state of the game board using emojis.">🔲 Show board state</div>',
        )

        minimalist = mo.ui.checkbox(
            value=False,
            label='<div data-tooltip="Strips away Node# and extra text labels.">➖ Minimalist Text</div>',
        )

        hide_unvisited = mo.ui.checkbox(
            value=True,
            label='<div data-tooltip="Hides nodes that do not yet have at least 1 rollout.">🙈 Hide Unvisited</div>',
        )

        # Build the visual layout
        tree_controls_layout = mo.vstack(
            [
                mo.hstack(
                    [start_node, max_nodes, max_children, zoom],
                    justify="space-between",
                ),
                mo.hstack(
                    [
                        show_ucb,
                        show_board,
                        minimalist,
                        hide_unvisited,
                        mo.md(
                            '<div data-tooltip="Try reducing max nodes or pruning children to reduce the tree size or reduce text on each node.">Not working?</div>'
                        ),
                    ],
                    justify="space-between",
                ),
            ]
        )

        return (
            start_node,
            max_nodes,
            max_children,
            zoom,
            show_ucb,
            show_board,
            minimalist,
            hide_unvisited,
            tree_controls_layout,
        )


    # ---------------------------------------------------------
    # UNPACK FOR TREE 1
    # ---------------------------------------------------------
    (
        start_node_input,
        max_nodes_slider,
        max_children_slider,
        zoom_slider,
        show_ucb_checkbox,
        show_board_checkbox,
        minimalist_nodes_checkbox,
        hide_unvisited_checkbox,
        tree_controls,
    ) = create_tree_ui()

    show_tree_checkbox = mo.ui.checkbox(
        label="🌲 Show MCTS Tree Visualization", value=False
    )
    return (
        create_tree_ui,
        hide_unvisited_checkbox,
        max_children_slider,
        max_nodes_slider,
        minimalist_nodes_checkbox,
        show_board_checkbox,
        show_tree_checkbox,
        show_ucb_checkbox,
        start_node_input,
        tree_controls,
        zoom_slider,
    )


@app.cell
def _(
    MCTSMermaidVisualizer,
    MyGame,
    NumpyTreeAdapter,
    get_phase,
    get_turn,
    hide_unvisited_checkbox,
    interactive_mermaid,
    max_children_slider,
    max_nodes_slider,
    minimalist_nodes_checkbox,
    show_board_checkbox,
    show_tree_checkbox,
    show_ucb_checkbox,
    start_node_input,
    tree_controls,
    zoom_slider,
):
    # ==========================================================
    # CELL 5: MERMAID TREE EXPLORER
    # ==========================================================
    _tick = get_phase()
    _time = get_turn()  # Binds to the time-travel slider so it updates!

    if not show_tree_checkbox.value:
        tree_display = mo.md(
            "*Check the box above to render the AI's search tree that it did most recently.*"
        )
    else:
        current_tree = MyGame.tree_history[_time]
        if current_tree is not None:
            # 1. Instantiate the Adapter
            adapter = NumpyTreeAdapter(
                current_tree,
                display_ucb=show_ucb_checkbox.value,
                display_board=show_board_checkbox.value,
                minimal_display=minimalist_nodes_checkbox.value,
                hide_unvisited=hide_unvisited_checkbox.value,
            )

            # 2. Instantiate the Visualizer
            visualizer = MCTSMermaidVisualizer(adapter)

            # 3. Render!
            _mermaid_html = interactive_mermaid(
                visualizer.to_mermaid(
                    start_ix=start_node_input.value,
                    max_total_children_weight=max_children_slider.value,  # convert to percent
                    max_total_node_weight=max_nodes_slider.value,
                ),
                initial_zoom=zoom_slider.value,
            )

            # Pack the sliders and inputs into a neat row

            tree_display = mo.vstack([tree_controls, _mermaid_html], gap=2)
        else:
            tree_display = mo.md(
                "*(The MCTS tree will appear here after the AI takes its first turn!)*"
            )

    # Render the main toggle checkbox on top of the display area
    mo.vstack(
        [
            show_tree_checkbox,
            tree_display,
        ],
        gap=1,
    )
    return


@app.cell
def _():
    # =====================================================================
    # 1. THE ADAPTER INTERFACE (The Contract)
    # =====================================================================
    class MCTSGraphAdapter:
        """
        The interface that any tree must implement to be visualized.
        """

        def get_valid_children(self, node_ix: int) -> list[tuple[int, int]]:
            """Returns a list of (action_index, child_node_ix)."""
            raise NotImplementedError

        def get_child_display_priority(self, node_ix: int) -> float:
            """Returns the score used to sort children locally (higher is better)."""
            raise NotImplementedError

        def get_global_display_priority(self, node_ix: int) -> float:
            """Returns the score used to sort nodes globally (higher is better)."""
            raise NotImplementedError

        def get_child_display_weight(self, node_ix: int) -> float:
            """Returns the 'cost' of displaying this child. Defaults to 1.0."""
            return 1.0

        def get_global_display_weight(self, node_ix: int) -> float:
            """Returns the 'cost' of displaying this node globally. Defaults to 1.0."""
            return 1.0

        def get_brackets(self, node_ix: int) -> tuple[str, str]:
            """Returns the Mermaid opening and closing brackets for the node shape."""
            return ("(", ")")

        def get_node_text(self, node_ix: int) -> str:
            """Returns the raw HTML/Text to display inside the node."""
            raise NotImplementedError

        def get_node_class(self, node_ix: int) -> str:
            """Returns the CSS class name for styling."""
            raise NotImplementedError

        def get_edge_string(
            self, parent_ix: int, child_ix: int, action: int
        ) -> str:
            """Returns the Mermaid string for the arrow connecting parent and child."""
            raise NotImplementedError

        def get_style_defs(self) -> list[str]:
            """Returns the list of CSS class definitions for Mermaid."""
            return []

        def get_extra_mermaid_lines(
            self, display_mask: dict[int, bool]
        ) -> list[str]:
            """Returns any arbitrary extra Mermaid lines."""
            return []


    # =====================================================================
    # 2. THE VISUALIZER (DFS + String Builder)
    # =====================================================================
    class MCTSMermaidVisualizer:
        """Pure visualizer that builds a Mermaid string using the Adapter contract."""

        def __init__(self, adapter: MCTSGraphAdapter):
            self.adapter = adapter

        def to_mermaid(
            self,
            start_ix=0,
            max_total_children_weight=None,
            max_total_node_weight=None,
        ):
            # ---------------------------------------------------------
            # PASS 1: DFS Discovery & Horizontal Pruning
            # ---------------------------------------------------------
            visited = set()
            subtree_nodes = []

            # if start_ix is invalid, just start at the root instead
            if start_ix >= self.adapter.tree.largest_used_node_ix:
                start_ix = 0

            def dfs(node_ix):
                if node_ix in visited:
                    return
                visited.add(node_ix)
                subtree_nodes.append(node_ix)

                children = self.adapter.get_valid_children(node_ix)

                # Prune horizontally by cumulative weight
                if (
                    max_total_children_weight is not None
                    and max_total_children_weight > 0
                ):
                    # Sort descending by priority (highest priority first)
                    children.sort(
                        key=lambda c: self.adapter.get_child_display_priority(
                            c[1]
                        ),
                        reverse=True,
                    )

                    surviving_children = []
                    accumulated_weight = 0.0

                    for action, child_ix in children:
                        surviving_children.append((action, child_ix))
                        accumulated_weight += (
                            self.adapter.get_child_display_weight(child_ix)
                        )
                        if accumulated_weight > max_total_children_weight:
                            break

                    children = surviving_children

                for action, child_ix in children:
                    dfs(child_ix)

            dfs(start_ix)

            # ---------------------------------------------------------
            # PASS 2: Vertical Pruning (Global Budget)
            # ---------------------------------------------------------
            display_mask = {ix: False for ix in subtree_nodes}

            if max_total_node_weight is not None and max_total_node_weight > 0:
                # Sort descending by global priority
                sorted_nodes = sorted(
                    subtree_nodes,
                    key=lambda ix: self.adapter.get_global_display_priority(ix),
                    reverse=True,
                )
                # Then only show nodes until we exceed the global budget
                accumulated_weight = 0.0
                for ix in sorted_nodes:
                    accumulated_weight += self.adapter.get_global_display_weight(
                        ix
                    )
                    if accumulated_weight > max_total_node_weight:
                        break
                    display_mask[ix] = True
            else:
                for ix in subtree_nodes:
                    display_mask[ix] = True

            display_mask[start_ix] = True  # Always anchor the root!

            # ---------------------------------------------------------
            # PASS 3: Draw the Graph String
            # ---------------------------------------------------------
            # lines = ["graph TD"]
            lines = [
                '%%{init: {"flowchart": {"nodeSpacing": 15, "rankSpacing": 15}}}%%',
                "graph TD",
            ]

            # Now the root node gets properly shaped brackets too!
            root_txt = self.adapter.get_node_text(start_ix)
            open_b, close_b = self.adapter.get_brackets(start_ix)
            lines.append(f"    N{start_ix}{open_b}{root_txt}{close_b}")

            classes_used = set()

            for node_ix in subtree_nodes:
                if not display_mask[node_ix]:
                    continue

                cls = self.adapter.get_node_class(node_ix)
                classes_used.add(f"    class N{node_ix} {cls};")

                for action, child_ix in self.adapter.get_valid_children(node_ix):
                    if child_ix in display_mask and display_mask[child_ix]:
                        child_txt = self.adapter.get_node_text(child_ix)
                        open_bracket, close_bracket = self.adapter.get_brackets(
                            child_ix
                        )

                        arrow = self.adapter.get_edge_string(
                            node_ix, child_ix, action
                        )

                        lines.append(
                            f"    N{node_ix} {arrow} N{child_ix}{open_bracket}{child_txt}{close_bracket}"
                        )

            lines.extend(self.adapter.get_extra_mermaid_lines(display_mask))
            lines.extend(list(classes_used))
            lines.extend(self.adapter.get_style_defs())

            return "\n".join(lines)

    return MCTSGraphAdapter, MCTSMermaidVisualizer


@app.cell
def _(N_COLS):
    def board_to_emoji_string(parent_board, child_board, cols=N_COLS):
        """
        Converts integer array boards to an emoji string,
        highlighting the newest move made in the child_board.
        """
        # Flatten the arrays to handle both 1D and 2D board structures easily
        child_flat = np.ravel(child_board)

        # If there is no parent (e.g., the root node), compare against a completely empty board
        if parent_board is None or len(parent_board) == 0:
            parent_flat = np.zeros_like(child_flat)
        else:
            parent_flat = np.ravel(parent_board)

        emoji_list = []

        for p, c in zip(parent_flat, child_flat):
            if p == c:
                # The square hasn't changed
                if c == 0:
                    emoji_list.append("⬜")
                elif c == 1:
                    emoji_list.append("🔴")
                elif c == 2:
                    emoji_list.append("🔵")
            else:
                # This is the newly placed piece!
                if c == 1:
                    emoji_list.append("⭕")
                elif c == 2:
                    emoji_list.append("Ⓜ️")
                else:
                    emoji_list.append("⬜")  # Fallback

        # Group the emojis into rows based on column count and join them with newlines
        rows = [
            "".join(emoji_list[i : i + cols])
            for i in range(0, len(emoji_list), cols)
        ]
        return "\n".join(rows)

    return (board_to_emoji_string,)


@app.cell
def _(Game, MCTSGraphAdapter, N_COLS, board_to_emoji_string):
    # =====================================================================
    # 3. YOUR SPECIFIC ADAPTER (For the OOP NumPy Tree)
    # =====================================================================
    class NumpyTreeAdapter(MCTSGraphAdapter):
        """Wraps your existing OOP Tree class to feed the visualizer."""

        def __init__(
            self,
            tree,
            display_ucb=False,
            enable_highlighting=False,
            display_board=True,
            hide_unvisited=False,
            minimal_display=True,
        ):
            self.tree = tree
            self.display_ucb = display_ucb
            self.enable_highlighting = enable_highlighting  # Store the flag!
            self.display_board = display_board
            self.hide_unvisited = hide_unvisited
            self.minimal_display = minimal_display

        def get_valid_children(self, node_ix: int) -> list[tuple[int, int]]:
            valid_children = []
            for action, child_ix in enumerate(self.tree.children_ixs[node_ix]):
                if (
                    child_ix != 0
                    and child_ix <= self.tree.largest_used_node_ix
                    and child_ix != self.tree.ILLEGAL_CHILD_FLAG
                ):
                    if (
                        self.hide_unvisited
                        and self.tree.N_sims[child_ix] == 0
                        and (child_ix not in self.tree.highlight_path)
                    ):
                        continue  # skip over unvisited children if the flag is set

                    valid_children.append((action, int(child_ix)))
            return valid_children

        def get_child_display_priority(self, node_ix: int) -> float:
            # highest local (child pruning) priority for nodes with the most simulations
            return float(self.tree.N_sims[node_ix])

        def get_child_display_weight(self, node_ix: int) -> float:
            """Weight is the probability mass: child visits / parent visits."""
            parent_ix = int(self.tree.parent_ix[node_ix])
            parent_visits = max(1, self.tree.N_sims[parent_ix])
            child_visits = self.tree.N_sims[node_ix]

            return float(
                100 * child_visits / parent_visits
            )  # return it as a percentage so the slider is on a scale of 0 to 100

        def get_global_display_priority(self, node_ix: int) -> float:
            # highest priority for nodes with the most simulations
            return float(self.tree.N_sims[node_ix])

        def get_brackets(self, node_ix: int) -> tuple[str, str]:
            # You can use "[(", ")]" for cylinders, "((", "))" for circles, etc.
            if self.tree.is_terminal[node_ix]:
                return (
                    "[(",
                    ")]",
                )  # use cylinder for terminal nodes to make it look different
            return ("(", ")")  # round rectangle for regular nodes

        def get_node_text(self, node_ix: int) -> str:
            if self.display_board:
                # 1. Get the parent node index
                parent_ix = self.tree.parent_ix[node_ix]

                # 2. Get the parent board (pass None if this is the root node)
                parent_board = (
                    self.tree.board[parent_ix] if parent_ix != -1 else None
                )

                # 3. Get the child board
                child_board = self.tree.board[node_ix]

                # 4. Generate the string
                board_string = "\n" + board_to_emoji_string(
                    parent_board, child_board, cols=N_COLS
                )
            else:
                board_string = ""

            reward_to_phrase_dict = {
                0: "Tie. <br> Reward: 0",
                1: "P1 wins <br> Reward: +1",
                -1: "P2 wins <br> Reward: -1",
            }

            if self.tree.is_terminal[node_ix]:
                return (
                    reward_to_phrase_dict[
                        int(Game.reward(self.tree.board[node_ix]))
                    ]
                    + board_string
                )

            visits = self.tree.N_sims[node_ix]
            w_total = self.tree.total_sim_reward[node_ix]
            raw_win_rate = w_total / visits  # if visits > 0 else 0.0

            ucb_score = self.tree.ucb[node_ix]
            optimism_bonus = ucb_score - raw_win_rate

            if node_ix == self.tree.ROOT_NODE_IX or not self.display_ucb:
                opt_and_ucb_text = ""
            elif visits > 0:
                opt_and_ucb_text = (  # show optimism bonus and overall UCB score
                    f"<br>Opt: {optimism_bonus:>+4.2f}"
                    + f"<br><b>UCB: {ucb_score:>+4.2f}</b>"
                )
            elif visits == 0 and self.display_ucb:
                opt_and_ucb_text = (
                    f"<br><b>Opt: +∞ </b><br><b>UCB: +∞ </b>"
                    if self.tree.player_turn[node_ix] == 2
                    else f"<br><b>Opt: -∞ </b><br><b>UCB: -∞ </b>"
                )

            if self.minimal_display == True:
                # return board_string
                return (
                    f"N={visits}\n{raw_win_rate:>+4.2f}"
                    + board_string
                    + opt_and_ucb_text
                )
            else:
                return (  # main text of a node
                    r"Node #"
                    + str(node_ix)
                    + board_string
                    + f"<br><br> <b> N sims: {visits} </b>"
                    + f"<br><b> Avg: {raw_win_rate:>+4.2f} </b>"
                    + opt_and_ucb_text
                )

        def get_node_class(self, node_ix: int) -> str:
            player = self.tree.player_turn[node_ix]
            if self.enable_highlighting and (node_ix in self.tree.highlight_path):
                return f"Hplayer{player}"  # highlited nodes
            return f"player{player}"  # regular nodes

        def get_edge_string(
            self, parent_ix: int, child_ix: int, action: int
        ) -> str:
            if self.enable_highlighting and child_ix in self.tree.highlight_path:
                return f"==>|A{action}|"  # *thick* line along the highlited path
            return f"---|A{action}|"  # regular line labeled between nodes

        def get_style_defs(self) -> list[str]:
            P1_STYLE = "fill:#FADADD,color:#000000"  # 7B241C"  # "#ff9999"  # Lighter pastel red
            P1_STROKE = "stroke:#C0392B"
            P2_STYLE = "fill:#D6EAF8,color:#000000"  ##1A5276"  # "#99ccff"  # Lighter pastel blue
            P2_STROKE = "stroke:#2471A3"
            HILIGHT_COLOR = "#222021"
            return [
                f"    classDef player1 font-size:12px,{P1_STYLE},{P1_STROKE},stroke-width:2px,font-family:monospace,line-height:1;",
                f"    classDef player2 font-size:12px,{P2_STYLE},{P2_STROKE},stroke-width:2px,font-family:monospace,line-height:1;",
                "    classDef sim font-size:12px,stroke:#000000,stroke-width:2px,font-family:monospace,line-height:1;",
                f"    classDef Hplayer1 font-size:12px,{P1_STYLE},stroke:{HILIGHT_COLOR},stroke-width:7px,font-family:monospace,line-height:1;",
                f"    classDef Hplayer2 font-size:12px,{P2_STYLE},stroke:{HILIGHT_COLOR},stroke-width:7px,font-family:monospace,line-height:1;",
            ]

        def get_extra_mermaid_lines(
            self, display_mask: dict[int, bool]
        ) -> list[str]:
            # add rollout simulation node if needed
            lines = []
            sim_ix = self.tree.simulation_ix
            if (
                sim_ix is not None
                and sim_ix <= self.tree.largest_used_node_ix
                and display_mask.get(sim_ix, False)
            ):
                reward_to_phrase_dict = {
                    0: "Tie. <br> Reward: 0",
                    1: "P1 wins <br> Reward: +1",
                    -1: "P2 wins <br> Reward: -1",
                }

                board_string = (
                    "\n"
                    + board_to_emoji_string(
                        self.tree.board[self.tree.simulation_ix],
                        self.tree.simulation_board,
                    )
                    if self.display_board
                    else ""
                )

                text = (
                    reward_to_phrase_dict[
                        int(Game.reward(self.tree.simulation_board))
                    ]
                    + board_string
                )

                # You can easily use cylinder brackets [ ] here instead of (( )) if you want!
                lines.append(f"    N{sim_ix} -.Rollout Sim...-> Nsim[({text})]")
                lines.append("    class Nsim sim;")
            return lines

    return (NumpyTreeAdapter,)


@app.cell
def _():
    return


@app.cell
def _(DARK_BACKGROUND_COLOR, GLOBAL_BACKGROUND_COLOR):
    def interactive_mermaid(
        diagram_code: str, initial_zoom: int = 100, box_height: str = "500px"
    ):
        html_page = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                html, body {{
                    margin: 0; padding: 0; font-family: sans-serif;
                    height: 100%; width: 100%;
                    background: {GLOBAL_BACKGROUND_COLOR};
                }}
                body {{
                    display: flex; flex-direction: column; 
                    overflow: hidden;
                }}
                :fullscreen, ::backdrop {{
                    background-color: {GLOBAL_BACKGROUND_COLOR} !important;
                }}
                .toolbar {{
                    padding: 8px 14px; background: {DARK_BACKGROUND_COLOR};
                    border-bottom: 1px solid #d5cfc6;
                    display: flex; gap: 8px; align-items: center;
                }}
                .toolbar button {{
                    cursor: pointer; padding: 5px 12px;
                    border: 1px solid #c5bfb6; border-radius: 4px;
                    background: {GLOBAL_BACKGROUND_COLOR}; font-size: 13px; color: #333;
                }}
                .toolbar button:hover {{ background: #e5e0d8; }}
                .toolbar strong {{ font-size: 13px; color: #555; margin-right: 6px; }}
                .scroll-container {{
                    flex: 1; overflow: auto; padding: 20px;
                    background: {GLOBAL_BACKGROUND_COLOR};
                }}
                #zoom-wrapper {{
                    /* Start at the percentage passed by the python slider */
                    width: {initial_zoom}%; 
                    transition: width 0.2s ease-in-out, min-width 0.2s ease-in-out;
                    background: {GLOBAL_BACKGROUND_COLOR}; padding: 20px;
                    border-radius: 8px;
                    box-sizing: border-box;
                }}
            </style>
        </head>
        <body>
            <div class="toolbar">
                <strong> P1: 🔴  |  P2: 🔵  |  New Moves: ⭕ / Ⓜ️ | Scores: P1 wins=+1, P2 wins=-1, Tie=0 </strong> <strong style="margin-left: auto;">Zoom:</strong>
                <button onclick="zoom(1.2)">➕ In</button>
                <button onclick="zoom(0.8)">➖ Out</button>
                <button onclick="resetZoom()">Reset</button>
                <button onclick="toggleFullScreen()" style="margin-left: auto;">⛶ Full Screen</button>
            </div>
            <div class="scroll-container">
                <div id="zoom-wrapper">
                    <pre class="mermaid" style="margin: 0;">{diagram_code}</pre>
                </div>
            </div>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{
                    startOnLoad: true,
                    theme: 'default',
                    themeVariables: {{ background: '{GLOBAL_BACKGROUND_COLOR}', mainBkg: '{GLOBAL_BACKGROUND_COLOR}' }}
                }});
            </script>
            <script>
                const wrapper = document.getElementById('zoom-wrapper');
                let currentWidth = null;

                function zoom(f) {{
                    if (currentWidth === null) {{
                        // This perfectly captures whatever pixel width the initial_zoom % resulted in
                        currentWidth = wrapper.clientWidth;
                    }}
                    currentWidth *= f;
                    wrapper.style.width = currentWidth + 'px';
                    wrapper.style.minWidth = currentWidth + 'px';
                }}

                function resetZoom() {{
                    // Reset back to the slider's value instead of 100%
                    wrapper.style.width = '{initial_zoom}%';
                    wrapper.style.minWidth = '0px';
                    currentWidth = null; 
                }}

                function toggleFullScreen() {{
                    if (!document.fullscreenElement) {{
                        document.documentElement.requestFullscreen().catch(err => {{
                            console.warn(`Error attempting to enable full-screen mode: ${{err.message}}`);
                        }});
                    }} else {{
                        if (document.exitFullscreen) {{
                            document.exitFullscreen();
                        }}
                    }}
                }}
            </script>
        </body>
        </html>
        """
        safe_html = html.escape(html_page)
        return mo.Html(f"""
        <div style="border: 1px solid #d5cfc6; border-radius: 8px; overflow: hidden;">
            <iframe
                srcdoc="{safe_html}"
                style="width: 100%; height: {box_height}; border: none; display: block;"
                allow="fullscreen"
            ></iframe>
        </div>
        """)

    return (interactive_mermaid,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
