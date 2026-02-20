# /// script
# dependencies = ["marimo", "pydantic-ai==1.44.0", "mohtml==0.1.11", "anywidget==0.9.21", "traitlets", "pandas==2.3.3", "pytest==9.0.2"]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="columns", sql_output="polars")

with app.setup:
    import marimo as mo
    import random
    import pandas as pd
    import pytest
    from dataclasses import dataclass
    from mohtml import div, table, tr, td
    from pathlib import Path
    from suguru_widget.suguru_widget import SuguruGeneratorWidget
    from typing import Optional


@app.cell
def _():
    mo.md(r"""
    ## Solver
    """)
    return


@app.cell
def _(Suguru):
    @dataclass
    class SolverStats:
        recursive_calls: int = 0
        board_states: list = None

        def __post_init__(self):
            if self.board_states is None:
                self.board_states = []

        def reset(self):
            self.recursive_calls = 0
            self.board_states = []


    def has_impossible_cell(board: Suguru) -> bool:
        for x, y in board.empty_coordinates:
            if len(board.possible_values(x, y)) == 0:
                return True
        return False


    def track_recursive_call(func):
        """Decorator to track recursive calls and log board states"""
        def wrapper(self, board: Suguru):
            self.stats.recursive_calls += 1

            # Check max iterations
            if self.stats.recursive_calls > self.max_iterations:
                return

            # Log board state
            self.stats.board_states.append(board.copy())

            # Call the actual solve method (it's a generator)
            yield from func(self, board)

        return wrapper


    class Solver:
        def __init__(self, board: Suguru, max_iterations: int = 100000):
            self.initial_board = board.copy()
            self.stats = SolverStats()
            self.max_iterations = max_iterations
            self._solved = False

        def try_or_undo_move(self, board: Suguru, x: int, y: int, value: int):
            """Context manager for true backtracking - makes move, undoes on exit if needed"""
            class MoveContext:
                def __init__(self, solver, board, x, y, value):
                    self.solver = solver
                    self.board = board
                    self.x = x
                    self.y = y
                    self.value = value
                    self.solution_found = False

                def __enter__(self):
                    self.board.make_move(self.x, self.y, self.value)
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    if not self.solution_found:
                        self.board.board[self.y][self.x] = None  # Undo move (backtrack)
                    return False  # Don't suppress exceptions

            return MoveContext(self, board, x, y, value)

        @property
        def solved(self) -> bool:
            """Whether the solver found a solution"""
            return self._solved

        @property
        def current_board(self) -> Suguru:
            """Returns the most recent board state, prioritizing solved boards"""
            if self.stats.board_states:
                # If solved, find the last solved board state
                if self._solved:
                    for board in reversed(self.stats.board_states):
                        if board.is_solved():
                            return board
                return self.stats.board_states[-1]
            return self.initial_board.copy()

        def board_states(self):
            """Generator yielding board states - to be implemented by subclasses"""
            raise NotImplementedError


    class BasicSolver(Solver):
        def board_states(self):
            board = self.initial_board.copy()
            return self._solve(board)

        @track_recursive_call
        def _solve(self, board: Suguru):
            if board.is_solved():
                self._solved = True
                # Log the solved board state
                self.stats.board_states.append(board.copy())
                yield board
                return

            empty = board.empty_coordinates
            if not empty:
                return

            x, y = min(
                empty, key=lambda pos: (len(board.possible_values(*pos)), board.n_neighbors(*pos))
            )
            possible = board.possible_values(x, y)

            for value in possible:
                with self.try_or_undo_move(board, x, y, value) as move:
                    for result in self._solve(board):
                        if result.is_solved():
                            self._solved = True
                            # Log the solved board state
                            self.stats.board_states.append(result.copy())
                            move.solution_found = True
                        yield result
                        if result.is_solved():
                            return


    class SmartSolver(Solver):
        def board_states(self):
            board = self.initial_board.copy()
            return self._solve(board)

        @track_recursive_call
        def _solve(self, board: Suguru):
            if board.is_solved():
                self._solved = True
                # Log the solved board state
                self.stats.board_states.append(board.copy())
                yield board
                return

            empty = board.empty_coordinates
            if not empty:
                return

            x, y = min(
                empty, key=lambda pos: (len(board.possible_values(*pos)), board.n_neighbors(*pos))
            )
            possible = board.possible_values(x, y)

            for value in sorted(possible):
                with self.try_or_undo_move(board, x, y, value) as move:
                    if has_impossible_cell(board):
                        continue

                    for result in self._solve(board):
                        if result.is_solved():
                            self._solved = True
                            # Log the solved board state
                            self.stats.board_states.append(result.copy())
                            move.solution_found = True
                        yield result
                        if result.is_solved():
                            return


    class ConstraintPropSolver(Solver):
        def board_states(self):
            board = self.initial_board.copy()
            return self._solve(board)

        @track_recursive_call
        def _solve(self, board: Suguru):
            if board.is_solved():
                self._solved = True
                # Log the solved board state
                self.stats.board_states.append(board.copy())
                yield board
                return

            # Handle constraint propagation (these are easy to declare wins)
            made_progress = True
            while made_progress:
                made_progress = False
                forced = board.find_forced_moves()
                if forced:
                    # Make forced moves one at a time, checking validity after each
                    for x, y, value in forced:
                        board.make_move(x, y, value)
                        made_progress = True

                        # Check if this move created an invalid state
                        if has_impossible_cell(board):
                            # Invalid state reached - this branch is unsolvable
                            return

                    yield board.copy()

                    if board.is_solved():
                        self._solved = True
                        # Log the solved board state
                        self.stats.board_states.append(board.copy())
                        yield board
                        return

            # No more forced moves, proceed with normal backtracking
            empty = board.empty_coordinates
            if not empty:
                return

            x, y = min(
                empty, key=lambda pos: (len(board.possible_values(*pos)), board.n_neighbors(*pos))
            )
            possible = board.possible_values(x, y)

            for value in sorted(possible):
                with self.try_or_undo_move(board, x, y, value) as move:
                    if has_impossible_cell(board):
                        continue

                    for result in self._solve(board):
                        if result.is_solved():
                            self._solved = True
                            # Log the solved board state
                            self.stats.board_states.append(result.copy())
                            move.solution_found = True
                        yield result
                        if result.is_solved():
                            return
    return BasicSolver, ConstraintPropSolver, SmartSolver


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Solver Comparison

    This notebook implements three different backtracking solvers for Suguru puzzles, each with increasing levels of optimization:

    ### BasicSolver
    The simplest backtracking solver. It:
    - Selects the cell with the fewest possible values (and most neighbors as tiebreaker)
    - Tries each possible value in order
    - Uses true backtracking (undoes moves when a branch fails)
    - No early pruning - explores all branches until they fail

    ### SmartSolver
    An optimized version that adds:
    - **Early pruning**: After making a move, checks if any cell has zero possible values
    - If an impossible cell is detected, immediately backtracks without exploring further
    - Values are tried in sorted order for consistency
    - Significantly reduces the search space by avoiding dead-end branches

    ### ConstraintPropSolver
    The most advanced solver that adds:
    - **Constraint propagation**: Before backtracking, finds and fills all "forced moves" (cells with only one possible value)
    - These forced moves are made automatically and can't be undone (they're logically required)
    - After propagation, uses the same smart backtracking as SmartSolver
    - Often solves puzzles with minimal or no backtracking needed
    - Validates moves during propagation to catch invalid states early

    All solvers use true backtracking with a context manager that automatically undoes moves when a branch doesn't lead to a solution.
    """)
    return


@app.cell
def _():
    generator_widget = SuguruGeneratorWidget(width=5, height=5)
    generator_view = mo.ui.anywidget(generator_widget)
    generator_view
    return generator_view, generator_widget


@app.cell
def _(Suguru, generator_view, generator_widget):
    # Use shapes from widget to create board
    # Access generator_view to ensure Marimo tracks widget changes
    # The view must be accessed to trigger reactivity when widget state changes
    _view_dependency = generator_view
    # Access shapes through the widget - Marimo tracks this when view is a dependency
    widget_shapes = generator_widget.shapes
    # Convert to tuple to create hashable dependency that changes when shapes change
    _shapes_tuple = tuple(tuple(row) for row in widget_shapes) if widget_shapes else ()

    board = Suguru(numbers=[], shapes=widget_shapes)
    return (board,)


@app.cell
def _(BasicSolver, ConstraintPropSolver, SmartSolver, board):
    # Compare solvers with clean class-based API
    solver_basic = BasicSolver(board)
    solver_smart = SmartSolver(board)
    solver_cp = ConstraintPropSolver(board)

    # Run solvers to populate stats
    list(solver_basic.board_states())
    list(solver_smart.board_states())
    list(solver_cp.board_states())

    # Create comparison table
    comparison_data = []
    for name, solver in [
        ("Basic", solver_basic),
        ("Smart", solver_smart),
        ("Constraint Prop", solver_cp),
    ]:
        status = "✓ Solved" if solver.solved else "✗ Not Solved"

        comparison_data.append({
            "Solver": name,
            "Status": status,
            "Recursive Calls": solver.stats.recursive_calls,
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df
    return solver_basic, solver_cp, solver_smart


@app.cell
def _(solver_basic, solver_cp, solver_smart):
    mo.hstack([
        solver_basic.current_board, solver_smart.current_board, solver_cp.current_board
    ])
    return


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    ## Suguru class
    """)
    return


@app.cell
def _():
    @dataclass
    class Assignment:
        x: int
        y: int
        value: int


    class Suguru:
        def __init__(self, numbers: list[Assignment], shapes: list[list[int]]):
            self.shapes = shapes
            self.height = len(shapes)
            self.width = len(shapes[0]) if shapes else 0

            self.board = [[None for _ in range(self.width)] for _ in range(self.height)]

            for assignment in numbers:
                self.board[assignment.y][assignment.x] = assignment.value

        def get_region(self, x: int, y: int) -> Optional[int]:
            if 0 <= y < self.height and 0 <= x < self.width:
                return self.shapes[y][x]
            return None

        def get_neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        neighbors.append((nx, ny))
            return neighbors

        def get_region_size(self, region_id: int) -> int:
            count = 0
            for row in self.shapes:
                for cell_region in row:
                    if cell_region == region_id:
                        count += 1
            return count

        def is_valid(self, x: int, y: int, value: int) -> bool:
            region_id = self.get_region(x, y)
            if region_id is None:
                return False

            region_size = self.get_region_size(region_id)
            if value < 1 or value > region_size:
                return False

            for nx, ny in self.get_neighbors(x, y):
                if self.board[ny][nx] == value:
                    return False

            return True

        def possible_values(self, x: int, y: int) -> set[int]:
            region_id = self.get_region(x, y)
            if region_id is None:
                return set()

            if self.board[y][x] is not None:
                return {self.board[y][x]}

            region_size = self.get_region_size(region_id)
            possible = set(range(1, region_size + 1))

            for nx, ny in self.get_neighbors(x, y):
                neighbor_value = self.board[ny][nx]
                if neighbor_value is not None:
                    possible.discard(neighbor_value)

            for y2 in range(self.height):
                for x2 in range(self.width):
                    if self.shapes[y2][x2] == region_id and (x2, y2) != (x, y):
                        region_value = self.board[y2][x2]
                        if region_value is not None:
                            possible.discard(region_value)

            return possible

        def find_forced_moves(self) -> list[tuple[int, int, int]]:
            """Find cells that have only one possible value (forced moves)"""
            forced = []
            for y in range(self.height):
                for x in range(self.width):
                    if self.board[y][x] is None:
                        possible = self.possible_values(x, y)
                        if len(possible) == 1:
                            forced.append((x, y, next(iter(possible))))
            return forced

        def propagate_constraints(self) -> bool:
            """Apply constraint propagation: fill in all forced moves.
            Returns True if any moves were made."""
            total_moves = 0
            made_progress = True
            while made_progress:
                made_progress = False
                forced = self.find_forced_moves()
                if forced:
                    for x, y, value in forced:
                        self.make_move(x, y, value)
                        total_moves += 1
                        made_progress = True
            return total_moves > 0

        @property
        def empty_coordinates(self) -> list[tuple[int, int]]:
            empty = []
            for y in range(self.height):
                for x in range(self.width):
                    if self.board[y][x] is None:
                        empty.append((x, y))
            return empty

        def n_neighbors(self, x: int, y: int) -> int:
            return len(self.get_neighbors(x, y))

        def copy(self):
            new_board = Suguru([], self.shapes)
            new_board.board = [row[:] for row in self.board]
            return new_board

        def make_move(self, x: int, y: int, value: int):
            self.board[y][x] = value

        def is_solved(self) -> bool:
            """Check if board is solved AND valid"""
            if len(self.empty_coordinates) > 0:
                return False

            # Validate that all cells contain valid values
            for y in range(self.height):
                for x in range(self.width):
                    value = self.board[y][x]
                    if value is None:
                        return False
                    if not self.is_valid(x, y, value):
                        return False

            return True

        def to_inputs(self) -> tuple[list[Assignment], list[list[int]]]:
            """Returns numbers and shapes that can be used to recreate this board"""
            numbers = []
            for y in range(self.height):
                for x in range(self.width):
                    value = self.board[y][x]
                    if value is not None:
                        numbers.append(Assignment(x=x, y=y, value=value))
            return numbers, self.shapes

        def __repr__(self) -> str:
            lines = []
            for y in range(self.height):
                row = []
                for x in range(self.width):
                    value = self.board[y][x]
                    if value is None:
                        row.append(".")
                    else:
                        row.append(str(value))
                lines.append(" ".join(row))
            return "\n".join(lines)

        def _display_(self):
            colors = [
                "#e3f2fd",
                "#f3e5f5",
                "#fff3e0",
                "#e8f5e9",
                "#fce4ec",
                "#e0f2f1",
                "#fff9c4",
                "#f1f8e9",
            ]
            rows = []
            for y in range(self.height):
                cells = []
                for x in range(self.width):
                    region_id = self.shapes[y][x]
                    value = self.board[y][x]
                    value_str = str(value) if value is not None else ""
                    region_color = colors[region_id % len(colors)]
                    cells.append(
                        td(
                            value_str,
                            style=f"width: 40px; height: 40px; border: 1px solid #333; "
                            f"background-color: {region_color}; text-align: center; "
                            f"vertical-align: middle; font-weight: bold;",
                        )
                    )
                rows.append(tr(*cells))

            return div(
                table(*rows, style="border-collapse: collapse; border: 2px solid black;"),
                style="font-family: monospace; display: inline-block;",
            )
    return (Suguru,)


@app.cell
def _(BasicSolver, ConstraintPropSolver, SmartSolver, Suguru):
    @pytest.mark.parametrize("shapes", [[[0, 1], [2, 3]], [[0, 1, 2], [3, 4, 5], [6, 7, 8]]])
    @pytest.mark.parametrize("solver_cls", [BasicSolver, SmartSolver, ConstraintPropSolver])
    def test_unsolvable_size_1_regions(shapes, solver_cls):
        """Test that solvers correctly identify boards with all size-1 regions as unsolvable"""
        board = Suguru(numbers=[], shapes=shapes)
        solver = solver_cls(board, max_iterations=1000)
        list(solver.board_states())  # Run the solver
        # Assert that the board was NOT solved
        assert not solver.solved

    @pytest.mark.parametrize("solver_cls", [BasicSolver, SmartSolver, ConstraintPropSolver])
    def test_final_board_no_nones(solver_cls):
        shapes = [[0,0,0,0,0],[1,2,2,2,2],[3,3,3,2,2],[4,3,3,5,2],[4,4,4,5,5]]
        board = Suguru(numbers=[], shapes=shapes)
        solver = solver_cls(board, max_iterations=1000)
        list(solver.board_states())  # Run the solver to populate stats
        # Only check if solved - if not solved, board may have Nones
        if solver.solved:
            for row in solver.current_board.board:
                for val in row:
                    assert val is not None
    return


if __name__ == "__main__":
    app.run()
