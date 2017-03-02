"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def heuristic_score_num_moves(game, player):
    """Outputs a heuristic score equal to the difference in current
    player's available moves and two times the opponent's available
    moves.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    
    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).
    
    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    own_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(own_moves - (2 * opponent_moves))

def heuristic_score_center_proximity(game, player):
    """A heuristic score that returns a given player's proximity to the
    center of the board. The intuition here is that players who are closest
    to the center have the greatest amount of mobility. Players near the
    edges are intrinsically constrained by the board.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    player_pos = game.get_player_location(player)
    center = [game.width / 2.0, game.height / 2.0]

    return float(abs(player_pos[0] - center[0]) + abs(player_pos[1] - center[1]))

def heuristic_score_mobility(game, player):
    """The difference in player mobility plus the difference of number of 
    potential moves.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    own_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    own_mobility = float(sum([len(game.__get_moves__(move)) for move in own_moves ]))
    opponent_mobility = float(sum([len(game.__get_moves__(move)) for move in opponent_moves ]))
    return (own_mobility - opponent_mobility) + (len(own_moves) - len(opponent_moves))

def heuristic_score_expanded_grid(game, player):
    """A heuristic score that expands the grid to twice its size. Each player's
    potential moves are multiplied times the expanded grid, and their difference
    is returned. The intuition here is that a larger grid would potentially
    offer the players higher chances of success during gameplay. With that in mind,
    a heuristic is offered as to who would win under those conditions.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    expanded_grid = (game.width * game.height) * 2
    own_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    return float(abs((len(own_moves) * expanded_grid) - (len(opponent_moves) * expanded_grid)))

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    return heuristic_score_mobility(game, player)
    

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def mm_or_ab(self, game, depth):
        # Helper method for selecting minimax or alphabeta pruning.
        return self.minimax(game, depth) if self.method == 'minimax' else self.alphabeta(game, depth)

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        if len(game.get_legal_moves(self)) == 0: return (-1, -1)
        if self.search_depth <= 0: self.iterative = True
        
        best_move = (None, (-1, -1))

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                grid_size = game.width * game.height
                for depth in range(1, grid_size):
                    best_move = self.mm_or_ab(game, depth)
                    if best_move[0] == float("inf"): break
            else:
                best_move = self.mm_or_ab(game, self.search_depth)

        except Timeout:
            return best_move[1]
        # Return the best move from the last completed search iteration
        return best_move[1]

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        scores_and_moves = set()
        player = self if maximizing_player else game.get_opponent(self)
        legal_moves = game.get_legal_moves(player)

        if len(legal_moves) == 0:
            return (game.utility(self),(-1,-1))

        for move in legal_moves:
            score,child_move = (self.score(game.forecast_move(move),self),move) if depth == 1 else self.minimax(game.forecast_move(move), depth-1, not maximizing_player)
            scores_and_moves.add((score, move))
        
        return max(scores_and_moves) if maximizing_player else min(scores_and_moves)


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        scores_and_moves = set()
        player = self if maximizing_player else game.get_opponent(self)
        legal_moves = game.get_legal_moves(player)
        
        if len(legal_moves) == 0:
            return (game.utility(self),(-1,-1))
        
        for move in legal_moves:
            if depth == 1:
                score = self.score(game.forecast_move(move), self)
                child_move = move
            else:
                score, child_move = self.alphabeta(game.forecast_move(move), depth-1, alpha, beta, not maximizing_player)
            
            if maximizing_player:
                if score >= beta: return (score,move)
                alpha = max(score, alpha)
            else:
                if score <= alpha: return (score,move)
                beta = min(score, beta)
            scores_and_moves.add((score,move))
        return max(scores_and_moves) if maximizing_player else min(scores_and_moves)