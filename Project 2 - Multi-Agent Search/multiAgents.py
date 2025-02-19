# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        points = 0

        # The less food pellets there are the better
        points -= len(newFood.asList()) * 10

        # The closer you are to a food pellet the better
        food_dist = []
        for food_pos in newFood.asList():
            food_dist.append( manhattanDistance(newPos, food_pos) )
        if len(food_dist):
            min_food_dist = min(food_dist)
            points -= min_food_dist * 1/2

        # Always keep on moving
        if action == "Stop":
            points -= 100

        # Always keep appropriate distance from ghost / ghosts
        ghost_dist = []
        for ghostPos in GameState.getGhostPositions(successorGameState):
            man_dist = manhattanDistance(newPos, ghostPos)
            ghost_dist.append(man_dist)
            if man_dist < 2:
                points -= 100

        # EAT THE GHOST!!!
        capsules = successorGameState.getCapsules()
        capsule_dist = []
        for capsule in capsules:
            man_dist = manhattanDistance(newPos, capsule)
            capsule_dist.append(man_dist)

        if len(capsule_dist):
            min_capsule_dist = min(capsule_dist)
            points -= min_capsule_dist

        counter = 0
        for ghostState in newGhostStates:
            if ghostState.scaredTimer:
                points += 100
                points -= ghost_dist[counter] * 2
            counter += 1

        return points

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Call minimax for each possible action, evaluate score and return best=max
        moves = []
        possible_actions = gameState.getLegalActions(0)
        for action in possible_actions:
            child = gameState.generateSuccessor(0, action)
            moves.append( (self.minimax(child, 1, 0), action) )
        return max(moves)[1]

    # Recursive implementation of the minimax algorithm
    def minimax(self, gameState: GameState, agentIndex, currDepth):
        # Base Case
        if currDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # Maximizing Agent
        if not agentIndex:
            maxEval = float('-inf')
            possible_actions = gameState.getLegalActions(agentIndex)
            for action in possible_actions:
                child = gameState.generateSuccessor(agentIndex, action)
                eval = self.minimax(child, agentIndex + 1, currDepth)
                maxEval = max(maxEval, eval)
            return maxEval

        # Minimizing Agent
        elif agentIndex:
            minEval = float('inf')
            possible_actions = gameState.getLegalActions(agentIndex)
            for action in possible_actions:
                child = gameState.generateSuccessor(agentIndex, action)
                # Move to next level in depth if necessary
                if agentIndex + 1 == gameState.getNumAgents():
                    eval = self.minimax(child, 0, currDepth + 1)
                else:
                    eval = self.minimax(child, agentIndex + 1, currDepth)
                minEval = min(minEval, eval)
            return minEval

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Call minimax for each possible action, evaluate score and return best=max
        moves = []
        alpha = float('-inf')
        possible_actions = gameState.getLegalActions(0)
        for action in possible_actions:
            child = gameState.generateSuccessor(0, action)
            value = self.minimax_ab(child, 1, 0, alpha, float('inf'))
            moves.append( (value, action) )
            if value > alpha: alpha = value
        return max(moves)[1]

    # Recursive implementation of the minimax algorithm with a-b pruning
    def minimax_ab(self, gameState: GameState, agentIndex, currDepth, alpha, beta):
        # Base Case
        if currDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # Maximizing Agent
        if not agentIndex:
            maxEval = float('-inf')
            possible_actions = gameState.getLegalActions(agentIndex)
            for action in possible_actions:
                child = gameState.generateSuccessor(agentIndex, action)
                eval = self.minimax_ab(child, agentIndex + 1, currDepth, alpha, beta)
                maxEval = max(maxEval, eval)
                # Check for pruning!
                if maxEval > beta: return maxEval
                alpha = max(alpha, maxEval)
            return maxEval

        # Minimizing Agent
        elif agentIndex:
            minEval = float('inf')
            possible_actions = gameState.getLegalActions(agentIndex)
            for action in possible_actions:
                child = gameState.generateSuccessor(agentIndex, action)
                # Move to next level in depth if necessary
                if agentIndex + 1 == gameState.getNumAgents():
                    eval = self.minimax_ab(child, 0, currDepth + 1, alpha, beta)
                else:
                    eval = self.minimax_ab(child, agentIndex + 1, currDepth, alpha, beta)
                minEval = min(minEval, eval)
                # Check for pruning!
                if minEval < alpha: return minEval
                beta = min(beta, minEval)
            return minEval

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Call expectimax for each possible action, evaluate score and return best=max
        moves = []
        possible_actions = gameState.getLegalActions(0)
        for action in possible_actions:
            child = gameState.generateSuccessor(0, action)
            moves.append( (self.minimax(child, 1, 0), action) )
        return max(moves)[1]

    # Recursive implementation of the minimax algorithm
    def minimax(self, gameState: GameState, agentIndex, currDepth):
        # Base Case
        if currDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # Maximizing Agent
        if not agentIndex:
            maxEval = float('-inf')
            possible_actions = gameState.getLegalActions(agentIndex)
            for action in possible_actions:
                child = gameState.generateSuccessor(agentIndex, action)
                eval = self.minimax(child, agentIndex + 1, currDepth)
                maxEval = max(maxEval, eval)
            return maxEval

        # Minimizing Agent
        elif agentIndex:
            #minEval = float('inf')
            sum = 0
            possible_actions = gameState.getLegalActions(agentIndex)
            for action in possible_actions:
                child = gameState.generateSuccessor(agentIndex, action)
                # Move to next level in depth if necessary
                if agentIndex + 1 == gameState.getNumAgents():
                    eval = self.minimax(child, 0, currDepth + 1)
                    sum += eval
                else:
                    eval = self.minimax(child, agentIndex + 1, currDepth)
                    sum += eval
                #minEval = min(minEval, eval)
            # Difference from minimax comes from the returned value
            return sum/len(possible_actions)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    points = scoreEvaluationFunction(currentGameState)

    # The less food pellets there are the better
    points += -len(newFood.asList()) * 10

    # The closer you are to a food pellet the better
    food_dist = []
    for food_pos in newFood.asList():
        food_dist.append(manhattanDistance(newPos, food_pos))
    if len(food_dist):
        min_food_dist = min(food_dist)
        points += -min_food_dist * 1 / 2

    # Always keep appropriate distance from ghost / ghosts
    ghost_dist = []
    for ghostPos in GameState.getGhostPositions(currentGameState):
        man_dist = manhattanDistance(newPos, ghostPos)
        ghost_dist.append(man_dist)
        if man_dist < 2:
            points -= 100

    # EAT THE GHOST!!!
    capsules = currentGameState.getCapsules()
    capsule_dist = []
    for capsule in capsules:
        man_dist = manhattanDistance(newPos, capsule)
        capsule_dist.append(man_dist)

    if len(capsule_dist):
        min_capsule_dist = min(capsule_dist)
        points -= min_capsule_dist

    counter = 0
    for ghostState in newGhostStates:
        if ghostState.scaredTimer:
            points += 100
            points -= ghost_dist[counter] * 2
        counter += 1

    return points

# Abbreviation
better = betterEvaluationFunction
