# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    """Code implemented can be found in the notes from Lecture 2 - Page 53
    Generic Search Algorithm for Graphs - Graph Search
    Note that since we are implementing DFS fringe is a stack"""
    closed = set()
    fringe = util.Stack()
    fringe.push( (problem.getStartState(), []) )

    while True:

        if fringe.isEmpty(): return None

        node, path  = fringe.pop()

        if problem.isGoalState(node): return path

        if node not in closed:
            closed.add(node)
            for successor, action, stepCost in problem.getSuccessors(node):
                fringe.push( (successor, path + [action]) )


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    """Code implemented can be found in the notes from Lecture 2 - Page 53
    Generic Search Algorithm for Graphs - Graph Search
    Note that since we are implementing BFS fringe is a queue"""
    closed = set()
    fringe = util.Queue()
    fringe.push( (problem.getStartState(), []) )

    while True:

        if fringe.isEmpty(): return None

        node, path  = fringe.pop()

        if problem.isGoalState(node): return path

        if node not in closed:
            closed.add(node)
            for successor, action, stepCost in problem.getSuccessors(node):
                fringe.push( (successor, path + [action]) )


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    """Code implemented can be found in the notes from Lecture 2 - Page 53
    Generic Search Algorithm for Graphs - Graph Search
    Note that since we are implementing UCS fringe is a priority queue
    Priority is determined by the distance cost"""
    closed = set()
    fringe = util.PriorityQueue()
    fringe.push( (problem.getStartState(), []), 0 )

    while True:

        if fringe.isEmpty(): return None

        node, path  = fringe.pop()

        if problem.isGoalState(node): return path

        if node not in closed:
            closed.add(node)
            for successor, action, stepCost in problem.getSuccessors(node):
                cost = problem.getCostOfActions(path) + stepCost
                fringe.push( (successor, path + [action]), cost )


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""\
    "*** YOUR CODE HERE ***"

    """Code implemented can be found in the notes from Lecture 2 - Page 53
    Generic Search Algorithm for Graphs - Graph Search
    Note that since we are implementing A* fringe is a priority queue
    Priority is determined by the distance cost plus the heuristic"""
    closed = set()
    fringe = util.PriorityQueue()
    fringe.push( (start := problem.getStartState(), []), 0 + heuristic(start, problem) )

    while True:

        if fringe.isEmpty(): return None

        node, path  = fringe.pop()

        if problem.isGoalState(node): return path

        if node not in closed:
            closed.add(node)
            for successor, action, stepCost in problem.getSuccessors(node):
                cost = problem.getCostOfActions(path) + stepCost \
                       + heuristic(successor, problem)
                fringe.push( (successor, path + [action]), cost )


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
