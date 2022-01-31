from api import State, util
import random, os
from itertools import chain
import math
import joblib

DEFAULT_MODEL = os.path.dirname(os.path.realpath(__file__)) + '/model1.pkl'


class Bot:

	# How many samples to take per move
	__num_samples = -1
	# How deep to sample
	__depth = -1
	__max_depth = -1
	__randomize = True
	moves = []

	def __init__(self, num_samples=4, depth=8, randomize=True, model_file=DEFAULT_MODEL):
		self.__num_samples = num_samples
		self.__depth = depth
		self.__model = joblib.load(model_file)
		self.__randomize = randomize
		self.__max_depth = depth


	####PHASE 1#####

	def move_phase_1(self, state):
		global moves
		# Does return a move at the end

		# See if we're player 1 or 2
		player = state.whose_turn()

		# Get a list of all legal moves in the current state
		moves = state.moves()

		# Sometimes many moves have the same, highest score, and we'd like the bot to pick a random one.
		# Shuffling the list of moves ensures that.
		random.shuffle(moves)

		best_score = float("-inf")
		best_move = None

		scores = [0.0] * len(moves)

		shorter_moves = self.cut_down(state, moves)
		belief = self.dynamic_balance(len(shorter_moves), self.__num_samples)

		# Implement the new functions (cut down), store the list and
		# change it to moves at line 50
		# Call dynamic balance function storing the number that it returns
		# Put stored number to line 53
		for move in shorter_moves:
			for s in range(belief):

				# If we are in an imperfect information state, make an assumption.
				#print("PHASE------>", state.get_phase())
				if state.get_phase() == 1:
					sample_state = state.make_assumption()
				else:
					sample_state = state

				score = self.evaluate(sample_state.next(move), player)

				if score > best_score:
					best_score = score
					best_move = move

		return best_move # Return the best scoring move

	def evaluate(self,
				 state,     # type: State
				 player     # type: int
			):
		# type: () -> float
		"""
		Evaluates the value of the given state for the given player
		:param state: The state to evaluate
		:param player: The player for whom to evaluate this state (1 or 2)
		:return: A float representing the value of this state for the given player. The higher the value, the better the
			state is for the player.
		"""

		score = 0.0

		for _ in range(self.__num_samples):

			st = state.clone()

			# Do some random moves
			for i in range(self.__depth):
				if st.finished():
					break

				st = st.next(random.choice(st.moves()))

			score += self.heuristic(st, player)

		return score/float(self.__num_samples) # Avaraging

	def heuristic(self, state, player):
		return util.ratio_points(state, player)

	def get_feature(self, state, move):
		# return a list [move/int,state.p1,...]
		feature_set = []
		# print(state.__leads_turn)

		# Add player 1's points to feature set
		p1_points = state.get_points(1)

		# Add player 2's points to feature set
		p2_points = state.get_points(2)

		# Add player 1's pending points to feature set
		p1_pending_points = state.get_pending_points(1)

		# Add plauer 2's pending points to feature set
		p2_pending_points = state.get_pending_points(2)

		# Get trump suit
		trump_suit = state.get_trump_suit()

		# Add phase to feature set
		phase = state.get_phase()

		# Add stock size to feature set
		stock_size = state.get_stock_size()

		# Add opponent's played card to feature set
		opponents_played_card = state.get_opponents_played_card()

		# ADDED ones:

		# Add if the card has a higher score the opponent's
		# Check if same suit
		# Check if trump suit
		if move[0] != None:
			trump_suit_comparison = 1 if util.get_suit(move[0]) == trump_suit else 0
			if not state.leader():
				higer_score = 1 if util.get_rank(move[0]) > util.get_rank(opponents_played_card) else 0
				same_suit = 1 if util.get_suit(move[0]) == util.get_suit(opponents_played_card) else 0
				state_leader = 0
			else:
				higer_score = 2
				same_suit = 2
				state_leader = 1
		else:
			trump_suit_comparison = 2
			higer_score = 3
			same_suit = 3
			state_leader = state.leader()

		# Move
		possible_move = move[0]

		perspective = state.get_perspective()

		# Perform one-hot encoding on the perspective.
		perspective = [card if card != 'U' else [1, 0, 0, 0, 0, 0] for card in perspective]
		perspective = [card if card != 'S' else [0, 1, 0, 0, 0, 0] for card in perspective]
		perspective = [card if card != 'P1H' else [0, 0, 1, 0, 0, 0] for card in perspective]
		perspective = [card if card != 'P2H' else [0, 0, 0, 1, 0, 0] for card in perspective]
		perspective = [card if card != 'P1W' else [0, 0, 0, 0, 1, 0] for card in perspective]
		perspective = [card if card != 'P2W' else [0, 0, 0, 0, 0, 1] for card in perspective]

		# Append one-hot encoded perspective to feature_set
		feature_set += list(chain(*perspective))

		# Append normalized points to feature_set
		total_points = p1_points + p2_points
		feature_set.append(p1_points / total_points if total_points > 0 else 0.)
		feature_set.append(p2_points / total_points if total_points > 0 else 0.)

		# Append normalized pending points to feature_set
		total_pending_points = p1_pending_points + p2_pending_points
		feature_set.append(p1_pending_points / total_pending_points if total_pending_points > 0 else 0.)
		feature_set.append(p2_pending_points / total_pending_points if total_pending_points > 0 else 0.)

		# Convert trump suit to id and add to feature set
		# You don't need to add anything to this part
		suits = ["C", "D", "H", "S"]
		trump_suit_onehot = [0, 0, 0, 0]
		trump_suit_onehot[suits.index(trump_suit)] = 1
		feature_set += trump_suit_onehot

		# Append one-hot encoded phase to feature set
		feature_set += [1, 0] if phase == 1 else [0, 1]

		# Append normalized stock size to feature set
		feature_set.append(stock_size / 10)

		# Append one-hot encoded opponent's card to feature set
		opponents_played_card_onehot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		opponents_played_card_onehot[opponents_played_card if opponents_played_card is not None else 20] = 1
		feature_set += opponents_played_card_onehot

		# ADDED

		# Append one-hot encoded possible move
		possible_move_onehot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		possible_move_onehot[possible_move] = 1
		feature_set += possible_move_onehot
		feature_set.append(possible_move)
		#
		higer_score_onehot = [0, 0, 0, 0]
		higer_score_onehot[higer_score] = 1
		feature_set.append(higer_score)
		#
		# same_suit_onehot = [0, 0, 0, 0]
		# same_suit_onehot[same_suit] = 1
		# feature_set += same_suit_onehot
		#
		# # Append added ones
		# trump_suit_comparison_onehot = [0, 0, 0]
		# trump_suit_comparison_onehot[trump_suit_comparison] = 1
		# feature_set += trump_suit_comparison_onehot
		#
		# #state leader
		 #feature_set.append(state_leader)

		# Return feature set
		return feature_set


	def dynamic_balance(self, n_shorter_moves, n_samp):
		global moves

		if n_shorter_moves == len(moves):
			return n_samp
		else:
			complexity_value = len(moves) ** n_samp
			return int(math.log(complexity_value, n_shorter_moves))

	def cut_down(self, state, moves):
		# Compute the Monte Carlo Sampling Value of each move and return a shorter list
		values = []
		new_list = []
		for move in moves:
			if move[1] == None and move[0] != None:
				feature = self.get_feature(state, move)
				# Predict the Monte Carlo Value for each move with ML ()
				value = self.__model.predict([feature])
				values.append((value,move))
			else:
				new_list.append(move)
		# Delete all the moves that are not optimal according to the model
		values.sort(reverse=True)
		shortened_list = values[::]
		# X is the cut-off value
		x = 0.3
		for i in range(len(values[2:])):
			if values[0][0] - values[i + 2][0] >= x:
				shortened_list.remove(values[i + 2])
		for tuple in shortened_list:
			new_list.append(tuple[1])
		return new_list

	#######PHASE 2 (alpha beta pruning)########

	def move_phase_2(self, state):
		val, move = self.value(state)

		return move

	def value(self, state, alpha=float('-inf'), beta=float('inf'), depth = 0):
		if state.finished():
			winner, points = state.winner()
			return (points, None) if winner == 1 else (-points, None)

		if depth == self.__max_depth:
			return self.heuristic_2(state)

		best_value = float('-inf') if self.maximizing(state) else float('inf')
		best_move = None

		moves = state.moves()

		if self.__randomize:
			random.shuffle(moves)

		for move in moves:

			next_state = state.next(move)
			value, _ = self.value(next_state)

			if self.maximizing(state):
				if value > best_value:
					best_value = value
					best_move = move
					alpha = best_value
			else:
				if value < best_value:
					best_value = value
					best_move = move
					beta = best_value

			# Prune the search tree
			# We know this state will never be chosen, so we stop evaluating its children
			if value >= beta or value <= alpha :
				break

		return best_value, best_move

	def maximizing(self, state):
		return state.whose_turn() == 1

	def heuristic_2(self, state):
		return util.ratio_points(state, 1) * 2.0 - 1.0, None

	#### GET MOVE ####

	def get_move(self, state):
		if state.get_phase() == 1:
			move = self.move_phase_1(state)
			return move
		else:
			move = self.move_phase_2(state)
			return move