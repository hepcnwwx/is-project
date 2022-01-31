from api import State, util, Deck
import random
import math


class Bot:

	# How many samples to take per move
	__num_samples = -1
	# How deep to sample
	__depth = -1
	__max_depth = -1
	__randomize = True
	moves = []

	def __init__(self, num_samples=4, depth=8, randomize=True):
		self.__num_samples = num_samples
		self.__depth = depth
		self.__randomize = randomize
		self.__max_depth = depth

	def get_move(self, state):
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

		moves_2 = state.moves()

		scores = [0.0] * len(moves)
		shorter_moves = self.cut_down(state, moves)
		if len(shorter_moves) > 0:
			belief = self.dynamic_balance(len(shorter_moves), self.__num_samples)
		else:
			belief = self.dynamic_balance(len(moves_2), self.__num_samples)
			shorter_moves = moves_2

		for move in shorter_moves:
			for s in range(belief):

				# If we are in an imperfect information state, make an assumption.
				if state.get_phase() == 1:
					sample_state = state.make_assumption()
				else:
					sample_state = state

				score = self.evaluate(sample_state.next(move), player)

				if score > best_score:
					best_score = score
					best_move = move
		return best_move # Return the best scoring move

	def evaluate(self,state,player):

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

	def dynamic_balance(self, n_shorter_moves, n_samp):
		global moves
		if n_shorter_moves == len(moves):
			return n_samp
		else:
			try:
				complexity_value = len(moves) ** n_samp
				return int(math.log(complexity_value, n_shorter_moves))
			except:
				return n_samp

	def cut_down(self, state, moves):
		# Compute the Monte Carlo Sampling Value of each move and return a shorter list
		for o in moves:
			if o[0] == None:
				return [o]
		values = []
		new_list = moves[::]
		trump_suit = state.get_trump_suit()

		#if we are the leader
		if state.leader() == state.whose_turn():
			briscola = False
			for move in moves:
				if move[0] != None or move[1] != None:
					if self.first_heur(moves, move, trump_suit):
						new_list.remove(move)
					else:
						return moves

				elif move[0] != None and move[1] != None:
					moves_left = moves.remove(move)
					for m in moves_left:
						if m[0] == move[0] or m[0] == move[1]:
							new_list.remove(m)
				else:
					return moves
				if Deck.get_suit(move[0]) == trump_suit:
					briscola = True
			if not briscola:
				best_val = 0
				best_move = []
				for n in new_list:
					if self.numeric_rank(n[0]) > best_val:
						best_move = []
						best_move.append(n)
						best_val = self.numeric_rank(n[0])
					elif self.numeric_rank(n[0]) == best_val:
						best_move.append(n)
					else:
						pass
				for p in best_move:
					new_list.remove(p)
			else:
				return new_list
		else:
			#if we are not the leader
			opponents_played_card = state.get_opponents_played_card()
			#if opponent plays ace of trump suit we play a non trump suit jack (if we have it)
			if opponents_played_card is not None:
				if Deck.get_suit(opponents_played_card) == trump_suit:
					if self.numeric_rank(opponents_played_card) == 3 or self.numeric_rank(opponents_played_card) == 4:
						worst_val = 5
						worst_move = []
						for n in moves: #new_list
							if self.numeric_rank(n[0]) < worst_val:
								worst_move = []
								worst_move.append(n)
								worst_val = self.numeric_rank(n[0])
							elif self.numeric_rank(n[0]) == worst_val:
								worst_move.append(n)
							else:
								pass
						return worst_move
					else:
						return new_list
				else:
					if self.numeric_rank(opponents_played_card) == 3 or self.numeric_rank(opponents_played_card) == 4:
						acc_moves = []
						for y in moves:
							if Deck.get_suit(opponents_played_card) == Deck.get_suit(y[0]) and self.numeric_rank(opponents_played_card) < self.numeric_rank(y[0]):
								acc_moves.append(y)
							elif Deck.get_suit(y[0]) == trump_suit:
								acc_moves.append(y)
							else:
								pass
						if len(acc_moves) > 0:
							return acc_moves
						else:
							return new_list
			else:
				pass
		return new_list

	def first_heur(self, moves, move, trump_suit):
		# if we have a trump ace or a 10 we don't play it ( if we have other cards)
		# Returns True if we have a trump Ace or 10
		mooove = []
		moves_left = moves
		moves_left.remove(move)
		for m in moves_left:
			mooove.append(self.numeric_rank(m[0]))
			# TODO: We might be able to cut down here, this doesn't have to be in a for loop
			if Deck.get_suit(move[0]) == trump_suit:
				if self.numeric_rank(move[0]) == 4 or self.numeric_rank(move[0]) == 3:
					if 0 not in mooove and 1 not in mooove:
						return True
		return False

	def numeric_rank(self, move):
		if Deck.get_rank(move) == "J":
			return 0
		if Deck.get_rank(move) == "Q":
			return 1
		if Deck.get_rank(move) == "K":
			return 2
		if Deck.get_rank(move) == "10":
			return 3
		if Deck.get_rank(move) == "A":
			return 4
