from api import State, util
import pickle
import os.path
from argparse import ArgumentParser
import time
import sys
import random
from itertools import chain
from sklearn.neural_network import MLPRegressor
import joblib


#################################################################################################################################################################################################################################
#FUNCTIONS
#########################################################################################################################################################################################

def get_feature(state, move):
    # return a list [move/int,state.p1,...]
    feature_set = []

    # Add player 1's points to feature set
    p1_points = state.get_points(1)

    # Add player 2's points to feature set
    p2_points = state.get_points(2)

    # Add player 1's pending points to feature set
    p1_pending_points = state.get_pending_points(1)

    # Add player 2's pending points to feature set
    p2_pending_points = state.get_pending_points(2)

    # Get trump suit
    trump_suit = state.get_trump_suit()

    # Add phase to feature set
    phase = state.get_phase()

    # Add stock size to feature set
    stock_size = state.get_stock_size()

    # Add opponent's played card to feature set
    opponents_played_card = state.get_opponents_played_card()

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

    # Return feature set
    return feature_set

def evaluate(state, player):
    score = 0.0

    for _ in range(1000): #self.__num_samples
        st = state.clone()
        # Do some random moves
        for i in range(10): #self.__depth
            if st.finished():
                break
            st = st.next(random.choice(st.moves()))
        score += heuristic(st, player)
    return score / float(1000) #self.__num_samples


def heuristic(state, player):
    return util.ratio_points(state, player)


#################################################################################################################################################################################################################################
#CREATING A DATASET
#########################################################################################################################################################################################


def create_dataset(path, states=10000, phase=1):

    data = []
    target = []

    # For progress bar
    bar_length = 30
    start = time.time()

    seed = 1
    for g in range(states - 1):
        # For progress bar
        if g % 10 == 0:
            percent = 100.0 * g / states
            sys.stdout.write('\r')
            sys.stdout.write(
                "Generating dataset: [{:{}}] {:>3}%".format('=' * int(percent / (100.0 / bar_length)), bar_length,
                                                            int(percent)))
            sys.stdout.flush()

        # Randomly generate a state object starting in specified phase.
        state = State.generate(seed, phase=phase)
        seed +=1
        state_vectors = []

        # Give the state a signature if in phase 1, obscuring information that a player shouldn't see.
        given_state = state.clone(signature=state.whose_turn()) if state.get_phase() == 1 else state  # TODO: che for problems

        # chose 1 possible random move
        possible_moves = given_state.moves()
        for i in possible_moves:
            if i[1] != None:
                possible_moves.remove(i)
        chosen_move = random.choice(possible_moves)

        # Add the features representation of a state to the state_vectors array
        #print("--",given_state)
        temp = get_feature(given_state, chosen_move)
        state_vectors.append(temp)

        #####################################################

        # evaluate move
        sample_state = given_state.make_assumption()

        score = evaluate(sample_state.next(chosen_move), state.whose_turn())

        # append the features to data and the score to target
        for state_vector in state_vectors:
            data.append(state_vector)
            target.append(score)

    with open(path, 'wb') as output:
        pickle.dump((data, target), output, pickle.HIGHEST_PROTOCOL)

    # For printing newline after progress bar
    print("\nDone. Time to generate dataset: {:.2f} seconds".format(time.time() - start))

    return data, target


#################################################################################################################################################################################################################################
#CREATING AND FITTING A MODEL
#########################################################################################################################################################################################


## Parse the command line options
parser = ArgumentParser()

parser.add_argument("-d", "--dset-path",
                    dest="dset_path",
                    help="Optional dataset path",
                    default="datasetproject.pkl")

parser.add_argument("-m", "--model-path",
                    dest="model_path",
                    help="Optional model path. Note that this path starts in bots/ml/ instead of the base folder, like dset_path above.",
                    default="model1.pkl")

parser.add_argument("-o", "--overwrite",
                    dest="overwrite",
                    action="store_true",
                    help="Whether to create a new dataset regardless of whether one already exists at the specified path.")

parser.add_argument("--no-train",
                    dest="train",
                    action="store_false",
                    help="Don't train a model after generating dataset.")

options = parser.parse_args()


if options.overwrite or not os.path.isfile(options.dset_path):
    create_dataset(options.dset_path, states=10000, phase=1)

if options.train:
    hidden_layer_sizes = (500, 10)
    learning_rate = 0.0001
    regularization_strength = 0.0001
    start = time.time()

    print("Starting training phase...")

    with open(options.dset_path, 'rb') as output:
        data, target = pickle.load(output)

    # Train a neural network
    learner = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate,
                            alpha=regularization_strength, verbose=False, early_stopping=True, n_iter_no_change=15)

    model = learner.fit(data, target)

    score = learner.score(data, target)
    print("SCORE ->", score)

    # Check for class imbalance
    count = {}
    for t in target:
        if t not in count:
            count[t] = 0
        count[t] += 1


    joblib.dump(model, "./bots/projectbot/" + options.model_path)

    end = time.time()

    print('Done. Time to train:', (end - start) / 60, 'minutes.')

