from argparse import ArgumentParser
from api import State, util, engine
import random


def run_tournament(options):

    botnames = options.players.split(",")

    bots = []
    for botname in botnames:
        bots.append(util.load_player(botname))

    n = len(bots)
    wins = [0] * len(bots)
    matches = [(p1, p2) for p1 in range(n) for p2 in range(n) if p1 < p2]
    counters = [0] * len(bots)
    points = [0] * len(bots)

    totalgames = (n*n - n)/2 * options.repeats
    playedgames = 0

    seed = 10000
    print('Playing {} games:'.format(int(totalgames)))
    for a, b in matches:
        for r in range(options.repeats):

            if random.choice([True, False]):
                p = [a, b]
            else:
                p = [b, a]

            # Generate a state with a random seed
            state = State.generate(seed, phase=int(options.phase))

            winner, score = engine.play(bots[p[0]], bots[p[1]], state, options.max_time * 1000, verbose=options.verbose, fast=options.fast)

            # Keep track of wins
            if winner is not None:
                winner = p[winner - 1]
                wins[winner] += score
                points[winner] += score
                if wins[winner] >= 7:
                    counters[winner] +=1
                    wins[winner] = 0

            playedgames += 1
            print('Played {} out of {:.0f} games ({:.0f}%): {} \r'.format(playedgames, totalgames, playedgames/float(totalgames) * 100, points))

            seed += 1

    print('Results:')
    for i in range(len(bots)):
        print('    bot {}: {} points'.format(bots[i], points[i]))
        print('    bot {}: {} GAMES'.format(bots[i], counters[i]))


if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-s", "--starting-phase",
                        dest="phase",
                        help="Which phase the game should start at.",
                        default=1)

    parser.add_argument("-p", "--players",
                        dest="players",
                        help="Comma-separated list of player names (enclose with quotes).",
                        default="rand,bully,rdeep")

    parser.add_argument("-r", "--repeats",
                        dest="repeats",
                        help="How many matches to play for each pair of bots",
                        type=int, default=10)

    parser.add_argument("-t", "--max-time",
                        dest="max_time",
                        help="maximum amount of time allowed per turn in seconds (default: 5)",
                        type=int, default=5)

    parser.add_argument("-f", "--fast",
                        dest="fast",
                        action="store_true",
                        help="This option forgoes the engine's check of whether a bot is able to make a decision in the allotted time, so only use this option if you are sure that your bot is stable.")

    parser.add_argument("-v", "--verbose",
                        dest="verbose",
                        action="store_true",
                        help="Print verbose information")

    options = parser.parse_args()

    run_tournament(options)
