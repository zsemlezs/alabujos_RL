# Public libraries
import numpy as np
import random
import time
from tqdm.notebook import tqdm
import sys, os

# Custom libraries
import agent as ag
import state_action_reward as sar


def block_print():
    sys.__stdout__ = sys.stdout
    sys.stdout = open(os.devnull, "w")


def enable_print():
    sys.stdout = sys.__stdout__


def bold(string):
    chr_start = "\033[1m"
    chr_end = "\033[0m"
    print(chr_start + string + chr_end)


def underline(string):
    chr_start = "\033[4m"
    chr_end = "\033[0m"
    print(chr_start + string + chr_end)


# Card class
class Card(object):
    def __init__(self, c, v):
        self.color = c
        self.value = v

    def evaluate_card(self, open_c):
        if (self.color == open_c):
            return True

    def print_card(self):
        return str(self.color) + " " + str(self.value)

    def show_card(self):
        print (self.color, self.value)

# Deck class
class Deck(object):
    def __init__(self):
        self.cards = list()
        self.build()
        self.shuffle()

    def build(self):
        colors = ["PIR", "ZOL", "TOK", "MAK"]
        cards_32 = [Card(c, v) for c in colors for v in range(1, 9)]

        for card in cards_32:
            self.cards.append(card)

    def shuffle(self):
        random.shuffle(self.cards)

    def draw_from_deck(self):
        return self.cards.pop()

class Player(object):
    """
    Player consists of a list of cards representing a players hand cards.
    Player can have a name, hand, playable hand. Thereform the players' state can be determined.
    """

    def __init__(self, name):
        self.name = name
        self.hand = list()
        self.hand_play = list()
        self.card_play = 0
        self.points = 0
        self.state = dict()
        self.actions = dict()
        self.action = 0
        self.disced_deck = list()
        self.order = 0
        agent.prev_state = 0

    def clear_disc(self):
        self.disced_deck.clear()

    def points_calc(self):
        for i in self.disced_deck:
            if i.color == 'PIR':
                if i.value == 8:
                    self.points += 20
                else:
                    self.points += 10
            if i.color == 'MAK' and i.value == 6:
                self.points += 40

    def evaluate_hand(self, card_open):

        self.hand_play.clear()
        # First, see that the card_open is still zero - this results that the player could play all card
        print('Card open color:', card_open.color)
        print('Card open value:', card_open.value)
        if card_open.color == 'INIT' and card_open.value == 0:
            for card in self.hand: self.hand_play.append(card)

        # See if there are cards played
        else:
            for card in self.hand:
                if card.evaluate_card(card_open.color):
                    self.hand_play.append(card)

            if len(self.hand_play) == 0: #if the playable hand is empty, that means no color matching -> all card from the hand is playable
                for card in self.hand: self.hand_play.append(card)

        self.show_hand_play()

    def draw(self, deck):

        for i in range(0, 8):
            card = deck.draw_from_deck()
            self.hand.append(card)
            print(f'{self.name} draws {card.print_card()}')

    def identify_state(self, card_open):
        """
        The state of the player is identified by looping through players' hand for each property of the state.
        """

        norm_cards = {"PIR": 2, "ZOL": 2, "TOK": 2, "MAK": 2}

        self.state = dict()
        self.state["OPEN"] = card_open.color

        # (1) State properties: normal hand cards
        for key, val in zip(norm_cards.keys(), norm_cards.values()):
            self.state[key] = min(
                [1 if (card.color == key) and (card.value in range(1, 9)) else 0 for card in self.hand].count(1), val)

        # (2) State properties: normal playable cards
        for key, val in zip(norm_cards.keys(), norm_cards.values()):
            self.state[key + "#"] = min(
                [1 if (card.color == key) and (card.value in range(1, 9)) else 0 for card in self.hand_play].count(1),
                val - 1)

    def identify_action(self):
        """
        All actions are evaluated if they are available to the player, dependent on his hand and card_open.
        """

        norm_cards = {"PIR": 2, "ZOL": 2, "TOK": 2, "MAK": 2}

        # (1) Action properties: normal playable cards
        for key in norm_cards.keys():
            self.actions[key] = min(
                [1 if (card.color == key) and (card.value in range(1, 9)) else 0 for card in self.hand_play].count(1),
                1)


    def play_agent(self, card_open):
        """
        Reflecting a players' intelligent move supported by the RL-algorithm, that consists of:
            - Identification of the players' state and available actions
            - Choose card_played
            - Update Q-values in case of TD

        Required parameters:
            - deck as deck
            - card_open as card
        """
        self.evaluate_hand(card_open)

        # Identify state & actions for action selection
        self.identify_state(card_open)
        self.identify_action()

        # Agent selects action
        self.action = agent.step(self.state, self.actions)

        # Selected action searches corresponding card

        # (1) Playing normal card with different color
        if (self.action in ["PIR", "ZOL", "TOK", "MAK"]) and (self.action != card_open.color):
            for card in self.hand:
                if (card.color == self.action) and (card.value in range(1, 9)):
                    break

        # (2) Playing normal card with same color
        elif (self.action in ["PIR", "ZOL", "TOK", "MAK"]) and (self.action == card_open.color):
            for card in self.hand:
                if (card.color == self.action) and (card.value in range(1, 9)):
                    break


        # Selected card is played
        self.card_play = card
        self.hand.remove(card)
        self.hand_play.pop()
        print(f'\n{self.name} plays {card.print_card()}')


        # Update Q Value
        if algorithm == "q-learning":
            agent.update(self.state, self.action)

    def play_rand(self, card_open):
        """
        Reflecting a players' random move, that consists of:
            - Shuffling players' hand cards
            - Lopping through hand cards and choosing the first available hand card to be played
            - Remove card from hand & replace card_open with it

        Required parameters: deck as deck
        """
        self.evaluate_hand(card_open)
        random.shuffle(self.hand_play)
        for card in self.hand:
            if card == self.hand_play[-1]:
                self.card_play = card
                self.hand.remove(card)
                self.hand_play.pop()
                print(f'\n{self.name} plays {card.print_card()}')
                break


    def show_hand(self):
        underline(f'\n{self.name}s hand:')
        for card in self.hand:
            card.show_card()

    def show_hand_play(self):
        underline(f'\n{self.name}s playable hand:')
        for card in self.hand_play:
            card.show_card()

class Turn(object):
    """
    Captures the process of a turn, that consists of:
        - Initialization of hand cards and open card before first turn
        - Chosen action by player
        - Counter action by oposite player in case of PL2 or PL4
    """

    def __init__(self, player_1, player_2, player_3, player_4):
        """
        Turn is initialized with standard deck, players and an open card
        """

        self.player_1 = player_1
        self.player_2 = player_2
        self.player_3 = player_3
        self.player_4 = player_4
        self.disc = list()
        #self.start_up()
        self.number_of_turn = 0
        self.card_open = 0
        self.losing_card = 0
        self.loser = 0

    def action(self, player):
        """
        Only reflecting the active players' action if he hand has not won yet.
        Only one player is leveraging the RL-algorithm, while the other makes random choices.
        """

        player_act = player

        if len(self.disc) == 0:
            player_act.play_rand(self.card_open) #to be updated to play optimal opening card
            self.card_open = player_act.card_play
            self.losing_card = player_act.card_play
        else:
            if player_act == self.player_1:
                player_act.play_agent(self.card_open)
            else:
                player_act.play_rand(self.card_open)

        # Check who is the loser
        if len(self.disc) == 0:
            self.loser = player_act
        else:
            if self.card_open.color == player_act.card_play.color:
                if self.losing_card.value < player_act.card_play.value:
                    self.loser = player_act

        print('Current loser midturn',self.loser.name)

        self.disc.append(player_act.card_play)

    def clear_disc(self):
        self.disc.clear()


class Game(object):
    """
    A game reflects an iteration of turns, until one player fulfills the winning condition of 0 hand cards.
    It initialized with two players and a turn object.
    """

    def __init__(self, player_1_name, player_2_name, player_3_name, player_4_name, comment):

        if comment == False: block_print()

        self.player_1 = Player(player_1_name)
        self.player_2 = Player(player_2_name)
        self.player_3 = Player(player_3_name)
        self.player_4 = Player(player_4_name)
        self.turn = Turn(player_1=self.player_1, player_2=self.player_2,
                         player_3=self.player_3, player_4=self.player_4)

        self.turn_no = 0
        self.winner = 0

        # With each new game the starting player is switched, in order to make it fair
        while self.winner == 0:
            self.turn_no += 1
            bold(f'\n---------- TURN {self.turn_no} ----------')
            # Building the deck at the start of every round
            self.deck = Deck()
            self.deck.shuffle()
            # Every player draw from the deck
            self.player_1.draw(self.deck)
            self.player_2.draw(self.deck)
            self.player_3.draw(self.deck)
            self.player_4.draw(self.deck)
            # Sitting order
            self.player_1.order = 0
            self.player_2.order = 1
            self.player_3.order = 2
            self.player_4.order = 3

            # Defining the starter player
            if self.turn_no % 4 == 0:
                player_act = self.player_1
                player_sec = self.player_2
                player_thi = self.player_3
                player_four = self.player_4
            elif self.turn_no % 4 == 1:
                player_act = self.player_2
                player_sec = self.player_3
                player_thi = self.player_4
                player_four = self.player_1
            elif self.turn_no % 4 == 2:
                player_act = self.player_3
                player_sec = self.player_4
                player_thi = self.player_1
                player_four = self.player_2
            elif self.turn_no % 4 == 3:
                player_act = self.player_4
                player_sec = self.player_1
                player_thi = self.player_2
                player_four = self.player_3
            self.turn.number_of_turn = 0

            while self.turn.number_of_turn != 8:
                bold(f'\n---------- SUB-TURN {self.turn.number_of_turn+1} ----------')
                self.turn.card_open = Card('INIT',0) #to reset the card_open at the start of every sub-round
                self.turn.action(player=player_act)
                self.turn.action(player=player_sec)
                self.turn.action(player=player_thi)
                self.turn.action(player=player_four)

                if self.turn.loser.name == self.player_1.name:
                    for card in self.turn.disc: self.player_1.disced_deck.append(card)
                    self.turn.clear_disc()
                    player_act = self.player_1
                    player_sec = self.player_2
                    player_thi = self.player_3
                    player_four = self.player_4
                elif self.turn.loser.name == self.player_2.name:
                    for card in self.turn.disc: self.player_2.disced_deck.append(card)
                    self.turn.clear_disc()
                    player_act = self.player_2
                    player_sec = self.player_3
                    player_thi = self.player_4
                    player_four = self.player_1
                elif self.turn.loser.name == self.player_3.name:
                    for card in self.turn.disc: self.player_3.disced_deck.append(card)
                    self.turn.clear_disc()
                    player_act = self.player_3
                    player_sec = self.player_4
                    player_thi = self.player_1
                    player_four = self.player_2
                elif self.turn.loser.name == self.player_4.name:
                    for card in self.turn.disc: self.player_4.disced_deck.append(card)
                    self.turn.clear_disc()
                    player_act = self.player_4
                    player_sec = self.player_1
                    player_thi = self.player_2
                    player_four = self.player_3

                self.turn.number_of_turn +=1
                print('Loser end of the turn:', self.turn.loser.name)



            # Points calc
            self.player_1.points_calc()
            self.player_2.points_calc()
            self.player_3.points_calc()
            self.player_4.points_calc()

            # Print points
            print('Player 1 points:', self.player_1.points)
            print('Player 2 points:', self.player_2.points)
            print('Player 3 points:', self.player_3.points)
            print('Player 4 points:', self.player_4.points)

            # Clear disced cards
            self.player_1.clear_disc()
            self.player_2.clear_disc()
            self.player_3.clear_disc()
            self.player_4.clear_disc()
            # Check loose
            if check_loose(self.player_1) == True:
                self.winner = check_winner(self.player_1,self.player_2,self.player_3,self.player_4)
                print('Name of the winner player:',self.winner.name)
                break
            if check_loose(self.player_2) == True:
                self.winner = check_winner(self.player_1,self.player_2,self.player_3,self.player_4)
                print('Name of the winner player:',self.winner.name)
                break
            if check_loose(self.player_3) == True:
                self.winner = check_winner(self.player_1,self.player_2,self.player_3,self.player_4)
                print('Name of the winner player:',self.winner.name)
                break
            if check_loose(self.player_4) == True:
                self.winner = check_winner(self.player_1,self.player_2,self.player_3,self.player_4)
                print('Name of the winner player:',self.winner.name)
                break

        #self.player_1.identify_state(self.turn.card_open)
        #agent.update(self.player_1.state, self.player_1.action)

        if comment == False: enable_print()


def tournament(iterations, algo, comment, agent_info):
    """
    A function that iterates various Games and outputs summary statistics over all executed simulations.
    """

    timer_start = time.time()

    # Selection of algorithm
    global agent, algorithm
    algorithm = algo

    if algo == "q-learning":
        agent = ag.QLearningAgent()
    else:
        agent = ag.MonteCarloAgent()

    winners, turns, coverage = list(), list(), list()
    agent.agent_init(agent_info)

    for i in tqdm(range(iterations)):
        time.sleep(0.01)

        game = Game(player_1_name="Bernhard",
                    player_2_name="Magdalena",
                    player_3_name="Yusuf",
                    player_4_name="Petrov",
                    comment=comment)

        winners.append(game.winner)
        turns.append(game.turn_no)
        coverage.append((agent.q != 0).values.sum())

    # Timer
    timer_end = time.time()
    timer_dur = timer_end - timer_start
    print(f'Execution lasted {round(timer_dur / 60, 2)} minutes ({round(iterations / timer_dur, 2)} games per second)')

    return winners, turns, coverage


# Definitions of losing or winning
def check_loose(player):
    if player.points >= 500:
        return True

def check_winner(player_1,player_2,player_3,player_4):
    winner = 0
    if player_2.points > player_1.points and player_3.points > player_1.points and player_4.points > player_1.points:
        winner = player_1
    if player_1.points > player_2.points and player_3.points > player_2.points and player_4.points > player_2.points:
        winner = player_2
    if player_1.points > player_3.points and player_2.points > player_3.points and player_4.points > player_3.points:
        winner = player_3
    if player_1.points > player_4.points and player_2.points > player_4.points and player_3.points > player_4.points:
        winner = player_4

    if winner == 0:
        if player_2.points >= player_1.points and player_3.points >= player_1.points and player_4.points >= player_1.points:
            winner = player_1
        if player_1.points >= player_2.points and player_3.points >= player_2.points and player_4.points >= player_2.points:
            winner = player_2
        if player_1.points >= player_3.points and player_2.points >= player_3.points and player_4.points >= player_3.points:
            winner = player_3
        if player_1.points >= player_4.points and player_2.points >= player_4.points and player_3.points >= player_4.points:
            winner = player_4
    return winner

