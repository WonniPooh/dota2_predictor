
import Heroes
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import operator

#class hero:
#   def _init_(self, name, hero_id, side):
#       self.name = name
#       self.hero_id = hero_id
#       self.side = side

######################################################################################################################################################

GLOBALHEROWINRATES_SOLO = np.zeros(len(Heroes.heroes), dtype = (float, 6))                              # 1-d array of solo winrates
GLOBALHEROWINRATES_DUO = np.zeros((len(Heroes.heroes), len(Heroes.heroes)), dtype = (float, 6))         # 2-d array for duo winrates
                                                                                                        # GLOBALHEROWINRATES_DUO is a bottom-triangle square matrix
GLOBALHEROWINRATES_ENEMIES = np.zeros((len(Heroes.heroes), len(Heroes.heroes)), dtype = (float, 6))     # matrix of winrates when heroes play 1vs1

######################################################################################################################################################
class dota2match:
    def __init__(self, game_id, data, winner, cluster_id, game_mode, game_type, radiant_team, dire_team):
        self.game_id = game_id
        self.data = data
        self.winner = winner    # TRUE for radiant or FALSE for dire
        self.cluster_id = cluster_id    # server id (or idk)
        self.game_mode = game_mode
        self.game_type = game_type
        self.radiant_team = radiant_team    # [ (hero_id, is_sup, lane), ... ] - from Heroes.heroes
                                                # is_sup - if hero playes a supportive role (TRUE FALSE)
                                                # line: 0-roaming, 1-safelane, 2-mid, 3-offlane, 4-jungle
        self.dire_team = dire_team  # same as heroes_radiant
    def show(self):
        if self.winner == 'TRUE':
            print "Radiant team victiry      ", self.game_id, self.data
        else:
            print "Dire team victiry      ", self.game_id, self.data
        hr = []
        for h in self.radiant_team:
            hr.append(Heroes.heroes[h[0]]["name"])
            hr.append(h)
        hd = []
        for h in self.dire_team:
            hd.append(Heroes.heroes[h[0]]["name"])
            hd.append(h)
        print hr, hd
        #print self.heroes_radiant, self.heroes_dire

######################################################################################################################################################
def CreateHeroesDataBase():     # checked
    """
    converts information about heroes from .txt format to a python module
    P.S. after executing this function move the new Heroes.py into directory with this code
    """
    infile = "Dataset/Heroes.txt"
    outfile = "Dataset/Heroes.py"

    fin = open(infile)
    fout = open(outfile, "w+")

    fout.write("heroes = (\n")
    for line in fin:
        word = line.split()
        fout.write("\t\t{\n")
        fout.write("\t\t\t'name': {0},\n".format(word[1]))
        fout.write("\t\t\t'id': {0},\n".format(word[0]))
        fout.write("\t\t\t'push': {0},\n".format(word[6]))
        fout.write("\t\t\t'nuker': {0},\n".format(word[2]))
        fout.write("\t\t\t'disabler': {0},\n".format(word[3]))
        fout.write("\t\t\t'initiator': {0},\n".format(word[7]))
        fout.write("\t\t\t'carry': {0},\n".format(word[8]))
        fout.write("\t\t\t'escape': {0},\n".format(word[5]))
        fout.write("\t\t\t'durable': {0},\n".format(word[9]))
        fout.write("\t\t\t'jungler': {0},\n".format(word[4]))
        fout.write("\t\t\t'outpush': {0},\n".format(word[10]))
        fout.write("\t\t\t'complexity': {0},\n".format(word[11]))
        fout.write("\t\t\t'melee': '{0}',\n".format(word[12]))

        if word[0] == '120':
            fout.write("\t\t}\n)")
        else:
            fout.write("\t\t},\n")

    fin.close()
    fout.close()
    #ID   NAME               N D J E P I C S O Complexity Melee  

######################################################################################################################################################
def LoadDota2Matches(filepath):     # checked
    """
    loads information about matches from a file
    filepath - path to a file

    return value - list of objects woth a tyupe dota2match
    """
    dota2matches = [];
    with open(filepath, "r") as f:
        for row in f:
            hr = []
            hd = []
            words = row.split()

            if words[1] < "'2017-10-01'":
                continue

            lane_counter = 0    # if there is info about hero roles
            for i in range(10):
                if words[3 + i * 4 + 3] == 'NULL':
                    words[3 + i * 4 + 3] = -1
                    lane_counter += 1
                if words[3 + i * 4 + 1] == 'TRUE':
                    hr.append((int(words[3 + i * 4]), words[3 + i * 4 + 2], int(words[3 + i * 4 + 3])))     # hero_id, is_sup, lane
                else:
                    hd.append((int(words[3 + i * 4]), words[3 + i * 4 + 2], int(words[3 + i * 4 + 3])))
            hr  =sorted(hr, key = lambda h: h[0])
            hd = sorted(hd, key = lambda h: h[0])

            if lane_counter > 0:    # if some heroes don't have info about their role
                continue

            m = dota2match(words[0], words[1], words[2], -1, -1, -1, hr, hd)
            dota2matches.append(m)
    return dota2matches

######################################################################################################################################################
def GenerateSubGroups(radiant_team, dire_team):     # checked
    """
    radiant_team - list of 5 heroes: [(hero_id, is_sup, lane), ...]
    dire_team - same as radiant_team

    return value - list of subgroups: [([(hero_id, is_sup, lane), ...], [(hero_id, is_sup, lane), ...]), ...], hero = (hero_id, is_sup, lane)
    """

    # first approximation: subgroups = lines. first 15 minutes players spend on their lines. they gain power (or feed their oppenent)
    # they go push as a team of 4-5 only after minute 15 (in 99% of games)

    if len(radiant_team) != 5 or len(dire_team) != 5:
        raise ValueError("Wrong amount of players in teams. should be equal to 5")

    radiant_group_top = [h for h in radiant_team if h[2] == 3 or h[2] == 0]
    radiant_group_mid = [h for h in radiant_team if h[2] == 2 or h[2] == 0]
    radiant_group_bot = [h for h in radiant_team if h[2] == 1 or h[2] == 0]
    radiant_group_jungle = [h for h in radiant_team if h[2] == 4]

    dire_group_top = [h for h in dire_team if h[2] == 1 or h[2] == 0]
    dire_group_mid = [h for h in dire_team if h[2] == 2 or h[2] == 0]
    dire_group_bot = [h for h in dire_team if h[2] == 3 or h[2] == 0]
    dire_group_jungle = [h for h in dire_team if h[2] == 4]

    #return [(radiant_group_top, dire_group_top), (radiant_group_mid, dire_group_mid), (radiant_group_bot, dire_group_bot), (radiant_group_jungle, []), ([], dire_group_jungle)]
    return [(radiant_group_top, dire_group_top), (radiant_group_mid, dire_group_mid), (radiant_group_bot, dire_group_bot)]

######################################################################################################################################################
def SetEasyStrength(group_a, group_b):      # checked
    """
    group_a - list of tuples: [(hero_id, is_sup, lane), ...] , look hero_id's in Heroes.heroes
    group_b - same as group_a

    return value - tuple: (strength_1, strength_2), strength_i  - unnormalized chance to win (float or integer)
    or
    return value - tuple: (win_chance_1, win_chance_2), win_chance_i - chance to win. summ = 1
    mb second one is better. we will see
    """
    global GLOBALHEROWINRATES_SOLO
    global GLOBALHEROWINRATES_DUO

    all_hero_parameters = ['push', 'nuker', 'disabler', 'initiator', 'carry', 'escape', 'durable', 'jungler', 'outpush']
    mask = ['outpush', 'nuker', 'initiator', 'disabler']
    
    strength_a = [GLOBALHEROWINRATES_SOLO[h[0]][0] for h in sorted(group_a, key = lambda h: h[1])]
    strength_b = [GLOBALHEROWINRATES_SOLO[h[0]][0] for h in sorted(group_b, key = lambda h: h[1])]
    #indexes_a = [[Heroes.heroes[h[0]][k]] for h in sorted(group_a, key = lambda h: h[1]) for k in mask]
    #indexes_b = [[Heroes.heroes[h[0]][k]] for h in sorted(group_b, key = lambda h: h[1]) for k in mask]
    
    summ_strength_a = 1
    for w in strength_a:
        summ_strength_a = summ_strength_a * (1 - w)

    summ_strength_b = 1
    for w in strength_b:
        summ_strength_b = summ_strength_b * (1 - w)

    result = (1 - summ_strength_a, 1 - summ_strength_b)
    return result

def SetHardStrength(group_a, group_b, balance = 0):
    """
    same as function above

    balance value shows if we use info, that radiant wins more often
    """
    if balance != 0 and balance != 1:
        raise ValueError("wrong value for balance variable. Use '0' for not considering radiant_wr > dire_wr. Use '1' otherwises")

    global GLOBALHEROWINRATES_SOLO
    global GLOBALHEROWINRATES_DUO

    strength_a = 0
    strength_b = 0

    if len(group_a) == 0:
        pass
    elif len(group_a) == 1:
        strength_a = GLOBALHEROWINRATES_SOLO[group_a[0][0]][balance * 2]
    elif len(group_a) == 2:
        group_a = sorted(group_a, key = lambda h: h[0])
        strength_a = GLOBALHEROWINRATES_DUO[group_a[1][0]][group_a[0][0]][balance * 2]
        if strength_a == 0:
            strength_a = GLOBALHEROWINRATES_DUO[group_a[0][0]][group_a[1][0]][balance * 2]
    elif len(group_a) == 3:
        group_a = sorted(group_a, key = lambda h: h[1])
        strength_a = GLOBALHEROWINRATES_DUO[ group_a[1][0] ][ group_a[2][0] ][balance * 2]
        if strength_a == 0:
            strength_a = GLOBALHEROWINRATES_DUO[group_a[2][0]][group_a[1][0]][balance * 2]
    elif len(group_a) > 3:
        #raise ValueError("How did you managed to find more then 3 people on the same lane!?")
        #strength_a = 1
        pass
        strength_a = 0.52
    if len(group_b) == 0:
        pass
    elif len(group_b) == 1:
        strength_b = GLOBALHEROWINRATES_SOLO[group_b[0][0]][balance * 4]
    elif len(group_b) == 2:
        group_b = sorted(group_b, key = lambda h: h[0])
        strength_b = GLOBALHEROWINRATES_DUO[group_b[1][0]][group_b[0][0]][balance * 4]
        if strength_b == 0:
            strength_b = GLOBALHEROWINRATES_DUO[group_b[0][0]][group_b[1][0]][balance * 4]
    elif len(group_b) == 3:
        group_b = sorted(group_b, key = lambda h: h[1])
        strength_b = GLOBALHEROWINRATES_DUO[group_b[1][0]][group_b[2][0]][balance * 4]
        if strength_b == 0:
            strength_b = GLOBALHEROWINRATES_DUO[group_b[2][0]][group_b[1][0]][balance * 4]
    elif len(group_b) > 3:
        #raise ValueError("How did you managed to find more then 3 people on the same lane!?")
        #strength_b = 1
        pass
        strength_b = 0.48

    result = (strength_a * len(group_a), strength_b * len(group_b))
    return result

######################################################################################################################################################
def MergeGroups(all_groups, groups_winchances):
    """
    all_groups - list. look at return value of GenerateSubGroups()
    groups_winchances - list. look at return value of SetEasyStrength()

    return value - tuple: (strength_1, strength_2), strength_i  - unnormalized chance to win
    or
    return value - tuple: (win_chance_1, win_chance_2), win_chance_i - chance to win. summ = 1
    well, same as return value of SetEasyStrength, but for all heroes from groups when they fight together
    """
    pass

######################################################################################################################################################
def GetHeroesFeatures(heroes_group, keys = Heroes.heroes[0].keys()):    # checked
    """
    heroes_group - list of heroes: [(hero_id, is_sup, lane), ...] , look hero_id's in Heroes.heroes
    keys - list of keys. a.k.a. maks

    return value - dictionary with features' values of a whole team
    """

    # linear summ
    result = dict(Heroes.heroes[1])  # equal to result = dict(Heroes.heroes[hero_id])
    del(result['name'])
    del(result['melee'])
    del(result['id'])
    for k in result.keys():
        if k not in keys:
            del(result[k])
        else:
            result[k] = 0

    for h in heroes_group:
        new_portion = dict(Heroes.heroes[h[0]])
        for i in result.keys():
            result[i] += new_portion[i]

    return result

######################################################################################################################################################
def SumHeroesFeatures(features_dict, keys = Heroes.heroes[0].keys()):     # checked
    """
    feature_dict - look at return value of GetFeatures(heroes_group)
    keys - list of keys to sum

    return value - integer, sum of features with a mask keys
    """
    result = 0
    for k in features_dict.keys():     # this doesn't work :-(
        if k in keys:
            result += features_dict[k]

    return result

######################################################################################################################################################
def GetFeatures(match, mode = -1):     # currently this function is a rubbish bin. sorry
    """
    match - object with a type 'dota2match'

    return value - list of features (integer ordered values)
    """
    global GLOBALHEROWINRATES_SOLO
    global GLOBALHEROWINRATES_DUO
    result = 0

    if mode == 0:    # method 0. features = random numbers
        result = [random.randint(0, 5) for i in range(5)]
        return result
    
    elif mode == 1:    # method 1. features = summ of masked parameters for the whole team
        all_hero_parameters = ['push', 'nuker', 'disabler', 'initiator', 'carry', 'escape', 'durable', 'jungler', 'outpush']
        mask = ['outpush', 'nuker', 'initiator', 'disabler']
        rt = GetHeroesFeatures(match.radiant_team, mask)
        dt = GetHeroesFeatures(match.dire_team, mask)
        result = [rt[k] for k in rt.keys()] + [dt[k] for k in dt.keys()]
        return result
    
    elif mode == 2:    # method 2. features = winrates for all orderd team members
        rt = [GLOBALHEROWINRATES_SOLO[h[0]][2] for h in sorted(match.radiant_team, key = lambda h: h[2])]
        dt = [GLOBALHEROWINRATES_SOLO[h[0]][4] for h in sorted(match.dire_team, key = lambda h: h[2])]
        result = rt + dt
        return result

    elif mode == 3:   # method 3. heroes' interection is taken into consideration
                     # currently it's not a method - it's a rubbbish bin
        subgroups = GenerateSubGroups(match.radiant_team, match.dire_team)
        features1 = [SetEasyStrength(sg[0], sg[1]) for sg in subgroups]
        rt1 = [f[0] for f in features1]
        dt1 = [f[1] for f in features1]
        rt = [GLOBALHEROWINRATES_SOLO[h[0]][0] for h in sorted(match.radiant_team, key = lambda h: h[2])]
        dt = [GLOBALHEROWINRATES_SOLO[h[0]][0] for h in sorted(match.dire_team, key = lambda h: h[2])]

        mask = ['outpush', 'nuker', 'initiator', 'disabler']
        #rt3 = [SumHeroesFeatures(GetHeroesFeatures(match.radiant_team, mask), mask)]
        #dt3 = [SumHeroesFeatures(GetHeroesFeatures(match.dire_team, mask), mask)]
        rt3 = [GetHeroesFeatures(match.radiant_team, mask)[k] for k in mask]
        dt3 = [GetHeroesFeatures(match.dire_team, mask)[k] for k in mask]

        result = rt + dt + rt3 + dt3 #+ rt3 + dt3
        return result

    elif mode == 4:    # metod 4. features = all masked parameters for every hero in match
        all_hero_parameters = ['push', 'nuker', 'disabler', 'initiator', 'carry', 'escape', 'durable', 'jungler', 'outpush']
        mask = ['outpush', 'nuker', 'initiator', 'disabler']
        rt = [GetHeroesFeatures([h], mask) for h in match.radiant_team]
        dt = [GetHeroesFeatures([h], mask) for h in match.dire_team]
        result = [rt[i][k] for i in range(5) for k in rt[0].keys()] + [dt[i][k] for i in range(5) for k in dt[0].keys()]
        return result

    elif mode == 5:
        balance = 0
        subgroups = GenerateSubGroups(match.radiant_team, match.dire_team)
        features1 = [SetHardStrength(sg[0], sg[1], balance) for sg in subgroups]
        rt1 = [f[0] for f in features1]
        dt1 = [f[1] for f in features1]

        match.radiant_team.sort(key = operator.itemgetter(2, 1))
        match.dire_team.sort(key = operator.itemgetter(2, 1))
        rt2 = [GLOBALHEROWINRATES_SOLO[h[0]][2 * balance] for h in match.radiant_team]
        dt2 = [GLOBALHEROWINRATES_SOLO[h[0]][4 * balance] for h in match.dire_team]

        result = rt1 + dt1 + rt2 + dt2
        return result
    elif mode == 6:
        balance = 0
        match.radiant_team.sort(key = operator.itemgetter(2, 1))
        match.dire_team.sort(key = operator.itemgetter(2, 1))
        result1 = [ GLOBALHEROWINRATES_ENEMIES[h1[0]][h2[0]][2 * balance] for h1 in match.radiant_team for h2 in match.dire_team ]

        mask = ['outpush', 'nuker']
        rt = [GetHeroesFeatures([h], mask) for h in match.radiant_team]
        dt = [GetHeroesFeatures([h], mask) for h in match.dire_team]
        result2 = [rt[i][k] for i in range(5) for k in rt[0].keys()] + [dt[i][k] for i in range(5) for k in dt[0].keys()]
        
        return result1 + result2
    else:
        raise ValueError("wrong value for argument mode. suggested: 0, 1, 2. current: {0}".format(mode))

        """
        s.sort(key = operator.itemgetter(1, 2))
        """

######################################################################################################################################################
def CheckWinRate(all_matches, group_a, group_b):
    """
    all_matches - list of objects with a type of 'dota2match'
    group_a - list of heroes: [(hero_id, is_sup, lane), ...] , look hero_id's in Heroes.heroes
    group_b - same as group_a

    return value - list [winrate, matches_played_with_this_heroes, wr_radiant, games_radiant, wr_dire, games_dire].  winrate of group_a against group_b
    """
    group_a = sorted(group_a, key = lambda h: h[0])
    group_b = sorted(group_b, key = lambda h: h[0])

    group_a = [h[0] for h in group_a]
    group_b = [h[0] for h in group_b]       # currently we don't analize hero roles, only their presence in game

    for a in group_a:
        if a in group_b:
            return [-1, 0, -1, 0, -1, 0]

    total_matches_radiant = 0
    group_a_wins_radiant = 0
    total_matches_dire = 0
    group_a_wins_dire = 0

    for m in all_matches:
        a_in_radiant = 1
        a_in_dire = 1
        b_in_radiant = 1
        b_in_dire = 1
        for a in group_a:
            if not a in [x[0] for x in m.radiant_team]:
                a_in_radiant = 0
                break
        if a_in_radiant == 0:
            for a in group_a:
                if not a in [x[0] for x in m.dire_team]:
                    a_in_dire = 0
                    break
        else: #then a_in_radiant == 1, so we need to check if b_in_dire == 1
            for b in group_b:
                if not b in [x[0] for x in m.dire_team]:
                    b_in_dire = 0
                    break
        if a_in_dire == 1: # then we need to check if b_in_radiant ==1
            for b in group_b:
                if not b in [x[0] for x in m.radiant_team]:
                    b_in_radiant = 0
                    break

        if a_in_radiant == 1 and b_in_dire == 1:
            if m.winner == 'TRUE':
                group_a_wins_radiant += 1
            total_matches_radiant += 1
        elif a_in_dire == 1 and b_in_radiant == 1:
            if m.winner == 'FALSE':
                group_a_wins_dire += 1
            total_matches_dire += 1
    #time to return
    if total_matches_radiant == 0 and total_matches_dire == 0:
        return [-1, 0, -1, 0, -1, 0]
    elif total_matches_radiant != 0 and total_matches_dire == 0:
        return [-1, 0, -1, 0, -1, 0]
    elif total_matches_radiant == 0 and total_matches_dire != 0:
        return [-1, 0, -1, 0, -1, 0]
    else:
        return [float(group_a_wins_radiant + group_a_wins_dire) / (total_matches_radiant + total_matches_dire), total_matches_radiant + total_matches_dire, float(group_a_wins_radiant) / total_matches_radiant, total_matches_radiant, float(group_a_wins_dire) / (total_matches_dire), total_matches_dire]

######################################################################################################################################################
def PredictWinNaive(all_matches, team_radiant, team_dire, max_group_size):
    """
    all_matches - list of objects with a type of 'dota2match'
    team_radiant - list of 5 heroes, [(hero_id, is_sup, lane), ...] , see hero_id's in dota2Heroes.heroes
    team_dire - same as team_radiant
    max_group_size - integer

    return value - prediction to team_radiant to win against team_dire
    """
    if max_group_size > 0:
        all_teams_a = [[a] for a in team_radiant]
        all_teams_b = [[b] for b in team_dire]
    if max_group_size > 1:
        all_teams_a += [[a1, a2] for a1 in team_radiant for a2 in team_radiant if a1 != a2]
        all_teams_b += [[b1, b2] for b1 in team_dire for b2 in team_dire if b1 != b2]
    if max_group_size > 2:
        all_teams_a += [[a1, a2, a3] for a1 in team_radiant for a2 in team_radiant for a3 in team_radiant if a1 != a2 and a1 != a3 and a2 != a3]
        all_teams_b += [[b1, b2, b3] for b1 in team_dire for b2 in team_dire for b3 in team_dire if b1 != b2 and b1 != b3 and b2 != b3]

    winrates = [CheckWinRate(all_matches, a, b)[0] for a in all_teams_a for b in all_teams_b]
    winrates_processed = [w for w in winrates if w >= 0]
    #return float(sum(winrates_processed)) / len(winrates_processed)
    return reduce(lambda x, y: x + y, winrates_processed) / len(winrates_processed)

def HandleGlobalVariables(all_matches, mode = 'all'):
    """
    function initializes globbal variables GLOBALHEROWINRATES
    function sets their values to 1-d and 2-d arrays
    """
    global GLOBALHEROWINRATES_SOLO
    global GLOBALHEROWINRATES_DUO
    global GLOBALHEROWINRATES_ENEMIES

    if mode == 'solo' or 'all':

        for a in GLOBALHEROWINRATES_SOLO:
            a = [0, 0, 0, 0, 0, 0]      # [winrate, matches_played_with_this_heroes, wr_radiant, games_radiant, wr_dire, games_dire]
        
        for m in all_matches:
            for  i in range(5):
                
                GLOBALHEROWINRATES_SOLO[m.radiant_team[i][0]][1] += 1
                GLOBALHEROWINRATES_SOLO[m.radiant_team[i][0]][3] += 1
                if m.winner == 'TRUE':
                    GLOBALHEROWINRATES_SOLO[m.radiant_team[i][0]][0] += 1
                    GLOBALHEROWINRATES_SOLO[m.radiant_team[i][0]][2] += 1

                GLOBALHEROWINRATES_SOLO[m.dire_team[i][0]][1] += 1
                GLOBALHEROWINRATES_SOLO[m.dire_team[i][0]][5] += 1
                if m.winner == 'FALSE':
                    GLOBALHEROWINRATES_SOLO[m.dire_team[i][0]][0] += 1
                    GLOBALHEROWINRATES_SOLO[m.dire_team[i][0]][4] += 1
        
        for a in GLOBALHEROWINRATES_SOLO:
            if a[1] != 0:
                a[0] = float(a[0]) / a[1]
            if a[3] != 0:
                a[2] = float(a[2]) / a[3]
            if a[5] != 0:
                a[4] = float(a[4]) / a[5]

    elif mode == 'duo' or 'all':
       
        for a in GLOBALHEROWINRATES_DUO:
            a = [0, 0, 0, 0, 0, 0]      # [winrate, matches_played_with_this_heroes, wr_radiant, games_radiant, wr_dire, games_dire]

        for m in all_matches:
            for i1 in range(5):
                for i2 in range(i1):

                    GLOBALHEROWINRATES_DUO[m.radiant_team[i1][0]][m.radiant_team[i2][0]][1] += 1
                    GLOBALHEROWINRATES_DUO[m.radiant_team[i1][0]][m.radiant_team[i2][0]][3] += 1
                    if m.winner == 'TRUE':
                        GLOBALHEROWINRATES_DUO[m.radiant_team[i1][0]][m.radiant_team[i2][0]][0] += 1
                        GLOBALHEROWINRATES_DUO[m.radiant_team[i1][0]][m.radiant_team[i2][0]][2] += 1

                    GLOBALHEROWINRATES_DUO[m.dire_team[i1][0]][m.dire_team[i2][0]][1] += 1
                    GLOBALHEROWINRATES_DUO[m.dire_team[i1][0]][m.dire_team[i2][0]][5] += 1
                    if m.winner == 'FALSE':
                        GLOBALHEROWINRATES_DUO[m.dire_team[i1][0]][m.dire_team[i2][0]][0] += 1
                        GLOBALHEROWINRATES_DUO[m.dire_team[i1][0]][m.dire_team[i2][0]][4] += 1
        
        for b in GLOBALHEROWINRATES_DUO:
            for a in b:    
                if a[1] != 0:
                    a[0] = float(a[0]) / a[1]
                if a[3] != 0:
                    a[2] = float(a[2]) / a[3]
                if a[5] != 0:
                    a[4] = float(a[4]) / a[5]

    elif mode == 'enemies' or 'all':
       
        for a in GLOBALHEROWINRATES_ENEMIES:
            a = [0, 0, 0, 0, 0, 0]      # [winrate, matches_played_with_this_heroes, wr_radiant, games_radiant, wr_dire, games_dire]

        for m in all_matches:
            for i1 in range(5):
                for i2 in range(5):
                    
                    GLOBALHEROWINRATES_ENEMIES[m.radiant_team[i1][0]][m.dire_team[i2][0]][1] += 1
                    GLOBALHEROWINRATES_ENEMIES[m.radiant_team[i1][0]][m.dire_team[i2][0]][3] += 1
                    GLOBALHEROWINRATES_ENEMIES[m.dire_team[i2][0]][m.radiant_team[i1][0]][1] += 1
                    GLOBALHEROWINRATES_ENEMIES[m.dire_team[i2][0]][m.radiant_team[i1][0]][5] += 1
                    if m.winner == 'TRUE':
                        GLOBALHEROWINRATES_ENEMIES[m.radiant_team[i1][0]][m.dire_team[i2][0]][0] += 1
                        GLOBALHEROWINRATES_ENEMIES[m.radiant_team[i1][0]][m.radiant_team[i2][0]][2] += 1
                        
                    if m.winner == 'FALSE':
                        GLOBALHEROWINRATES_ENEMIES[m.dire_team[i2][0]][m.radiant_team[i1][0]][0] += 1
                        GLOBALHEROWINRATES_ENEMIES[m.dire_team[i2][0]][m.radiant_team[i1][0]][4] += 1
                    
        for b in GLOBALHEROWINRATES_ENEMIES:
            for a in b:    
                if a[1] != 0:
                    a[0] = float(a[0]) / a[1]
                if a[3] != 0:
                    a[2] = float(a[2]) / a[3]
                if a[5] != 0:
                    a[4] = float(a[4]) / a[5]

    else:
        raise ValueError("wrong value for argument mode. suggested: 'solo', 'duo', 'enemy', 'all'. current: {0}".format(mode)) 


######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
def CheckHeroesDistribution(all_matches, mode = 'winrate'):
    """
    temporary function, helps to visualize information
    draws a plot wich show heroes' pick frequency or winrate
    """
    m = 0
    if mode == 'winrate':
        m = 0
    elif mode == 'total':
        m = 1
    else:
        raise ValueError('wrong value for mode argument. suggested: winrate, total')

    games_played = [CheckWinRate(all_matches, [(i + 1, -1, -1)], [])[m] for i in range(len(Heroes.heroes))]
    games_played.sort()
    games_played = [x for x in games_played if x > 0]   # deleting information about dummy (not existing) heroes

    #(n, bins, patches) = plt.hist([i[1] for i in games_played], 40, label = 'number of games')
    plt.plot(games_played, label = 'Y: winrate of a hero\nX: hero_id (sorted)')
    plt.legend(loc='upper left')
    plt.show()

######################################################################################################################################################
def CheckAnomalities(all_matches):
    """
    test function. helps to understand data
    """
    anom_array = []
    for i in range(1000):   # choosing 1000 random pairs. (if we check all groups it will take more then 10 years)
        a = random.randint(1, 120)
        b = random.randint(1, 120)
        c = random.randint(1, 120)
        d = random.randint(1, 120)
        ac = CheckWinRate(all_matches, [(a, -1, -1)], [(c, -1, -1)])
        ad = CheckWinRate(all_matches, [(a, -1, -1)], [(d, -1, -1)])
        bc = CheckWinRate(all_matches, [(b, -1, -1)], [(c, -1, -1)])
        bd = CheckWinRate(all_matches, [(b, -1, -1)], [(d, -1, -1)])
        abcd = CheckWinRate(all_matches, [(a, -1, -1), (b, -1, -1)], [(c, -1, -1), (d, -1, -1)])
        if (abcd[0] < 0) or (ac[0] < 0) or (ad[0] < 0) or (bc[0] < 0) or (bd[0] < 0):
            continue
        anom = float(ac[0] + ad[0] + bc[0] + bd[0]) / 4 - abcd[0]
        anom_array.append(anom)
        print abcd, ac, ad, bc, bd
    plt.hist(anom_array, 20)
    plt.show()
######################################################################################################################################################
def CheckMultiCollinearity(keys):
    """
    checks collinearity between heroes' variables
    keys - list of keys, a.k.a. mask

    returns nothing but results are printed
    """
    xs = np.zeros((len(keys), len(Heroes.heroes) - 1))
    for j in range(len(Heroes.heroes) - 1):
        for i in range(len(keys)):
            xs[i][j] = Heroes.heroes[j + 1][keys[i]]# + 1 + random.gauss(0, 0.01)
    corr = np.corrcoef(xs)  # correlation matrix
    w, v = np.linalg.eig(corr) # eigen values & eigen vectors

    print "eigen values of correlation matrix:"
    print w

######################################################################################################################################################
def PlayGround():
    """
    function created for tests
    computes avarage values of indexes when match is lost and when it's won
    """

    # reading data and creating all needed structures. takes approximately 5-6 seconds (on my laptop)
    dota2matches = LoadDota2Matches(filepath = 'Dataset/Matches.txt')
    print "Numbber of mathes: ", len(dota2matches)

    # here we do some tests. have fun

    index = []  # indef = f(features)
    result = [] # 0 or 1
    all_features = ['push', 'nuker', 'disabler', 'initiator', 'carry', 'escape', 'durable', 'jungler', 'outpush']    
    used_features = []
    attack_features = ['nuker', 'disabler', 'initiator', 'push']
    defense_features = ['escape', 'durable', 'disabler']

    #CheckMultiCollinearity(all_features)

    fout = open("log.txt", "w+")

    for m in dota2matches:
        radiant_features = GetHeroesFeatures(m.radiant_team)
        dire_features = GetHeroesFeatures(m.dire_team)
        ##########
        i = 0   # i, i1, i2 are indexes that discribe team's strength
        
        #i = SumFeatures(radiant_featuser, used_features) - SumFeatures(dire_features, used_features)
        
        i1 = SumHeroesFeatures(radiant_features, attack_features) / float(1 + SumHeroesFeatures(dire_features, defense_features)) #for team radiant
        i2 = SumHeroesFeatures(dire_features , attack_features) / float(1 + SumHeroesFeatures(radiant_features, defense_features))    # for team dire
        i = i1 / (i1 + i2)
        ##########
        index.append(i)
        if m.winner == 'TRUE':
            result.append(1)
        else:
            result.append(0)

        fout.write("{0}\n".format([i1, i2, i, m.winner]))
        
    print len(index)

    fout.close()

    win_avg_index = 0   # average index value when won 
    loose_avg_index = 0 # average index value when lost
    for i in range(len(index)):
        if result[i] == 1:
            win_avg_index += index[i]
        else:
            loose_avg_index += index[i]
    win_avg_index = win_avg_index / float(sum(result))
    loose_avg_index = loose_avg_index / float(len(result) - sum(result))
    #print "win_avg: ", win_avg
    #print "loose_avg: ", loose_avg
    print "\n delta", win_avg_index - loose_avg_index
    print used_features

    #plt.scatter(index, result)
    #plt.show()


######################################################################################################################################################
def main():
    # reading data and creating all needed structures. takes approximately 5-6 seconds (on my laptop)
    dota2matches = LoadDota2Matches(filepath = 'Dataset/Matches.txt')
    print "Numbber of mathes: ", len(dota2matches)

    dota2matches_train, dota2matches_test = train_test_split(dota2matches, test_size = 0.05)
    
    print "features genegetion is started"
    HandleGlobalVariables(dota2matches_train, mode = 'solo')
    HandleGlobalVariables(dota2matches_train, mode = 'duo')
    #HandleGlobalVariables(dota2matches_train, mode = 'enemies')
    data_train = np.array([GetFeatures(m, 5) for m in dota2matches_train])
    print "train created"
    data_test = np.array([GetFeatures(m, 5) for m in dota2matches_test])
    print "test created"
    
    data_train = data_train.T
    corr = np.corrcoef(data_train)  # correlation matrix
    w, v = np.linalg.eig(corr) # eigen values & eigen vectors
    print w
    data_train = data_train.T
    print "eigen values are shown above"
    
    data_train_2 = pd.get_dummies(pd.DataFrame(data_train))
    data_test_2 = pd.get_dummies(pd.DataFrame(data_test))
    
    X_train = data_train_2
    X_test = data_test_2
    y_train = []
    y_test = []
    for m in dota2matches_train:
        if m.winner == 'TRUE':
            y_train.append(1)
        else:
            y_train.append(0)
    for m in dota2matches_test:
        if m.winner == 'TRUE':
            y_test.append(1)
        else:
            y_test.append(0)

    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print confusion_matrix
    print 'Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test))
    print classifier.coef_



if __name__ == "__main__":
    main()
    #PlayGround()
    #CreateHeroesDataBase()

