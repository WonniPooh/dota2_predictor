# -*- coding: utf-8 -*-

import Heroes
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns

#class hero:
#   def _init_(self, name, hero_id, side):
#       self.name = name
#       self.hero_id = hero_id
#       self.side = side
class dota2match:
    def __init__(self, game_id, data, winner, cluster_id, game_mod, game_type, radiant_team, dire_team):
        self.game_id = game_id
        self.data = data
        self.winner = winner    # TRUE for radiant or FALSE for dire
        self.cluster_id = cluster_id    # server id (or idk)
        self.game_mod = game_mod
        self.game_type = game_type
        self.radiant_team = radiant_team    # [(hero_id, is_sup, lane), ...] - from Heroes.heroes
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
    loads information about mathes from a file
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
def SetEasyStrength(GLOBALHEROWINRATES, group_a, group_b):      # checked
    """
    group_a - list of tuples: [(hero_id, is_sup, lane), ...] , look hero_id's in Heroes.heroes
    group_b - same as group_a

    return value - tuple: (strength_1, strength_2), strength_i  - unnormalized chance to win (float or integer)
    or
    return value - tuple: (win_chance_1, win_chance_2), win_chance_i - chance to win. summ = 1
    mb second one is better. we will see
    """
    all_hero_parameters = ['push', 'nuker', 'disabler', 'initiator', 'carry', 'escape', 'durable', 'jungler', 'outpush']
    mask = ['outpush', 'nuker', 'initiator', 'disabler']
    
    strength_a = [GLOBALHEROWINRATES[h[0] - 1][0] for h in sorted(group_a, key = lambda h: h[1])]
    strength_b = [GLOBALHEROWINRATES[h[0] - 1][0] for h in sorted(group_b, key = lambda h: h[1])]
    #indexes_a = [[Heroes.heroes[h[0]][k]] for h in sorted(group_a, key = lambda h: h[1]) for k in mask]
    #indexes_b = [[Heroes.heroes[h[0]][k]] for h in sorted(group_b, key = lambda h: h[1]) for k in mask]
    
    summ_strength_a = 1
    for w in strength_a:
        summ_strength_a = summ_strength_a * w     # chance to fail
    #summ_strength_a = 1 - summ_strength_a       # chance not to fail
    summ_strength_b = 1
    for w in strength_b:
        summ_strength_b = summ_strength_b * w     # chance to fail
    #summ_strength_b = 1 - summ_strength_b       # chance not to fail

    result = (summ_strength_a, summ_strength_b)
    return result

def SetEasyStrength2(GLOBALHEROWINRATES, group_a, group_b): 
    """
    tests of a function above
    """
    mask = ['outpush']
    strength_a = [GLOBALHEROWINRATES[h[0] - 1][0] for h in sorted(group_a, key = lambda h: h[1])]
    strength_b = [GLOBALHEROWINRATES[h[0] - 1][0] for h in sorted(group_b, key = lambda h: h[1])]
    #indexes_a = [[Heroes.heroes[h[0]][k]] for h in sorted(group_a, key = lambda h: h[1]) for k in mask]
    #indexes_b = [[Heroes.heroes[h[0]][k]] for h in sorted(group_b, key = lambda h: h[1]) for k in mask]
    
    summ_strength_a = 1
    for w in strength_a:
        summ_strength_a = summ_strength_a * (1 - w)     # chance to fail
    summ_strength_a = 1 - summ_strength_a       # chance not to fail
    summ_strength_b = 1
    for w in strength_b:
        summ_strength_b = summ_strength_b * (1 - w)     # chance to fail
    summ_strength_b = 1 - summ_strength_b       # chance not to fail

    result = (summ_strength_a, summ_strength_b)

    return result

######################################################################################################################################################
def MergeGroups(all_groups, all_winchances):
    """
    all_groups - list. look at return value of GenerateSubGroups()
    all_winchances - list. look at return value of SetEasyStrength()

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
def GetFeatures(all_matches, match, GLOBALHEROWINRATES = [], mod = -1):
    """
    match - object with a type 'dota2match'

    return value - list of features (integer ordered values)
    """
    result = 0

    if mod == 0:    # metod 0. features = random numbers
        result = [random.randint(0, 5) for i in range(5)]
        return result
    
    elif mod == 1:    # metod 1. features = summ of masked parameters for the whole team
        all_hero_parameters = ['push', 'nuker', 'disabler', 'initiator', 'carry', 'escape', 'durable', 'jungler', 'outpush']
        mask = ['outpush', 'nuker', 'initiator', 'disabler']
        rt = GetHeroesFeatures(match.radiant_team, mask)
        dt = GetHeroesFeatures(match.dire_team, mask)
        result = [rt[k] for k in rt.keys()] + [dt[k] for k in dt.keys()]
        return result
    
    elif mod == 2:    # metod 2. features = winrates for all orderd team members
        rt = [GLOBALHEROWINRATES[h[0] - 1][2] for h in sorted(match.radiant_team, key = lambda h: h[2])]
        dt = [GLOBALHEROWINRATES[h[0] - 1][4] for h in sorted(match.dire_team, key = lambda h: h[2])]
        result = rt + dt
        return result

    elif mod == 3:   # metod 3. heroes' interection is taken into consideration
        subgroups = GenerateSubGroups(match.radiant_team, match.dire_team)
        features2 = [SetEasyStrength2(GLOBALHEROWINRATES, sg[0], sg[1]) for sg in subgroups]
        features1 = [SetEasyStrength(GLOBALHEROWINRATES, sg[0], sg[1]) for sg in subgroups]
        rt1 = [f[0] for f in features1]
        dt1 = [f[1] for f in features1]
        rt2 = [f[0] for f in features2]
        dt2 = [f[1] for f in features2]
        rt = [GLOBALHEROWINRATES[h[0] - 1][0] for h in sorted(match.radiant_team, key = lambda h: h[2])]
        dt = [GLOBALHEROWINRATES[h[0] - 1][0] for h in sorted(match.dire_team, key = lambda h: h[2])]

        mask = ['outpush', 'nuker', 'initiator', 'disabler']
        rt3 = [GetHeroesFeatures(match.radiant_team, mask)[k] for k in mask]
        dt3 = [GetHeroesFeatures(match.dire_team, mask)[k] for k in mask]

        result = rt + dt + rt1 + dt1 + rt3 + dt3
        return result

    elif mod == 4:    # metod 4. features = all masked parameters for every hero in match
        all_hero_parameters = ['push', 'nuker', 'disabler', 'initiator', 'carry', 'escape', 'durable', 'jungler', 'outpush']
        mask = ['outpush', 'nuker', 'initiator', 'disabler']
        rt = [GetHeroesFeatures([h], mask) for h in match.radiant_team]
        dt = [GetHeroesFeatures([h], mask) for h in match.dire_team]
        result = [rt[i][k] for i in range(5) for k in rt[0].keys()] + [dt[i][k] for i in range(5) for k in dt[0].keys()]
        return result

    else:
        raise ValueError("wrong value for argument mod. suggested: 0, 1, 2. current: {0}".format(mod))

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

######################################################################################################################################################
def CheckHeroesDistribution(all_matches, mod = 'winrate'):
    """
    temporary function, helps to visualize information
    draws a plot wich show heroes' pick frequency or winrate
    """
    m = 0
    if mod == 'winrate':
        m = 0
    elif mod == 'total':
        m = 1
    else:
        raise ValueError('wrong value for mod argument. suggested: winrate, total')

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
    dota2matches_train, dota2matches_test = train_test_split(dota2matches, test_size = 0.20)
    GLOBALHEROWINRATES = []
    
    print "matches splited"
    GLOBALHEROWINRATES = [CheckWinRate(dota2matches_train, [(h['id'], 'FALSE', 0)], []) for h in Heroes.heroes if h['id'] != 'ID']
    data_train = np.array([GetFeatures(dota2matches_train, m, GLOBALHEROWINRATES, 3) for m in dota2matches_train])
    print "train created"
    GLOBALHEROWINRATES = [CheckWinRate(dota2matches, [(h['id'], 'FALSE', 0)], []) for h in Heroes.heroes if h['id'] != 'ID']
    data_test = np.array([GetFeatures(dota2matches_test, m, GLOBALHEROWINRATES, 3) for m in dota2matches_test])
    print "test created"
    """
    data = data.T
    corr = np.corrcoef(data)  # correlation matrix
    w, v = np.linalg.eig(corr) # eigen values & eigen vectors
    print w
    data = data.T
    """

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
    print "We done something"
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print confusion_matrix
    print 'Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test))
    print classifier.coef_



if __name__ == "__main__":
    main()
    #PlayGround()
    #CreateHeroesDataBase()

