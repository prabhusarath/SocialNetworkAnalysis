"""
classify.py
"""
import matplotlib.pyplot as plt
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import pickle
import csv
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import numpy as np
from numpy import recfromtxt
import os
import re
import sys
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile

"""
#Code for Creating Test and Train Data from Tweets
url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
zipfile = ZipFile(BytesIO(url.read()))
afinn_file = zipfile.open('AFINN/AFINN-111.txt')
afinn = dict()
for line in afinn_file:
    parts = line.strip().split()
    if len(parts) == 2:
        afinn[parts[0].decode("utf-8")] = int(parts[1])

def afinn_sentiment2(terms, afinn, verbose=False):
    pos = 0
    neg = 0
    for t in terms:
        if t in afinn:
            if verbose:
                print('\t%s=%d' % (t, afinn[t]))
            if afinn[t] > 0:
                pos += afinn[t]
            else:
                neg += afinn[t]
    
    Final_score = pos+neg
    if Final_score>=0:
        return 1
    else:
        return 0
    pass

def tokenize(text):
    return re.sub('\W+', ' ', text.lower()).split()

# CSV File Writting Function
#spamWriter_Train = csv.writer(open('TrainData.csv', 'w', encoding='utf-8'))
#spamWriter_Test = csv.writer(open('TestData.csv', 'w', encoding='utf-8'))
spamWriter_Test1 = csv.writer(open('TestData1.csv', 'w', encoding='utf-8'))

#train_tweets = pickle.load( open( "tweets.pkl", "rb" ))
#test_tweets = pickle.load( open( "tweets2.pkl", "rb" ))
test_tweets1 = pickle.load( open( "tweets1.pkl", "rb" ))

for i in range(0,len(test_tweets1)):
    l=[]
    Retweet = re.sub('http\S+','', test_tweets1[i]['text'])

    doc = tokenize(Retweet)
    l.append(afinn_sentiment2(doc, afinn, verbose=True))
    Retweet1 = Retweet.split()

    if len(Retweet1)>0:
        if Retweet1[0] != 'RT':
            l.append(Retweet)
    
    spamWriter_Test1.writerow(l)  """




def read_Train_data():
    labeled_tweets = []
    labels1=[]
    Pos_tweets = 0
    Neg_tweets = 0
    with open("TrainData.csv", 'rt',encoding='utf-8') as csvfile1:
        filereader = csv.reader(csvfile1)
        for  row in filereader:
            labeled_tweets.append(row[1])
            labels1.append(int(row[0]))
            if int(row[0]) == 0:
                if Neg_tweets == 0:
                    twen = row[1]
                Neg_tweets += 1
            else:
                if Pos_tweets == 0:
                    twep = row[1]
                Pos_tweets += 1
        return np.array([d for d in labeled_tweets]), np.array([d for d in labels1]),Pos_tweets,Neg_tweets,twep,twen

def read_Test_data():
    labeled_tweets = []
    labels1=[]
    Pos_tweets1 = 0
    Neg_tweets1 = 0
    with open("TestData.csv", 'rt',encoding='utf-8') as csvfile1:
        filereader = csv.reader(csvfile1)
        for  row in filereader:
            labeled_tweets.append(row[1])
            labels1.append(int(row[0]))
            if int(row[0]) == 0:
                Neg_tweets1 += 1
            else:
                Pos_tweets1 += 1
        return np.array([d for d in labeled_tweets]), np.array([d for d in labels1]),Pos_tweets1,Neg_tweets1

def tokenize(doc, keep_internal_punct=False):
    if (keep_internal_punct):
        string_tokenize = re.sub(r"""'(?!(?!> '))|'(?!(?<! '))""",'',doc.lower())
        return (np.array(re.sub(r"""["?,$!@&#><^%]""",'', string_tokenize).split()))
    else:
        return (np.array(re.sub(r'\W',' ', doc.lower()).split()))  
    pass

def token_features(tokens, feats):
    token_features_string,counter =(),0
    
    for strings in tokens:
        token_features_string += (str("token=") + str(strings),)
    
    for feats_words in list(token_features_string):
        feats[feats_words] += (counter+1)
    
    pass

def token_pair_features(tokens, feats, k=3):
    windows_list,counter = (),0
    
    for win_range in range(len(tokens)-(k-1)):
        windows_list += (tokens[win_range:win_range+k],)  
    
    for tokens_in_windows in list(windows_list):
        for windows_size in range(k-1,k):
            for cartesian_prod in combinations(tokens_in_windows,windows_size):
                feats[str("token_pair=") + str(cartesian_prod[0])+"__"+str(cartesian_prod[1])] += (counter+1)
                
    pass

neg_words = set(['abilities', 'ability', 'aboard', 'absolve', 'absolved', 'absolves', 'absolving', 'absorbed', 'accept', 'accepted', 'accepting', 'accepts', 'accomplish', 'accomplished', 'accomplishes', 'achievable', 'acquit', 'acquits', 'acquitted', 'acquitting', 'active', 'adequate', 'admire', 'admired', 'admires', 'admiring', 'adopt', 'adopts', 'adorable', 'adore', 'adored', 'adores', 'advanced', 'advantage', 'advantages', 'adventure', 'adventures', 'adventurous', 'affection', 'affectionate', 'agog', 'agree', 'agreeable', 'agreed', 'agreement', 'agrees', 'alive', 'allow', 'amaze', 'amazed', 'amazes', 'amazing', 'ambitious', 'amuse', 'amused', 'amusement', 'amusements', 'anticipation', 'appease', 'appeased', 'appeases', 'appeasing', 'applaud', 'applauded', 'applauding', 'applauds', 'applause', 'appreciate', 'appreciated', 'appreciates', 'appreciating', 'appreciation', 'approval', 'approved', 'approves', 'ardent', 'asset', 'assets', 'astonished', 'astound', 'astounded', 'astounding', 'astoundingly', 'astounds', 'attract', 'attracted', 'attracting', 'attraction', 'attractions', 'attracts', 'audacious', 'authority', 'avid', 'award', 'awarded', 'awards', 'awesome', 'backed', 'backing', 'backs', 'bargain', 'beatific', 'beauties', 'beautiful', 'beautifully', 'beautify', 'beloved', 'benefit', 'benefits', 'benefitted', 'benefitting', 'best', 'better', 'big', 'bless', 'blesses', 'blessing', 'bliss', 'blissful', 'blithe', 'blockbuster', 'bold', 'boldly', 'boost', 'boosted', 'boosting', 'boosts', 'brave', 'breakthrough', 'breathtaking', 'bright', 'brightest', 'brightness', 'brilliant', 'brisk', 'buoyant', 'calm', 'calmed', 'calming', 'calms', 'capable', 'captivated', 'care', 'carefree', 'careful', 'carefully', 'cares', 'celebrate', 'celebrated', 'celebrates', 'celebrating', 'certain', 'chance', 'chances', 'charm', 'charming', 'cheer', 'cheered', 'cheerful', 'cheering', 'cheers', 'cheery', 'cherish', 'cherished', 'cherishes', 'cherishing', 'chic', 'clarifies', 'clarity', 'classy', 'clean', 'cleaner', 'clear', 'cleared', 'clearly', 'clears', 'clever', 'comedy', 'comfort', 'comfortable', 'comforting', 'comforts', 'commend', 'commended', 'commit', 'commitment', 'commits', 'committed', 'committing', 'compassionate', 'compelled', 'competent', 'competitive', 'comprehensive', 'conciliate', 'conciliated', 'conciliates', 'conciliating', 'confidence', 'confident', 'congrats', 'congratulate', 'congratulation', 'congratulations', 'consent', 'consents', 'consolable', 'convince', 'convinced', 'convinces', 'convivial', 'cool', 'courage', 'courageous', 'courteous', 'courtesy', 'coziness', 'creative', 'curious', 'cute', 'daredevil', 'daring', 'dauntless', 'dear', 'dearly', 'debonair', 'decisive', 'dedicated', 'defender', 'defenders', 'delight', 'delighted', 'delighting', 'delights', 'desirable', 'desire', 'desired', 'desirous', 'determined', 'devoted', 'diamond', 'dream', 'dreams', 'eager', 'earnest', 'ease', 'easy', 'ecstatic', 'effective', 'effectively', 'elated', 'elation', 'elegant', 'elegantly', 'embrace', 'empathetic', 'enchanted', 'encourage', 'encouraged', 'encouragement', 'encourages', 'endorse', 'endorsed', 'endorsement', 'endorses', 'energetic', 'engage', 'engages', 'engrossed', 'enjoy', 'enjoying', 'enjoys', 'enlighten', 'enlightened', 'enlightening', 'enlightens', 'enrapture', 'ensure', 'ensuring', 'enterprising', 'entertaining', 'enthral', 'enthusiastic', 'entitled', 'entrusted', 'esteemed', 'ethical', 'euphoria', 'euphoric', 'exasperated', 'excellence', 'excellent', 'excite', 'excited', 'excitement', 'exciting', 'exclusive', 'exhilarated', 'exhilarates', 'exhilarating', 'exonerate', 'exonerated', 'exonerates', 'exonerating', 'expand', 'expands', 'exploration', 'explorations', 'extend', 'extends', 'exuberant', 'exultant', 'exultantly', 'fabulous', 'fair', 'faith', 'faithful', 'fame', 'fan', 'fantastic', 'fascinate', 'fascinated', 'fascinates', 'fascinating', 'favor', 'favored', 'favorite', 'favorited', 'favorites', 'favors', 'fearless', 'feeling', 'fervent', 'fervid', 'festive', 'fine', 'fit', 'fitness', 'flagship', 'focused', 'fond', 'fondness', 'forgive', 'forgiving', 'fortunate', 'free', 'freedom', 'fresh', 'friendly', 'frisky', 'ftw', 'fulfill', 'fulfilled', 'fulfills', 'fun', 'funky', 'funnier', 'funny', 'futile', 'gain', 'gained', 'gaining', 'gains', 'gallant', 'gallantly', 'gallantry', 'generous', 'genial', 'gift', 'glad', 'glamorous', 'glamourous', 'glee', 'gleeful', 'glorious', 'glory', 'god', 'godsend', 'good', 'goodness', 'grace', 'gracious', 'grand', 'grant', 'granted', 'granting', 'grants', 'grateful', 'gratification', 'great', 'greater', 'greatest', 'greet', 'greeted', 'greeting', 'greetings', 'greets', 'growing', 'growth', 'guarantee', 'ha', 'haha', 'hahaha', 'hahahah', 'hail', 'hailed', 'happiness', 'happy', 'hardier', 'hardy', 'haunting', 'healthy', 'heartfelt', 'heaven', 'heavenly', 'help', 'helpful', 'helping', 'helps', 'hero', 'heroes', 'heroic', 'highlight', 'hilarious', 'honest', 'honor', 'honored', 'honoring', 'honour', 'honoured', 'honouring', 'hope', 'hopeful', 'hopefully', 'hopes', 'hoping', 'hug', 'huge', 'hugs', 'humerous', 'humor', 'humorous', 'humour', 'humourous', 'hurrah', 'immortal', 'immune', 'importance', 'important', 'impress', 'impressed', 'impresses', 'impressive', 'improve', 'improved', 'improvement', 'improves', 'improving', 'increase', 'increased', 'indestructible', 'infatuated', 'infatuation', 'influential', 'innovate', 'innovates', 'innovation', 'innovative', 'inquisitive', 'inspiration', 'inspirational', 'inspire', 'inspired', 'inspires', 'inspiring', 'intact', 'integrity', 'intelligent', 'intense', 'interest', 'interested', 'interesting', 'interests', 'intricate', 'intrigues', 'invincible', 'invite', 'inviting', 'invulnerable', 'irresistible', 'irresponsible', 'jaunty', 'jesus', 'jewel', 'jewels', 'jocular', 'join', 'joke', 'jokes', 'jolly', 'jovial', 'joy', 'joyful', 'joyfully', 'joyous', 'jubilant', 'justice', 'justifiably', 'justified', 'keen', 'kind', 'kinder', 'kiss', 'kudos', 'landmark', 'laugh', 'laughed', 'laughing', 'laughs', 'laughting', 'launched', 'lawl', 'legal', 'legally', 'lenient', 'lifesaver', 'lighthearted', 'like', 'liked', 'likes', 'lively', 'lmao', 'lmfao', 'lol', 'lovable', 'love', 'loved', 'lovelies', 'lovely', 'loving', 'loyal', 'loyalty', 'luck', 'luckily', 'lucky', 'marvel', 'marvelous', 'marvels', 'masterpiece', 'masterpieces', 'matter', 'matters', 'mature', 'meaningful', 'medal', 'meditative', 'mercy', 'merry', 'methodical', 'miracle', 'mirth', 'mirthful', 'mirthfully', 'motivate', 'motivated', 'motivating', 'motivation', 'natural', 'nice', 'nifty', 'noble', 'novel', 'obsessed', 'oks', 'ominous', 'once-in-a-lifetime', 'opportunities', 'opportunity', 'optimism', 'optimistic', 'outreach', 'outstanding', 'overjoyed', 'paradise', 'pardon', 'pardoned', 'pardoning', 'pardons', 'passionate', 'peace', 'peaceful', 'peacefully', 'perfect', 'perfected', 'perfectly', 'perfects', 'picturesque', 'playful', 'pleasant', 'please', 'pleased', 'pleasure', 'popular', 'positive', 'positively', 'powerful', 'praise', 'praised', 'praises', 'praising', 'pray', 'praying', 'prays', 'prepared', 'pretty', 'privileged', 'proactive', 'progress', 'prominent', 'promise', 'promised', 'promises', 'promote', 'promoted', 'promotes', 'promoting', 'prospect', 'prospects', 'prosperous', 'protect', 'protected', 'protects', 'proud', 'proudly', 'rapture', 'raptured', 'raptures', 'rapturous', 'ratified', 'reach', 'reached', 'reaches', 'reaching', 'reassure', 'reassured', 'reassures', 'reassuring', 'recommend', 'recommended', 'recommends', 'redeemed', 'rejoice', 'rejoiced', 'rejoices', 'rejoicing', 'relaxed', 'reliant', 'relieve', 'relieved', 'relieves', 'relieving', 'relishing', 'remarkable', 'rescue', 'rescued', 'rescues', 'resolute', 'resolve', 'resolved', 'resolves', 'resolving', 'respected', 'responsible', 'responsive', 'restful', 'restore', 'restored', 'restores', 'restoring', 'revered', 'revive', 'revives', 'reward', 'rewarded', 'rewarding', 'rewards', 'rich', 'rigorous', 'rigorously', 'robust', 'rofl', 'roflcopter', 'roflmao', 'romance', 'rotfl', 'rotflmfao', 'rotflol', 'safe', 'safely', 'safety', 'salient', 'satisfied', 'save', 'saved', 'scoop', 'secure', 'secured', 'secures', 'self-confident', 'serene', 'sexy', 'share', 'shared', 'shares', 'significance', 'significant', 'sincere', 'sincerely', 'sincerest', 'sincerity', 'slick', 'slicker', 'slickest', 'smart', 'smarter', 'smartest', 'smile', 'smiled', 'smiles', 'smiling', 'sobering', 'solid', 'solidarity', 'solution', 'solutions', 'solve', 'solved', 'solves', 'solving', 'soothe', 'soothed', 'soothing', 'sophisticated', 'spark', 'sparkle', 'sparkles', 'sparkling', 'spirit', 'spirited', 'splendid', 'sprightly', 'stable', 'stamina', 'steadfast', 'stimulate', 'stimulated', 'stimulates', 'stimulating', 'stout', 'straight', 'strength', 'strengthen', 'strengthened', 'strengthening', 'strengthens', 'strong', 'stronger', 'strongest', 'stunning', 'suave', 'substantial', 'substantially', 'success', 'successful', 'sunshine', 'super', 'superb', 'superior', 'support', 'supported', 'supporter', 'supporters', 'supporting', 'supportive', 'supports', 'survived', 'surviving', 'survivor', 'sweet', 'swift', 'swiftly', 'sympathetic', 'sympathy', 'tender', 'terrific', 'thank', 'thankful', 'thanks', 'thoughtful', 'thrilled', 'tolerant', 'top', 'tops', 'tranquil', 'treasure', 'treasures', 'triumph', 'triumphant', 'true', 'trust', 'trusted', 'unbiased', 'unequaled', 'unified', 'united', 'unmatched', 'unstoppable', 'untarnished', 'useful', 'usefulness', 'validate', 'validated', 'validates', 'validating', 'vested', 'vibrant', 'vigilant', 'vindicate', 'vindicated', 'vindicates', 'vindicating', 'virtuous', 'vision', 'visionary', 'visioning', 'visions', 'vitality', 'vitamin', 'vivacious', 'want', 'warm', 'warmth', 'wealth', 'wealthy', 'welcome', 'welcomed', 'welcomes', 'whimsical', 'willingness', 'win', 'winner', 'winning', 'wins', 'winwin', 'wish', 'wishes', 'wishing', 'won', 'wonderful', 'woo', 'woohoo', 'wooo', 'woow', 'worshiped', 'worth', 'worthy', 'wow', 'wowow', 'wowww', 'yeah', 'yearning', 'yeees', 'yes', 'youthful', 'yummy', 'zealous'])
pos_words = set(['abandon', 'abandoned', 'abandons', 'abducted', 'abduction', 'abductions', 'abhor', 'abhorred', 'abhorrent', 'abhors', 'absentee', 'absentees', 'abuse', 'abused', 'abuses', 'abusive', 'accident', 'accidental', 'accidentally', 'accidents', 'accusation', 'accusations', 'accuse', 'accused', 'accuses', 'accusing', 'ache', 'aching', 'acrimonious', 'admit', 'admits', 'admitted', 'admonish', 'admonished', 'affected', 'afflicted', 'affronted', 'afraid', 'aggravate', 'aggravated', 'aggravates', 'aggravating', 'aggression', 'aggressions', 'aggressive', 'aghast', 'agonise', 'agonised', 'agonises', 'agonising', 'agonize', 'agonized', 'agonizes', 'agonizing', 'alarm', 'alarmed', 'alarmist', 'alarmists', 'alas', 'alert', 'alienation', 'allergic', 'alone', 'ambivalent', 'anger', 'angers', 'angry', 'anguish', 'anguished', 'animosity', 'annoy', 'annoyance', 'annoyed', 'annoying', 'annoys', 'antagonistic', 'anti', 'anxiety', 'anxious', 'apathetic', 'apathy', 'apeshit', 'apocalyptic', 'apologise', 'apologised', 'apologises', 'apologising', 'apologize', 'apologized', 'apologizes', 'apologizing', 'apology', 'appalled', 'appalling', 'apprehensive', 'arrest', 'arrested', 'arrests', 'arrogant', 'ashame', 'ashamed', 'ass', 'assassination', 'assassinations', 'assfucking', 'asshole', 'attack', 'attacked', 'attacking', 'attacks', 'avert', 'averted', 'averts', 'avoid', 'avoided', 'avoids', 'await', 'awaited', 'awaits', 'awful', 'awkward', 'axe', 'axed', 'bad', 'badass', 'badly', 'bailout', 'bamboozle', 'bamboozled', 'bamboozles', 'ban', 'banish', 'bankrupt', 'bankster', 'banned', 'barrier', 'bastard', 'bastards', 'battle', 'battles', 'beaten', 'beating', 'belittle', 'belittled', 'bereave', 'bereaved', 'bereaves', 'bereaving', 'betray', 'betrayal', 'betrayed', 'betraying', 'betrays', 'bias', 'biased', 'bitch', 'bitches', 'bitter', 'bitterly', 'bizarre', 'blah', 'blame', 'blamed', 'blames', 'blaming', 'blind', 'block', 'blocked', 'blocking', 'blocks', 'bloody', 'blurry', 'boastful', 'bomb', 'bore', 'bored', 'boring', 'bother', 'bothered', 'bothers', 'bothersome', 'boycott', 'boycotted', 'boycotting', 'boycotts', 'brainwashing', 'bribe', 'broke', 'broken', 'brooding', 'bullied', 'bullshit', 'bully', 'bullying', 'bummer', 'burden', 'burdened', 'burdening', 'burdens', 'cancel', 'cancelled', 'cancelling', 'cancels', 'cancer', 'careless', 'casualty', 'catastrophe', 'catastrophic', 'cautious', 'censor', 'censored', 'censors', 'chagrin', 'chagrined', 'challenge', 'chaos', 'chaotic', 'charged', 'charges', 'charmless', 'chastise', 'chastised', 'chastises', 'chastising', 'cheat', 'cheated', 'cheater', 'cheaters', 'cheats', 'cheerless', 'childish', 'chilling', 'choke', 'choked', 'chokes', 'choking', 'clash', 'clouded', 'clueless', 'cock', 'cocksucker', 'cocksuckers', 'cocky', 'coerced', 'collapse', 'collapsed', 'collapses', 'collapsing', 'collide', 'collides', 'colliding', 'collision', 'collisions', 'colluding', 'combat', 'combats', 'complacent', 'complain', 'complained', 'complains', 'condemn', 'condemnation', 'condemned', 'condemns', 'conflict', 'conflicting', 'conflictive', 'conflicts', 'confuse', 'confused', 'confusing', 'conspiracy', 'constrained', 'contagion', 'contagions', 'contagious', 'contempt', 'contemptuous', 'contemptuously', 'contend', 'contender', 'contending', 'contentious', 'contestable', 'controversial', 'controversially', 'cornered', 'corpse', 'costly', 'cover-up', 'coward', 'cowardly', 'cramp', 'crap', 'crash', 'crazier', 'craziest', 'crazy', 'crestfallen', 'cried', 'cries', 'crime', 'criminal', 'criminals', 'crisis', 'critic', 'criticism', 'criticize', 'criticized', 'criticizes', 'criticizing', 'critics', 'cruel', 'cruelty', 'crush', 'crushed', 'crushes', 'crushing', 'cry', 'crying', 'cunt', 'curse', 'cut', 'cuts', 'cutting', 'cynic', 'cynical', 'cynicism', 'damage', 'damages', 'damn', 'damned', 'damnit', 'danger', 'darkest', 'darkness', 'dead', 'deadlock', 'deafening', 'death', 'debt', 'deceit', 'deceitful', 'deceive', 'deceived', 'deceives', 'deceiving', 'deception', 'defeated', 'defect', 'defects', 'defenseless', 'defer', 'deferring', 'defiant', 'deficit', 'degrade', 'degraded', 'degrades', 'dehumanize', 'dehumanized', 'dehumanizes', 'dehumanizing', 'deject', 'dejected', 'dejecting', 'dejects', 'delay', 'delayed', 'demand', 'demanded', 'demanding', 'demands', 'demonstration', 'demoralized', 'denied', 'denier', 'deniers', 'denies', 'denounce', 'denounces', 'deny', 'denying', 'depressed', 'depressing', 'derail', 'derailed', 'derails', 'deride', 'derided', 'derides', 'deriding', 'derision', 'despair', 'despairing', 'despairs', 'desperate', 'desperately', 'despondent', 'destroy', 'destroyed', 'destroying', 'destroys', 'destruction', 'destructive', 'detached', 'detain', 'detained', 'detention', 'devastate', 'devastated', 'devastating', 'dick', 'dickhead', 'die', 'died', 'difficult', 'diffident', 'dilemma', 'dipshit', 'dire', 'direful', 'dirt', 'dirtier', 'dirtiest', 'dirty', 'disabling', 'disadvantage', 'disadvantaged', 'disappear', 'disappeared', 'disappears', 'disappoint', 'disappointed', 'disappointing', 'disappointment', 'disappointments', 'disappoints', 'disaster', 'disasters', 'disastrous', 'disbelieve', 'discard', 'discarded', 'discarding', 'discards', 'disconsolate', 'disconsolation', 'discontented', 'discord', 'discounted', 'discouraged', 'discredited', 'disdain', 'disgrace', 'disgraced', 'disguise', 'disguised', 'disguises', 'disguising', 'disgust', 'disgusted', 'disgusting', 'disheartened', 'dishonest', 'disillusioned', 'disinclined', 'disjointed', 'dislike', 'dismal', 'dismayed', 'disorder', 'disorganized', 'disoriented', 'disparage', 'disparaged', 'disparages', 'disparaging', 'displeased', 'dispute', 'disputed', 'disputes', 'disputing', 'disqualified', 'disquiet', 'disregard', 'disregarded', 'disregarding', 'disregards', 'disrespect', 'disrespected', 'disruption', 'disruptions', 'disruptive', 'dissatisfied', 'distort', 'distorted', 'distorting', 'distorts', 'distract', 'distracted', 'distraction', 'distracts', 'distress', 'distressed', 'distresses', 'distressing', 'distrust', 'distrustful', 'disturb', 'disturbed', 'disturbing', 'disturbs', 'dithering', 'dizzy', 'dodging', 'dodgy', 'dolorous', 'doom', 'doomed', 'doubt', 'doubted', 'doubtful', 'doubting', 'doubts', 'douche', 'douchebag', 'downcast', 'downhearted', 'downside', 'drag', 'dragged', 'drags', 'drained', 'dread', 'dreaded', 'dreadful', 'dreading', 'dreary', 'droopy', 'drop', 'drown', 'drowned', 'drowns', 'drunk', 'dubious', 'dud', 'dull', 'dumb', 'dumbass', 'dump', 'dumped', 'dumps', 'dupe', 'duped', 'dysfunction', 'eerie', 'eery', 'embarrass', 'embarrassed', 'embarrasses', 'embarrassing', 'embarrassment', 'embittered', 'emergency', 'emptiness', 'empty', 'enemies', 'enemy', 'ennui', 'enrage', 'enraged', 'enrages', 'enraging', 'enslave', 'enslaved', 'enslaves', 'envies', 'envious', 'envy', 'envying', 'erroneous', 'error', 'errors', 'escape', 'escapes', 'escaping', 'eviction', 'evil', 'exaggerate', 'exaggerated', 'exaggerates', 'exaggerating', 'exclude', 'excluded', 'exclusion', 'excuse', 'exempt', 'exhausted', 'expel', 'expelled', 'expelling', 'expels', 'exploit', 'exploited', 'exploiting', 'exploits', 'expose', 'exposed', 'exposes', 'exposing', 'fad', 'fag', 'faggot', 'faggots', 'fail', 'failed', 'failing', 'fails', 'failure', 'failures', 'fainthearted', 'fake', 'fakes', 'faking', 'fallen', 'falling', 'falsified', 'falsify', 'farce', 'fascist', 'fascists', 'fatalities', 'fatality', 'fatigue', 'fatigued', 'fatigues', 'fatiguing', 'fear', 'fearful', 'fearing', 'fearsome', 'feeble', 'felonies', 'felony', 'fiasco', 'fidgety', 'fight', 'fire', 'fired', 'firing', 'flees', 'flop', 'flops', 'flu', 'flustered', 'fool', 'foolish', 'fools', 'forced', 'foreclosure', 'foreclosures', 'forget', 'forgetful', 'forgotten', 'frantic', 'fraud', 'frauds', 'fraudster', 'fraudsters', 'fraudulence', 'fraudulent', 'frenzy', 'fright', 'frightened', 'frightening', 'frikin', 'frowning', 'frustrate', 'frustrated', 'frustrates', 'frustrating', 'frustration', 'fuck', 'fucked', 'fucker', 'fuckers', 'fuckface', 'fuckhead', 'fucking', 'fucktard', 'fud', 'fuked', 'fuking', 'fuming', 'funeral', 'funerals', 'furious', 'gag', 'gagged', 'ghost', 'giddy', 'gloom', 'gloomy', 'glum', 'goddamn', 'grave', 'gray', 'greed', 'greedy', 'greenwash', 'greenwasher', 'greenwashers', 'greenwashing', 'grey', 'grief', 'grieved', 'gross', 'guilt', 'guilty', 'gullibility', 'gullible', 'gun', 'hacked', 'hapless', 'haplessness', 'hard', 'hardship', 'harm', 'harmed', 'harmful', 'harming', 'harms', 'harried', 'harsh', 'harsher', 'harshest', 'hate', 'hated', 'haters', 'hates', 'hating', 'haunt', 'haunted', 'haunts', 'havoc', 'heartbreaking', 'heartbroken', 'heavyhearted', 'hell', 'helpless', 'hesitant', 'hesitate', 'hid', 'hide', 'hides', 'hiding', 'hindrance', 'hoax', 'homesick', 'hooligan', 'hooliganism', 'hooligans', 'hopeless', 'hopelessness', 'horrendous', 'horrible', 'horrific', 'horrified', 'hostile', 'huckster', 'humiliated', 'humiliation', 'hunger', 'hurt', 'hurting', 'hurts', 'hypocritical', 'hysteria', 'hysterical', 'hysterics', 'idiot', 'idiotic', 'ignorance', 'ignorant', 'ignore', 'ignored', 'ignores', 'ill', 'illegal', 'illiteracy', 'illness', 'illnesses', 'imbecile', 'immobilized', 'impatient', 'imperfect', 'impose', 'imposed', 'imposes', 'imposing', 'impotent', 'imprisoned', 'inability', 'inaction', 'inadequate', 'incapable', 'incapacitated', 'incensed', 'incompetence', 'incompetent', 'inconsiderate', 'inconvenience', 'inconvenient', 'indecisive', 'indifference', 'indifferent', 'indignant', 'indignation', 'indoctrinate', 'indoctrinated', 'indoctrinates', 'indoctrinating', 'ineffective', 'ineffectively', 'infected', 'inferior', 'inflamed', 'infringement', 'infuriate', 'infuriated', 'infuriates', 'infuriating', 'inhibit', 'injured', 'injury', 'injustice', 'inquisition', 'insane', 'insanity', 'insecure', 'insensitive', 'insensitivity', 'insignificant', 'insipid', 'insult', 'insulted', 'insulting', 'insults', 'interrogated', 'interrupt', 'interrupted', 'interrupting', 'interruption', 'interrupts', 'intimidate', 'intimidated', 'intimidates', 'intimidating', 'intimidation', 'irate', 'ironic', 'irony', 'irrational', 'irresolute', 'irreversible', 'irritate', 'irritated', 'irritating', 'isolated', 'itchy', 'jackass', 'jackasses', 'jailed', 'jealous', 'jeopardy', 'jerk', 'joyless', 'jumpy', 'kill', 'killed', 'killing', 'kills', 'lack', 'lackadaisical', 'lag', 'lagged', 'lagging', 'lags', 'lame', 'lawsuit', 'lawsuits', 'lazy', 'leak', 'leaked', 'leave', 'lethargic', 'lethargy', 'liar', 'liars', 'libelous', 'lied', 'limitation', 'limited', 'limits', 'litigation', 'litigious', 'livid', 'loathe', 'loathed', 'loathes', 'loathing', 'lobby', 'lobbying', 'lonely', 'lonesome', 'longing', 'loom', 'loomed', 'looming', 'looms', 'loose', 'looses', 'loser', 'losing', 'loss', 'lost', 'lowest', 'lugubrious', 'lunatic', 'lunatics', 'lurk', 'lurking', 'lurks', 'mad', 'maddening', 'made-up', 'madly', 'madness', 'mandatory', 'manipulated', 'manipulating', 'manipulation', 'meaningless', 'mediocrity', 'melancholy', 'menace', 'menaced', 'mess', 'messed', 'mindless', 'misbehave', 'misbehaved', 'misbehaves', 'misbehaving', 'mischief', 'mischiefs', 'miserable', 'misery', 'misgiving', 'misinformation', 'misinformed', 'misinterpreted', 'misleading', 'misread', 'misreporting', 'misrepresentation', 'miss', 'missed', 'missing', 'mistake', 'mistaken', 'mistakes', 'mistaking', 'misunderstand', 'misunderstanding', 'misunderstands', 'misunderstood', 'moan', 'moaned', 'moaning', 'moans', 'mock', 'mocked', 'mocking', 'mocks', 'mongering', 'monopolize', 'monopolized', 'monopolizes', 'monopolizing', 'moody', 'mope', 'moping', 'moron', 'motherfucker', 'motherfucking', 'mourn', 'mourned', 'mournful', 'mourning', 'mourns', 'mumpish', 'murder', 'murderer', 'murdering', 'murderous', 'murders', 'myth', 'n00b', 'naive', 'nasty', 'naïve', 'needy', 'negative', 'negativity', 'neglect', 'neglected', 'neglecting', 'neglects', 'nerves', 'nervous', 'nervously', 'niggas', 'nigger', 'no', 'noisy', 'nonsense', 'noob', 'nosey', 'notorious', 'numb', 'nuts', 'obliterate', 'obliterated', 'obnoxious', 'obscene', 'obsolete', 'obstacle', 'obstacles', 'obstinate', 'odd', 'offend', 'offended', 'offender', 'offending', 'offends', 'offline', 'oppressed', 'oppressive', 'optionless', 'outcry', 'outmaneuvered', 'outrage', 'outraged', 'overload', 'overlooked', 'overreact', 'overreacted', 'overreaction', 'overreacts', 'oversell', 'overselling', 'oversells', 'oversimplification', 'oversimplified', 'oversimplifies', 'oversimplify', 'overstatement', 'overstatements', 'overweight', 'oxymoron', 'pain', 'pained', 'panic', 'panicked', 'panics', 'paradox', 'parley', 'passive', 'passively', 'pathetic', 'pay', 'penalty', 'pensive', 'peril', 'perjury', 'perpetrator', 'perpetrators', 'perplexed', 'persecute', 'persecuted', 'persecutes', 'persecuting', 'perturbed', 'pesky', 'pessimism', 'pessimistic', 'petrified', 'phobic', 'pileup', 'pique', 'piqued', 'piss', 'pissed', 'pissing', 'piteous', 'pitied', 'pity', 'poised', 'poison', 'poisoned', 'poisons', 'pollute', 'polluted', 'polluter', 'polluters', 'pollutes', 'poor', 'poorer', 'poorest', 'possessive', 'postpone', 'postponed', 'postpones', 'postponing', 'poverty', 'powerless', 'prblm', 'prblms', 'pressure', 'pressured', 'pretend', 'pretending', 'pretends', 'prevent', 'prevented', 'preventing', 'prevents', 'prick', 'prison', 'prisoner', 'prisoners', 'problem', 'problems', 'profiteer', 'propaganda', 'prosecute', 'prosecuted', 'prosecutes', 'prosecution', 'protest', 'protesters', 'protesting', 'protests', 'provoke', 'provoked', 'provokes', 'provoking', 'pseudoscience', 'punish', 'punished', 'punishes', 'punitive', 'pushy', 'puzzled', 'quaking', 'questionable', 'questioned', 'questioning', 'racism', 'racist', 'racists', 'rage', 'rageful', 'rainy', 'rant', 'ranter', 'ranters', 'rants', 'rape', 'rapist', 'rash', 'rebellion', 'recession', 'reckless', 'refuse', 'refused', 'refusing', 'regret', 'regretful', 'regrets', 'regretted', 'regretting', 'reject', 'rejected', 'rejecting', 'rejects', 'relentless', 'remorse', 'repulse', 'repulsed', 'resentful', 'resign', 'resigned', 'resigning', 'resigns', 'restless', 'restrict', 'restricted', 'restricting', 'restriction', 'restricts', 'retained', 'retard', 'retarded', 'retreat', 'revenge', 'revengeful', 'ridiculous', 'rig', 'rigged', 'riot', 'riots', 'risk', 'risks', 'rob', 'robber', 'robed', 'robing', 'robs', 'ruin', 'ruined', 'ruining', 'ruins', 'sabotage', 'sad', 'sadden', 'saddened', 'sadly', 'sappy', 'sarcastic', 'scam', 'scams', 'scandal', 'scandalous', 'scandals', 'scapegoat', 'scapegoats', 'scare', 'scared', 'scary', 'sceptical', 'scold', 'scorn', 'scornful', 'scream', 'screamed', 'screaming', 'screams', 'screwed', 'scumbag', 'sedition', 'seditious', 'seduced', 'self-deluded', 'selfish', 'selfishness', 'sentence', 'sentenced', 'sentences', 'sentencing', 'severe', 'shaky', 'shame', 'shamed', 'shameful', 'shattered', 'shit', 'shithead', 'shitty', 'shock', 'shocked', 'shocking', 'shocks', 'shoot', 'short-sighted', 'short-sightedness', 'shortage', 'shortages', 'shrew', 'shy', 'sick', 'sigh', 'silencing', 'silly', 'sinful', 'singleminded', 'skeptic', 'skeptical', 'skepticism', 'skeptics', 'slam', 'slash', 'slashed', 'slashes', 'slashing', 'slavery', 'sleeplessness', 'sluggish', 'slut', 'smear', 'smog', 'sneaky', 'snub', 'snubbed', 'snubbing', 'snubs', 'solemn', 'somber', 'son-of-a-bitch', 'sore', 'sorrow', 'sorrowful', 'sorry', 'spam', 'spammer', 'spammers', 'spamming', 'speculative', 'spiritless', 'spiteful', 'squelched', 'stab', 'stabbed', 'stabs', 'stall', 'stalled', 'stalling', 'stampede', 'startled', 'starve', 'starved', 'starves', 'starving', 'steal', 'steals', 'stereotype', 'stereotyped', 'stifled', 'stingy', 'stolen', 'stop', 'stopped', 'stopping', 'stops', 'strange', 'strangely', 'strangled', 'stressed', 'stressor', 'stressors', 'stricken', 'strike', 'strikers', 'strikes', 'struck', 'struggle', 'struggled', 'struggles', 'struggling', 'stubborn', 'stuck', 'stunned', 'stupid', 'stupidly', 'subversive', 'suck', 'sucks', 'suffer', 'suffering', 'suffers', 'suicidal', 'suicide', 'suing', 'sulking', 'sulky', 'sullen', 'suspect', 'suspected', 'suspecting', 'suspects', 'suspend', 'suspended', 'suspicious', 'swear', 'swearing', 'swears', 'swindle', 'swindles', 'swindling', 'tard', 'tears', 'tense', 'tension', 'terrible', 'terribly', 'terrified', 'terror', 'terrorize', 'terrorized', 'terrorizes', 'thorny', 'thoughtless', 'threat', 'threaten', 'threatened', 'threatening', 'threatens', 'threats', 'thwart', 'thwarted', 'thwarting', 'thwarts', 'timid', 'timorous', 'tired', 'tits', 'toothless', 'torn', 'torture', 'tortured', 'tortures', 'torturing', 'totalitarian', 'totalitarianism', 'tout', 'touted', 'touting', 'touts', 'tragedy', 'tragic', 'trap', 'trapped', 'trauma', 'traumatic', 'travesty', 'treason', 'treasonous', 'trembling', 'tremulous', 'tricked', 'trickery', 'trouble', 'troubled', 'troubles', 'tumor', 'twat', 'ugly', 'unacceptable', 'unappreciated', 'unapproved', 'unaware', 'unbelievable', 'unbelieving', 'uncertain', 'unclear', 'uncomfortable', 'unconcerned', 'unconfirmed', 'unconvinced', 'uncredited', 'undecided', 'underestimate', 'underestimated', 'underestimates', 'underestimating', 'undermine', 'undermined', 'undermines', 'undermining', 'undeserving', 'undesirable', 'uneasy', 'unemployment', 'unequal', 'unethical', 'unfair', 'unfocused', 'unfulfilled', 'unhappy', 'unhealthy', 'unimpressed', 'unintelligent', 'unjust', 'unlovable', 'unloved', 'unmotivated', 'unprofessional', 'unresearched', 'unsatisfied', 'unsecured', 'unsettled', 'unsophisticated', 'unstable', 'unsupported', 'unsure', 'unwanted', 'unworthy', 'upset', 'upsets', 'upsetting', 'uptight', 'urgent', 'useless', 'uselessness', 'vague', 'verdict', 'verdicts', 'vexation', 'vexing', 'vicious', 'victim', 'victimize', 'victimized', 'victimizes', 'victimizing', 'victims', 'vile', 'violate', 'violated', 'violates', 'violating', 'violence', 'violent', 'virulent', 'vitriolic', 'vociferous', 'vulnerability', 'vulnerable', 'walkout', 'walkouts', 'wanker', 'war', 'warfare', 'warn', 'warned', 'warning', 'warnings', 'warns', 'waste', 'wasted', 'wasting', 'wavering', 'weak', 'weakness', 'weary', 'weep', 'weeping', 'weird', 'whitewash', 'whore', 'wicked', 'widowed', 'withdrawal', 'woebegone', 'woeful', 'worn', 'worried', 'worry', 'worrying', 'worse', 'worsen', 'worsened', 'worsening', 'worsens', 'worst', 'worthless', 'wrathful', 'wreck', 'wrong', 'wronged', 'wtf', 'yucky', 'zealot', 'zealots'])

def lexicon_features(tokens, feats):
    
    tokens_items,negative_items,positive_items =[],[],[]
    
    for tokenize_string in tokens:
        tokens_items.append(tokenize_string.upper())
    
    
    for negative_string in neg_words:
        negative_items.append(negative_string.upper())
        
        
    for positive_string in pos_words:
        positive_items.append(positive_string.upper())
            
    total_positive_count= set(tokens_items).intersection(positive_items)
    total_negative_count= set(tokens_items).intersection(negative_items)
    feats['pos_words']=len(total_positive_count)
    feats['neg_words']=len(total_negative_count) 

    pass  

def featurize(tokens, feature_fns):
    feats = defaultdict(lambda: 0)
    
    for functions in feature_fns:
        functions(tokens, feats)
        
    return sorted(feats.items(), key=lambda x: x[0])
    pass

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    total_doc_list,vocab_dicts,data,indices,indptr =[],{},[],[],[]
    
    for token_items in tokens_list:
        doc_list = featurize(token_items,feature_fns)
        for documents in doc_list:
            vocab_dicts[documents[0]]=documents[1]   
        total_doc_list.append(doc_list)

    if vocab == None:
        vocab={}
        for vocab_items in sorted(vocab_dicts.keys()):
            vocab[vocab_items]=len(vocab)

    values_dictionary ={}
    for docs_items in total_doc_list:
        for val_items in docs_items:
            if val_items[0] not in values_dictionary:
                values_dictionary[val_items[0]]=1
            else: 
                values_dictionary[val_items[0]]=values_dictionary[val_items[0]]+1
                        
    for doc_items in range(len(total_doc_list)):
        for dic_items in total_doc_list[doc_items]:
            if (values_dictionary[dic_items[0]] >= min_freq):
                if dic_items[0] in vocab:
                    data.append(dic_items[1])
                    indices.append(doc_items)
                    indptr.append(vocab[dic_items[0]])
    
    data_array = np.array(data,np.int64)
    indices_array = np.array(indices,np.int64)
    indptr_array = np.array(indptr,np.int64)
    
    return csr_matrix((data_array, (indices_array, indptr_array)), shape=(len(total_doc_list),len(vocab))),vocab  
    pass

def accuracy_score(truth, predicted):
    return len(np.where(truth==predicted)[0]) / len(truth)

def cross_validation_accuracy(clf, X, labels, k):
    testing_accuracies = ()
    cross_validation_calculation = KFold(len(labels), k)
    for data_train, data_test in cross_validation_calculation:
        clf.fit(X[data_train], labels[data_train])
        new_values = clf.predict(X[data_test])
        accuracies_values = accuracy_score(labels[data_test], new_values)
        testing_accuracies += (accuracies_values,)
    average_testing_accuracy = np.mean(list(testing_accuracies))
    return average_testing_accuracy
    pass

def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    all_combinations =[]
    for features in range(1,len(feature_fns)+1):
        all_combinations.append(combinations(feature_fns,features))
    combined_values=[]
    all_final_combinations=[[]]
    for list_vals in all_combinations:
        for vals in list_vals:
            combined_values.append(vals)
    all_data = [tuple(all_data_val) for all_data_val in (punct_vals,combined_values,min_freqs)]
    for all_data_val in all_data:
        all_final_combinations=[com+[val] for com in all_final_combinations for val in all_data_val]
    dicts_list=()
    for prod in all_final_combinations:
        comb_dicts={}
        prouct_values = tuple(prod)
        comb_dicts['punct']=prouct_values[0]
        comb_dicts['features']=prouct_values[1]
        comb_dicts['min_freq']=prouct_values[2]
        classifier = LogisticRegression()
        tokens_list = [tokenize(doc_list,prouct_values[0]) for doc_list in docs]
        csr_matrix,v= vectorize(tokens_list,list(prouct_values[1]),prouct_values[2])
        comb_dicts['accuracy'] = cross_validation_accuracy(classifier,csr_matrix,labels,5)
        dicts_list += (comb_dicts,)
    final_sorted_dicts=sorted(list(dicts_list), key=lambda k: k['accuracy'],reverse=True) 
    return final_sorted_dicts
    pass

def plot_sorted_accuracies(results):
    
    accuracies_list=[]
    for accuracy in results:
        accuracies_list.append(accuracy['accuracy'])
    
    x = range(0,len(accuracies_list))
    y = sorted(accuracies_list)
    
    plt.plot(x, y)
    plt.xlabel('setting')
    plt.ylabel('accuracy')
    plt.savefig(r'accuracies.png')
    pass

def mean_accuracy_per_setting(results):
    
    mean_dict={}
    final_list =[]
    
    for values in results:
        for key,val in values.items():
            if key!='accuracy':
                if key=='features':
                    feature_funcs=[]
                    for func_names in val:
                        feature_funcs.append(func_names.__name__)
                    val=feature_funcs
                if key+str(val) in mean_dict.keys():
                    mean_dict[key+str(val)].append(values['accuracy'])
                else:
                    mean_dict[key+str(val)]=[values['accuracy']]
       
    for setting,acc in mean_dict.items():
        Addition_values=0
        for acc_vals in acc:
            Addition_values = Addition_values + acc_vals
        final_accuracy=float(Addition_values)/len(acc)
        final_list.append((final_accuracy,setting))

    return final_list
    pass

def fit_best_classifier(docs, labels, best_result):
    classifier= LogisticRegression()
    tokenized_list= [tokenize(documents,best_result['punct']) for documents in docs]
    csr_matrix,vocab= vectorize(tokenized_list,list(best_result['features']),best_result['min_freq'])
    classifier.fit(csr_matrix,labels)
    return classifier,vocab
    pass

def parse_test_data(best_result, vocab):    
    test_docs, test_labels,p1,n1 = read_Test_data()
    tokenized_list= [tokenize(documents,best_result['punct']) for documents in test_docs]
    csr_matrix,cols= vectorize(tokenized_list,list(best_result['features']),best_result['min_freq'],vocab)
    return test_docs,test_labels,csr_matrix,p1,n1
    pass

def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    

    sys.stdout = open('summary_file.txt','a')
    feature_fns = [token_features, token_pair_features, lexicon_features]
    docs,labels,p,n,tp,tn = read_Train_data()
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Parse test data
    test_docs, test_labels, X_test,p2,n2 = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))
    
    print('Number of instances per class found:')
    print('Positive= %d' %(p+p2))
    print('Negative= %d' %(n+n2))

    print('One example from each class:')
    print('Positive_tweet= %s' %(tp.encode("utf-8")))
    print('Negative_tweet= %s' %(tn.encode("utf-8")))

                
if __name__ == '__main__':
    main()
   