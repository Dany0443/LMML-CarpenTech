import re
import unicodedata
from collections import Counter
from functools import cache


NAMES = (
    'Alimentum',
    'Aromi',
    'Bibimbap House',
    'Blue Spice',
    'Browns Cambridge',
    'Clowns',
    'Cocum',
    'Cotto',
    'Fitzbillies',
    'Giraffe',
    'Green Man',
    'Loch Fyne',
    'Midsummer House',
    'Strada',
    'Taste of Cambridge',
    'Cambridge Blue',
    'Cricketers',
    'Dumpling Tree',
    'Eagle',
    'Golden Curry',
    'Golden Palace',
    'Mill',
    'Olive Grove',
    'Phoenix',
    'Plough',
    'Punter',
    'Rice Boat',
    'Twenty Two',
    'Vaults',
    'Waterman',
    'Wrestlers',
    'Travellers Rest Beefeater',
    'Wildwood',
    'Zizzi',
    'All Bar One',
    'Café Adriatic',
    'Café Brazil',
    'Café Rouge',
    'Café Sicilia',
    'Clare Hall',
    'Crowne Plaza Hotel',
    'Rainbow Vegetarian Café',
    'Raja Indian Cuisine',
    'Ranch',
    'Burger King',
    'Yippee Noodle Bar',
    'Sorrento',
    'Portland Arms',
    'Bakers',
)

PATTERNS = {
    'food:japanese': 'japan|sushi',
    'food:english': 'english|british',
    'food:french': 'french|wine.*cheese|cheese.*wine',
    'food:italian': 'italian|pasta|spaghetti',
    'food:indian': 'indian',
    'food:chinese': 'chinese',
    'food:fast': 'fast food|burger|hamburger|fries',
    'type:coffee': 'coffee|caf[eé]',
    'type:pub': '\\bpub|\\bbar\\b',
    'area:river': 'river',
    'price:low': 'cheap|inexpensive|low.{0,3}(?:price|cost)|less than.{0,12}20|under.{0,12}20|below.{0,5}20|budget|affordable',
    'price:mid': 'moderat|mid.{0,3}(?:price|range)|medium.{0,3}(?:price|range)|average.{0,3}price|20\\s*(?:to|and)?\\s*(?:pounds\\s*)?25|reasonable.{0,3}pric|fairly.{0,3}pric',
    'price:high': 'expensive|high.{0,4}(?:price|cost|end)|more than.{0,12}30|over.{0,12}30|above.{0,12}30|30.*(?:up|and)|big spender|higher budget',
    'rating:low': '1 (?:out|star)|one star|one out|low.{0,10}(?:rat|review)|poor.{0,10}rat|rat.{0,8}low',
    'rating:mid': '3 (?:out|star)|three star|three out|average.{0,12}(?:rat|review)|(?:rat|review).{0,12}average|decent rat|moderate customer rat',
    'rating:high': '5 (?:out|star)|five star|five out|high.{0,12}(?:rat|review)|(?:rat|review).{0,12}high|excellent.{0,12}(?:rat|review)',
}


def normalize(text):
    letters = unicodedata.normalize('NFKD', text.lower())
    text = ''.join(letter for letter in letters if not unicodedata.combining(letter))
    text = text.replace('-', ' ').replace('£', ' pounds ')
    return re.sub(r'\s+', ' ', text)


@cache
def slots(text):
    text = normalize(text)
    concepts = set()
    for name in NAMES:
        name = normalize(name)
        if re.search(r'\b' + re.escape(name) + r'\b', text):
            concepts.add('entity:' + name)
            # indian in a landmark name must not become the venue's cuisine
            text = text.replace(name, ' ')
    for concept, pattern in PATTERNS.items():
        if re.search(pattern, text):
            concepts.add(concept)
    negative = (
        r'not.{0,15}(?:family|child|kid)|non.{0,4}(?:family|child|kid)|'
        r'no (?:family|child|kid)|adult|(?:isn.t|aren.t).{0,4}family'
    )
    if re.search(negative, text):
        concepts.add('family:no')
    elif re.search(r'famil|child|kid', text):
        concepts.add('family:yes')
    return frozenset(concepts)


def expected(refs, votes=3):
    counts = Counter(concept for text in refs for concept in slots(text))
    return {concept for concept, count in counts.items() if count >= votes}
