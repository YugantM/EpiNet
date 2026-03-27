"""
Tiny Q&A Dataset for EpiNet evaluation.

Format: answer-selection (binary classification).
  Input  : tokenised "question [SEP] candidate_answer"
  Label  : 1 = correct answer,  0 = distractor

The dataset covers 7 topic clusters deliberately chosen so that:
  - A context-aware model (EpiNet) should improve as topic context accumulates.
  - Topic switches are sharp, probing the modulation state's ability to adapt.

Topics: geography, science, history, technology, sport, food, literature

Each question has exactly 1 correct answer and 3 distractors.
→ 4 binary examples per question.

Total: 80 questions × 4 = 320 examples (used as train/val split).
"""

import random
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# Raw Q&A data
# ──────────────────────────────────────────────────────────────────────────────

_QA_RAW: List[Dict] = [
    # ── Geography ──────────────────────────────────────────────────────────────
    {"q": "What is the capital of France?",            "a": "Paris",        "d": ["London", "Berlin", "Madrid"],           "topic": "geography"},
    {"q": "Which country has the largest land area?",  "a": "Russia",       "d": ["Canada", "China", "USA"],               "topic": "geography"},
    {"q": "What is the longest river in the world?",   "a": "Nile",         "d": ["Amazon", "Yangtze", "Mississippi"],     "topic": "geography"},
    {"q": "Which continent is the Sahara Desert on?",  "a": "Africa",       "d": ["Asia", "Australia", "South America"],   "topic": "geography"},
    {"q": "What is the smallest country in the world?","a": "Vatican City", "d": ["Monaco", "San Marino", "Liechtenstein"],"topic": "geography"},
    {"q": "Which ocean is the largest?",               "a": "Pacific",      "d": ["Atlantic", "Indian", "Arctic"],         "topic": "geography"},
    {"q": "In which country is Mount Everest located?","a": "Nepal",        "d": ["Tibet", "India", "Bhutan"],             "topic": "geography"},
    {"q": "What is the capital of Japan?",             "a": "Tokyo",        "d": ["Osaka", "Kyoto", "Hiroshima"],          "topic": "geography"},
    {"q": "Which country has the most natural lakes?", "a": "Canada",       "d": ["Russia", "Finland", "USA"],             "topic": "geography"},
    {"q": "What is the capital of Australia?",         "a": "Canberra",     "d": ["Sydney", "Melbourne", "Brisbane"],      "topic": "geography"},
    {"q": "Which is the deepest lake in the world?",   "a": "Lake Baikal",  "d": ["Lake Superior", "Caspian Sea", "Lake Tanganyika"], "topic": "geography"},
    {"q": "What is the highest mountain in Africa?",   "a": "Kilimanjaro",  "d": ["Mount Kenya", "Atlas", "Rwenzori"],     "topic": "geography"},

    # ── Science ────────────────────────────────────────────────────────────────
    {"q": "What is the chemical symbol for gold?",          "a": "Au",       "d": ["Ag", "Fe", "Gd"],                      "topic": "science"},
    {"q": "How many bones are in the adult human body?",    "a": "206",      "d": ["198", "214", "220"],                    "topic": "science"},
    {"q": "What planet is closest to the Sun?",             "a": "Mercury",  "d": ["Venus", "Earth", "Mars"],               "topic": "science"},
    {"q": "What is the speed of light in km/s?",            "a": "300000",   "d": ["150000", "450000", "186000"],           "topic": "science"},
    {"q": "What is the most abundant gas in Earth's atmosphere?", "a": "Nitrogen", "d": ["Oxygen", "Carbon dioxide", "Argon"], "topic": "science"},
    {"q": "What force keeps planets in orbit?",             "a": "Gravity",  "d": ["Magnetism", "Friction", "Electrostatics"], "topic": "science"},
    {"q": "What is the atomic number of carbon?",           "a": "6",        "d": ["4", "8", "12"],                         "topic": "science"},
    {"q": "What part of the cell contains DNA?",            "a": "Nucleus",  "d": ["Mitochondria", "Ribosome", "Vacuole"],  "topic": "science"},
    {"q": "What is the unit of electrical resistance?",     "a": "Ohm",      "d": ["Volt", "Watt", "Ampere"],               "topic": "science"},
    {"q": "Which planet has the most moons?",               "a": "Saturn",   "d": ["Jupiter", "Neptune", "Uranus"],         "topic": "science"},
    {"q": "What is the powerhouse of the cell?",            "a": "Mitochondria", "d": ["Nucleus", "Ribosome", "Golgi body"], "topic": "science"},
    {"q": "What is H2O commonly known as?",                 "a": "Water",    "d": ["Hydrogen peroxide", "Salt water", "Hydroxide"], "topic": "science"},

    # ── History ────────────────────────────────────────────────────────────────
    {"q": "In which year did World War II end?",            "a": "1945",     "d": ["1943", "1944", "1946"],                 "topic": "history"},
    {"q": "Who was the first President of the United States?", "a": "George Washington", "d": ["Thomas Jefferson", "John Adams", "Benjamin Franklin"], "topic": "history"},
    {"q": "In which year did the Berlin Wall fall?",        "a": "1989",     "d": ["1987", "1991", "1985"],                 "topic": "history"},
    {"q": "Who wrote the Declaration of Independence?",     "a": "Thomas Jefferson", "d": ["Benjamin Franklin", "John Adams", "James Madison"], "topic": "history"},
    {"q": "Which empire built the Colosseum?",              "a": "Roman",    "d": ["Greek", "Ottoman", "Byzantine"],        "topic": "history"},
    {"q": "In what year did the Titanic sink?",             "a": "1912",     "d": ["1914", "1908", "1916"],                 "topic": "history"},
    {"q": "Who was the first person to walk on the Moon?",  "a": "Neil Armstrong", "d": ["Buzz Aldrin", "Yuri Gagarin", "John Glenn"], "topic": "history"},
    {"q": "Which country was the first to give women the right to vote?", "a": "New Zealand", "d": ["Australia", "Finland", "Norway"], "topic": "history"},
    {"q": "What year did the French Revolution begin?",     "a": "1789",     "d": ["1776", "1799", "1804"],                 "topic": "history"},
    {"q": "Who discovered penicillin?",                     "a": "Alexander Fleming", "d": ["Louis Pasteur", "Robert Koch", "Marie Curie"], "topic": "history"},
    {"q": "Which ancient wonder was located in Alexandria?","a": "Lighthouse of Alexandria", "d": ["Colossus of Rhodes", "Hanging Gardens", "Statue of Zeus"], "topic": "history"},
    {"q": "In which city was Julius Caesar assassinated?",  "a": "Rome",     "d": ["Athens", "Carthage", "Alexandria"],     "topic": "history"},

    # ── Technology ─────────────────────────────────────────────────────────────
    {"q": "Who co-founded Apple with Steve Jobs?",          "a": "Steve Wozniak", "d": ["Bill Gates", "Paul Allen", "Jony Ive"], "topic": "technology"},
    {"q": "What does CPU stand for?",                       "a": "Central Processing Unit", "d": ["Computer Processing Unit", "Core Processing Utility", "Central Program Unit"], "topic": "technology"},
    {"q": "Which company created the Python programming language?", "a": "No company — Guido van Rossum created it", "d": ["Microsoft", "Google", "Sun Microsystems"], "topic": "technology"},
    {"q": "What does HTTP stand for?",                      "a": "HyperText Transfer Protocol", "d": ["High Transfer Text Protocol", "Hyper Terminal Transfer Protocol", "HyperText Transmission Process"], "topic": "technology"},
    {"q": "What does RAM stand for?",                       "a": "Random Access Memory", "d": ["Read Access Memory", "Rapid Application Memory", "Remote Access Module"], "topic": "technology"},
    {"q": "Which language is primarily used for web page structure?", "a": "HTML", "d": ["CSS", "JavaScript", "PHP"],        "topic": "technology"},
    {"q": "What is the name of Google's mobile operating system?", "a": "Android", "d": ["iOS", "HarmonyOS", "Symbian"],    "topic": "technology"},
    {"q": "Who invented the World Wide Web?",               "a": "Tim Berners-Lee", "d": ["Vint Cerf", "Bill Gates", "Marc Andreessen"], "topic": "technology"},
    {"q": "What does GPU stand for?",                       "a": "Graphics Processing Unit", "d": ["General Purpose Unit", "Graphical Parallel Unit", "Grid Processing Utility"], "topic": "technology"},
    {"q": "What year was the first iPhone released?",       "a": "2007",     "d": ["2005", "2008", "2010"],                 "topic": "technology"},
    {"q": "Which data structure uses LIFO ordering?",       "a": "Stack",    "d": ["Queue", "Heap", "Tree"],                "topic": "technology"},
    {"q": "What does SQL stand for?",                       "a": "Structured Query Language", "d": ["Sequential Query Logic", "Structured Queue Language", "System Query Layer"], "topic": "technology"},

    # ── Sport ──────────────────────────────────────────────────────────────────
    {"q": "How many players are on a standard football (soccer) team?", "a": "11", "d": ["10", "9", "12"],                  "topic": "sport"},
    {"q": "In which sport would you perform a slam dunk?",  "a": "Basketball","d": ["Volleyball", "Tennis", "Baseball"],     "topic": "sport"},
    {"q": "How many Grand Slam tournaments are there in tennis?", "a": "4",  "d": ["3", "5", "6"],                          "topic": "sport"},
    {"q": "Which country invented cricket?",                "a": "England",  "d": ["India", "Australia", "South Africa"],   "topic": "sport"},
    {"q": "How long is a standard marathon in km?",         "a": "42.195",   "d": ["40", "45", "38.5"],                     "topic": "sport"},
    {"q": "What sport is played at Wimbledon?",             "a": "Tennis",   "d": ["Cricket", "Croquet", "Badminton"],      "topic": "sport"},
    {"q": "How many rings are on the Olympic flag?",        "a": "5",        "d": ["4", "6", "7"],                          "topic": "sport"},
    {"q": "Which country hosts the Tour de France cycling race?", "a": "France", "d": ["Italy", "Spain", "Belgium"],        "topic": "sport"},
    {"q": "What is the maximum score in ten-pin bowling?",  "a": "300",      "d": ["200", "250", "400"],                    "topic": "sport"},
    {"q": "In which sport is a shuttlecock used?",          "a": "Badminton","d": ["Tennis", "Squash", "Volleyball"],       "topic": "sport"},
    {"q": "How many points is a touchdown worth in American football?", "a": "6", "d": ["7", "3", "5"],                     "topic": "sport"},
    {"q": "Which country won the first FIFA World Cup in 1930?", "a": "Uruguay", "d": ["Brazil", "Argentina", "Italy"],     "topic": "sport"},

    # ── Food ───────────────────────────────────────────────────────────────────
    {"q": "What is the main ingredient in guacamole?",      "a": "Avocado",  "d": ["Tomato", "Lime", "Onion"],              "topic": "food"},
    {"q": "From which country does sushi originate?",       "a": "Japan",    "d": ["China", "Korea", "Thailand"],           "topic": "food"},
    {"q": "What type of pastry is used to make croissants?","a": "Puff pastry","d": ["Choux pastry", "Shortcrust pastry", "Filo pastry"], "topic": "food"},
    {"q": "What is the most widely consumed meat in the world?", "a": "Pork","d": ["Chicken", "Beef", "Lamb"],              "topic": "food"},
    {"q": "What ingredient makes bread rise?",              "a": "Yeast",    "d": ["Baking powder", "Salt", "Sugar"],       "topic": "food"},
    {"q": "What is the Italian word for pie (as used in pizza)?", "a": "Pizza","d": ["Torte", "Focaccia", "Calzone"],        "topic": "food"},
    {"q": "Which nut is used to make marzipan?",            "a": "Almond",   "d": ["Walnut", "Hazelnut", "Cashew"],         "topic": "food"},
    {"q": "What is the primary ingredient in hummus?",      "a": "Chickpeas","d": ["Lentils", "White beans", "Tofu"],       "topic": "food"},
    {"q": "From which fruit is wine primarily made?",       "a": "Grapes",   "d": ["Plums", "Berries", "Apples"],           "topic": "food"},
    {"q": "What gives chilli peppers their heat?",          "a": "Capsaicin","d": ["Piperine", "Allicin", "Gingerol"],      "topic": "food"},
    {"q": "What is parmesan primarily made from?",          "a": "Cow's milk","d": ["Goat's milk", "Sheep's milk", "Buffalo milk"], "topic": "food"},
    {"q": "What type of bean is used to make tofu?",        "a": "Soybean",  "d": ["Kidney bean", "Black bean", "Chickpea"], "topic": "food"},

    # ── Literature ─────────────────────────────────────────────────────────────
    {"q": "Who wrote '1984'?",                              "a": "George Orwell", "d": ["Aldous Huxley", "Ray Bradbury", "H.G. Wells"], "topic": "literature"},
    {"q": "Who wrote 'Pride and Prejudice'?",               "a": "Jane Austen", "d": ["Charlotte Brontë", "George Eliot", "Mary Shelley"], "topic": "literature"},
    {"q": "What is the first book of the Bible?",           "a": "Genesis",  "d": ["Exodus", "Psalms", "Proverbs"],         "topic": "literature"},
    {"q": "Who wrote 'The Great Gatsby'?",                  "a": "F. Scott Fitzgerald", "d": ["Ernest Hemingway", "John Steinbeck", "William Faulkner"], "topic": "literature"},
    {"q": "In which Shakespeare play does the character Ophelia appear?", "a": "Hamlet", "d": ["Macbeth", "Othello", "King Lear"], "topic": "literature"},
    {"q": "Who wrote 'Don Quixote'?",                       "a": "Miguel de Cervantes", "d": ["Gabriel García Márquez", "Jorge Luis Borges", "Pablo Neruda"], "topic": "literature"},
    {"q": "What is the name of Harry Potter's pet owl?",    "a": "Hedwig",   "d": ["Errol", "Pigwidgeon", "Fawkes"],        "topic": "literature"},
    {"q": "Who wrote 'Crime and Punishment'?",              "a": "Fyodor Dostoevsky", "d": ["Leo Tolstoy", "Anton Chekhov", "Ivan Turgenev"], "topic": "literature"},
    {"q": "In which novel would you find the character Jay Gatsby?", "a": "The Great Gatsby", "d": ["The Sun Also Rises", "Tender Is the Night", "This Side of Paradise"], "topic": "literature"},
    {"q": "Who wrote 'Brave New World'?",                   "a": "Aldous Huxley", "d": ["George Orwell", "H.G. Wells", "Arthur C. Clarke"], "topic": "literature"},
    {"q": "What is the subtitle of Frankenstein?",          "a": "The Modern Prometheus", "d": ["A Gothic Tale", "The New Adam", "The Creature"], "topic": "literature"},
    {"q": "Who is the author of the Sherlock Holmes stories?", "a": "Arthur Conan Doyle", "d": ["Agatha Christie", "Edgar Allan Poe", "Wilkie Collins"], "topic": "literature"},
]


# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary and tokeniser
# ──────────────────────────────────────────────────────────────────────────────

class QAVocabulary:
    """Character-free word-level vocabulary with a [SEP] special token."""

    PAD, UNK, BOS, EOS, SEP = 0, 1, 2, 3, 4

    def __init__(self) -> None:
        self._w2i: Dict[str, int] = {
            "<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "<SEP>": 4
        }
        self._i2w: Dict[int, str] = {v: k for k, v in self._w2i.items()}

    def build(self, texts: List[str]) -> None:
        seen = set(self._w2i.keys())
        for text in texts:
            for tok in _tokenise(text):
                if tok not in seen:
                    idx = len(self._w2i)
                    self._w2i[tok] = idx
                    self._i2w[idx] = tok
                    seen.add(tok)

    def encode(self, question: str, answer: str, max_len: int = 64) -> List[int]:
        q_toks = _tokenise(question)
        a_toks = _tokenise(answer)
        ids = [self.BOS] + [self._w2i.get(t, self.UNK) for t in q_toks]
        ids += [self.SEP] + [self._w2i.get(t, self.UNK) for t in a_toks]
        ids += [self.EOS]
        ids = ids[:max_len]
        ids += [self.PAD] * (max_len - len(ids))
        return ids

    def decode(self, ids: List[int]) -> str:
        return " ".join(
            self._i2w.get(i, "<UNK>")
            for i in ids
            if i not in (self.PAD, self.BOS, self.EOS)
        )

    def __len__(self) -> int:
        return len(self._w2i)


def _tokenise(text: str) -> List[str]:
    """Simple lowercase word tokeniser; keeps punctuation as separate tokens."""
    import re
    tokens = re.findall(r"[a-z0-9]+|[^\w\s]", text.lower())
    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ──────────────────────────────────────────────────────────────────────────────

def build_qa_examples(
    raw: List[Dict] = None,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[int], List[str]]:
    """
    Expand raw Q&A pairs into (question, candidate, label, topic) tuples.

    Returns
    -------
    questions  : list of question strings
    candidates : list of candidate answer strings
    labels     : list of 0/1 ints
    topics     : list of topic tag strings
    """
    if raw is None:
        raw = _QA_RAW
    random.seed(seed)

    questions, candidates, labels, topics = [], [], [], []
    for item in raw:
        # Positive example
        questions.append(item["q"])
        candidates.append(item["a"])
        labels.append(1)
        topics.append(item["topic"])
        # Negative examples
        for d in item["d"]:
            questions.append(item["q"])
            candidates.append(d)
            labels.append(0)
            topics.append(item["topic"])

    # Shuffle together
    combined = list(zip(questions, candidates, labels, topics))
    random.shuffle(combined)
    questions, candidates, labels, topics = map(list, zip(*combined))
    return questions, candidates, labels, topics


class QADataset(Dataset):
    """
    PyTorch Dataset for Q&A answer selection.

    Each item is a padded token sequence encoding "question [SEP] candidate"
    and a binary label (1 = correct, 0 = distractor).
    """

    def __init__(
        self,
        questions:  List[str],
        candidates: List[str],
        labels:     List[int],
        topics:     List[str],
        vocab:      QAVocabulary,
        max_len:    int = 64,
    ) -> None:
        self.topics = topics
        self.raw_questions  = questions
        self.raw_candidates = candidates
        self.labels    = torch.tensor(labels, dtype=torch.long)
        self.token_ids = torch.tensor(
            [vocab.encode(q, a, max_len) for q, a in zip(questions, candidates)],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.token_ids[idx], self.labels[idx]


def build_qa_loaders(
    batch_size: int = 16,
    max_len:    int = 64,
    val_split:  float = 0.2,
    seed:       int = 42,
) -> Tuple[DataLoader, DataLoader, QAVocabulary, Dict]:
    """
    Build train/val DataLoaders for the Q&A task.

    Returns
    -------
    train_loader, val_loader, vocab, meta
      meta contains per-topic counts and total example counts.
    """
    questions, candidates, labels, topics = build_qa_examples(seed=seed)

    vocab = QAVocabulary()
    vocab.build(questions + candidates)

    n = len(questions)
    n_val   = int(n * val_split)
    n_train = n - n_val

    # Stratified-ish split: keep topic distribution similar in both splits
    random.seed(seed + 1)
    indices = list(range(n))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    def make_ds(idx):
        return QADataset(
            [questions[i]  for i in idx],
            [candidates[i] for i in idx],
            [labels[i]     for i in idx],
            [topics[i]     for i in idx],
            vocab, max_len,
        )

    train_ds = make_ds(train_idx)
    val_ds   = make_ds(val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    from collections import Counter
    meta = {
        "n_total":      n,
        "n_train":      n_train,
        "n_val":        n_val,
        "vocab_size":   len(vocab),
        "n_positive":   sum(labels),
        "n_negative":   n - sum(labels),
        "topics":       dict(Counter(topics)),
    }
    return train_loader, val_loader, vocab, meta


# ──────────────────────────────────────────────────────────────────────────────
# Interactive evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_questions_by_topic(topic: str) -> List[Dict]:
    """Return all raw Q&A items for a given topic."""
    return [item for item in _QA_RAW if item["topic"] == topic]


def topics() -> List[str]:
    """Return the list of unique topics in order."""
    seen, result = set(), []
    for item in _QA_RAW:
        if item["topic"] not in seen:
            result.append(item["topic"])
            seen.add(item["topic"])
    return result
