"""
Member 1 — Preprocessing
Normalization functions for business entity resolution.
"""
import re
import unicodedata
import pandas as pd

CHUNK = 200_000


def load_sample(path, n=25_000, seed=42):
    """Load a sample. Tries UTF-8, falls back to Latin-1."""
    try:
        chunks = [c for c in pd.read_csv(path, sep='\t', chunksize=CHUNK,
                                          dtype=str, encoding='utf-8',
                                          encoding_errors='replace')]
    except UnicodeDecodeError:
        chunks = [c for c in pd.read_csv(path, sep='\t', chunksize=CHUNK,
                                          dtype=str, encoding='latin-1')]
    df = pd.concat(chunks, ignore_index=True)
    return df.sample(n=min(n, len(df)), random_state=seed)


# ═══════════════════════════════════════════════════
# BUSINESS NAME
# ═══════════════════════════════════════════════════

LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'co', 'company',
    'ltd', 'limited', 'llc', 'llp', 'lp', 'plc', 'pvt', 'private',
    'gmbh', 'sa', 'sarl', 'bv', 'nv', 'ag', 'oy', 'ab', 'as',
    'aps', 'srl', 'spa', 'kk', 'pte', 'sdn', 'bhd', 'sl', 'sas',
    'snc', 'scs', 'sca', 'eirl',
}

NAME_ABBREV = {
    '&': 'and',
    'intl': 'international',
    'int': 'international',
    'tech': 'technology',
    'svcs': 'services',
    'svc': 'service',
    'mgmt': 'management',
    'grp': 'group',
    'ent': 'enterprises',
    'assoc': 'associates',
    'bros': 'brothers',
    'mfg': 'manufacturing',
    'natl': 'national',
}


def normalize_business_name(name):
    """Lowercase, strip punctuation, expand abbrevs, remove legal suffixes."""
    if not isinstance(name, str) or not name:
        return ""
    name = unicodedata.normalize('NFKD', name).lower()
    name = re.sub(r'[^\w\s]', ' ', name)
    toks = [NAME_ABBREV.get(t, t) for t in name.split()]
    toks = [t for t in toks if t not in LEGAL_SUFFIXES]
    return ' '.join(toks).strip()


def name_blocking_key(name, n=4):
    """First n alphanumeric chars of normalized name."""
    s = re.sub(r'[^a-z0-9]', '', normalize_business_name(name))
    return s[:n]


# ═══════════════════════════════════════════════════
# COUNTRY
# ═══════════════════════════════════════════════════

COUNTRY_ALIASES = {
    'us': 'united states', 'usa': 'united states', 'u.s.': 'united states',
    'u.s.a.': 'united states', 'america': 'united states',
    'united states of america': 'united states',
    'uk': 'united kingdom', 'u.k.': 'united kingdom',
    'great britain': 'united kingdom', 'england': 'united kingdom',
    'in': 'india', 'ind': 'india', 'bharat': 'india',
    'fr': 'france', 'fra': 'france',
}


def normalize_country(country):
    """Canonical country name. Preserves unknown values."""
    if not isinstance(country, str) or not country:
        return ""
    return COUNTRY_ALIASES.get(country.lower().strip(), country.lower().strip())


# ═══════════════════════════════════════════════════
# POSTAL CODE
# ═══════════════════════════════════════════════════

def extract_postal_code(addr, country=""):
    """Extract ZIP (US), PIN (India), or generic postal code."""
    if not isinstance(addr, str) or not addr:
        return ""
    c = normalize_country(country)
    if c == 'united states':
        m = re.search(r'\b(\d{5})(?:-\d{4})?\b', addr)
    elif c == 'india':
        m = re.search(r'\b(\d{6})\b', addr)
    else:
        m = re.search(r'\b(\d{5,6})\b', addr)
    return m.group(1) if m else ""


# ═══════════════════════════════════════════════════
# ADDRESS
# ═══════════════════════════════════════════════════

ADDR_ABBREV = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'av': 'avenue',
    'blvd': 'boulevard', 'dr': 'drive', 'ln': 'lane', 'ct': 'court',
    'pl': 'place', 'sq': 'square', 'fl': 'floor', 'bldg': 'building',
    'apt': 'apartment', 'ste': 'suite', 'hwy': 'highway',
    'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
    'nr': 'near', 'opp': 'opposite',
    'r': 'rue', 'bd': 'boulevard', 'rte': 'route',
    'che': 'chemin', 'imp': 'impasse', 'all': 'allee',
}


def normalize_address(addr):
    """Lowercase, expand abbreviations, strip punctuation."""
    if not isinstance(addr, str) or not addr:
        return ""
    addr = unicodedata.normalize('NFKD', addr).lower()
    addr = re.sub(r'[^\w\s]', ' ', addr)
    toks = [ADDR_ABBREV.get(t, t) for t in addr.split()]
    return ' '.join(toks).strip()


def extract_house_number(addr):
    """First numeric token in address."""
    if not isinstance(addr, str) or not addr:
        return ""
    m = re.search(r'\b(\d+[A-Za-z]?)\b', addr)
    return m.group(1) if m else ""


# ═══════════════════════════════════════════════════
# DATAFRAME PIPELINE
# ═══════════════════════════════════════════════════

def preprocess_dataframe(df):
    """Apply all normalizations. Returns new df with extra columns."""
    df = df.copy()
    df['business_name_norm'] = df['business_name'].apply(normalize_business_name)
    df['business_name_key'] = df['business_name'].apply(name_blocking_key)
    df['business_address_norm'] = df['business_address'].apply(normalize_address)
    df['country_norm'] = df['country'].apply(normalize_country)
    df['postal_code'] = df.apply(
        lambda r: extract_postal_code(r['business_address'], r['country']),
        axis=1
    )
    df['house_number'] = df['business_address'].apply(extract_house_number)
    return df