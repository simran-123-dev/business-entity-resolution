import csv
import re
import sqlite3
from pathlib import Path

BASE = Path("dataset")
TEST = BASE / "test"
OUTPUT = Path("blocking_output")

OUTPUT.mkdir(exist_ok=True)

DB_PATH = OUTPUT / "blocking_index.db"
CANDIDATE_FILE = OUTPUT / "candidate_pairs.tsv"

MAX_BLOCK_SIZE = 200


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


LEGAL_WORDS = {
    "inc", "incorporated", "llc", "ltd", "limited",
    "corp", "corporation", "co", "company",
    "private", "pvt", "plc", "llp", "the"
}


# =========================================================
# BLOCKING KEYS
# =========================================================

def generate_block_keys(name, address, country):

    country = normalize_text(country)
    name = normalize_text(name)
    address = normalize_text(address)

    keys = set()

    # 1. Exact normalized name
    if name:
        keys.add(f"N|{country}|{name}")

    # 2. Name prefix
    tokens = [
        x for x in name.split()
        if len(x) >= 4 and x not in LEGAL_WORDS
    ]

    if tokens:
        keys.add(f"P|{country}|{tokens[0][:4]}")

    # 3. Postal / PIN
    postal = re.findall(r"\b\d{5,6}\b", address)

    if postal:
        keys.add(f"Z|{country}|{postal[0]}")

    # 4. House number + address word
    house = re.search(r"\b\d+[A-Za-z]?\b", address)

    address_tokens = [
        x for x in address.split()
        if len(x) >= 4 and not x.isdigit()
    ]

    if house and address_tokens:
        keys.add(
            f"H|{country}|{house.group(0)}|{address_tokens[0]}"
        )

    return keys


# =========================================================
# BUILD INDEX
# =========================================================

def build_index():

    for p in [
        DB_PATH,
        Path(str(DB_PATH) + "-wal"),
        Path(str(DB_PATH) + "-shm")
    ]:
        if p.exists():
            p.unlink()

    print("\nCreating SQLite index...")

    conn = sqlite3.connect(DB_PATH)

    cur = conn.cursor()

    cur.execute("PRAGMA journal_mode=OFF")
    cur.execute("PRAGMA synchronous=OFF")

    cur.execute("""
        CREATE TABLE blocks (
            block_key TEXT,
            entity_id TEXT
        )
    """)

    conn.commit()

    total = 0
    last_report = 0

    for file_path in [
        TEST / "test_source2.tsv",
        TEST / "test_source3.tsv"
    ]:

        print(f"\nIndexing {file_path.name}")

        batch = []

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="replace",
            newline=""
        ) as f:

            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:

                keys = generate_block_keys(
                    row.get("business_name", ""),
                    row.get("business_address", ""),
                    row.get("country", "")
                )

                for key in keys:
                    batch.append(
                        (key, row["entity_id"])
                    )

                if len(batch) >= 50000:

                    cur.executemany(
                        "INSERT INTO blocks VALUES (?, ?)",
                        batch
                    )

                    conn.commit()

                    total += len(batch)
                    batch.clear()

                    if total - last_report >= 1_000_000:
                        print(
                            f"  indexed block records: {total:,}"
                        )
                        last_report = total

        if batch:

            cur.executemany(
                "INSERT INTO blocks VALUES (?, ?)",
                batch
            )

            conn.commit()

            total += len(batch)
            batch.clear()

    print(f"\nTotal block records: {total:,}")

    print("Creating block index...")

    cur.execute("""
        CREATE INDEX idx_blocks_key
        ON blocks(block_key)
    """)

    conn.commit()

    # -----------------------------------------------------
    # Keep only reasonably sized blocks
    # -----------------------------------------------------

    print("Calculating block sizes...")

    cur.execute("""
        CREATE TABLE valid_blocks AS
        SELECT block_key
        FROM blocks
        GROUP BY block_key
        HAVING COUNT(*) <= ?
    """, (MAX_BLOCK_SIZE,))

    conn.commit()

    cur.execute("""
        CREATE INDEX idx_valid_blocks
        ON valid_blocks(block_key)
    """)

    conn.commit()

    count = cur.execute(
        "SELECT COUNT(*) FROM valid_blocks"
    ).fetchone()[0]

    print(
        f"Valid blocking keys: {count:,}"
    )

    conn.close()


# =========================================================
# CANDIDATE GENERATION
# =========================================================

def generate_candidates():

    print("\nGenerating candidate pairs...")

    conn = sqlite3.connect(DB_PATH)

    cur = conn.cursor()

    source1 = TEST / "test_source1.tsv"

    with open(
        source1,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f, open(
        CANDIDATE_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as out:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        writer = csv.writer(
            out,
            delimiter="\t"
        )

        writer.writerow([
            "source1_entity_id",
            "candidate_entity_id"
        ])

        rows = 0
        pairs = 0

        for row in reader:

            s1_id = row["entity_id"]

            keys = list(
                generate_block_keys(
                    row.get("business_name", ""),
                    row.get("business_address", ""),
                    row.get("country", "")
                )
            )

            candidates = set()

            if keys:

                placeholders = ",".join(
                    ["?"] * len(keys)
                )

                query = f"""
                    SELECT DISTINCT b.entity_id
                    FROM blocks b
                    INNER JOIN valid_blocks v
                    ON b.block_key = v.block_key
                    WHERE b.block_key IN ({placeholders})
                """

                cur.execute(query, keys)

                for result in cur:
                    candidates.add(result[0])

            for candidate_id in candidates:

                writer.writerow([
                    s1_id,
                    candidate_id
                ])

                pairs += 1

            rows += 1

            if rows % 10000 == 0:

                print(
                    f"  S1 processed: {rows:,} | "
                    f"candidate pairs: {pairs:,}"
                )

    conn.close()

    print("\n======================================")
    print("Candidate generation completed")
    print("======================================")

    print(f"S1 entities: {rows:,}")
    print(f"Candidate pairs: {pairs:,}")
    print(f"Output: {CANDIDATE_FILE}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("======================================")
    print(" BUSINESS ENTITY BLOCKING")
    print("======================================")

    build_index()
    generate_candidates()

    print("\nDONE.")