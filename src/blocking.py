import argparse
import csv
import hashlib
import re
import shutil
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "blocking_output"
PARTITION_DIR = OUTPUT_DIR / "target_partitions"
CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
CANDIDATE_DB = OUTPUT_DIR / "candidate_accumulator.sqlite"

PARTITION_COUNT = 128
MAX_BLOCK_SIZE = 150
MAX_CANDIDATES_PER_SOURCE1 = 5
MAX_PARTITION_CACHE_KIB = 24_000
KEY_WEIGHTS = {"N": 12.0, "P": 3.0, "Z": 2.5, "A": 2.5}

LEGAL_AND_GENERIC_WORDS = {
    "ag", "and", "bv", "co", "company", "corp", "corporation", "gmbh",
    "inc", "incorporated", "limited", "llc", "llp", "lp", "ltd", "plc",
    "private", "pte", "pvt", "sarl", "sas", "sa", "the",
}
ADDRESS_GENERIC_WORDS = LEGAL_AND_GENERIC_WORDS | {
    "avenue", "ave", "boulevard", "blvd", "drive", "dr", "highway", "hwy",
    "lane", "ln", "road", "rd", "street", "st", "route", "rue", "chemin",
}
POSTAL_RE = re.compile(r"(?<!\w)(\d{4,10})(?!\w)", flags=re.UNICODE)
HOUSE_NUMBER_RE = re.compile(r"(?<!\w)(\d+[a-z]?)(?!\w)", flags=re.UNICODE)


def normalize_text(value):
    value = unicodedata.normalize("NFKC", value or "").casefold()
    normalized = []
    for character in value:
        category = unicodedata.category(character)
        if character.isspace() or category[0] in {"L", "N", "M"}:
            normalized.append(character)
        else:
            normalized.append(" ")
    return " ".join("".join(normalized).split())


def _distinctive_tokens(value, ignored_words, limit):
    tokens = {
        token for token in value.split()
        if len(token) >= 3 and not token.isdigit() and token not in ignored_words
    }
    return sorted(tokens, key=lambda token: (-len(token), token))[:limit]


def generate_block_keys(name, address, country):
    name = normalize_text(name)
    address = normalize_text(address)
    country = normalize_text(country) or "_unknown_country"
    keys = set()

    if name and len(name) <= 256:
        keys.add(f"N|{country}|{name}")

    name_tokens = _distinctive_tokens(name, LEGAL_AND_GENERIC_WORDS, 1)
    if name_tokens:
        keys.add(f"P|{country}|{name_tokens[0][:6]}")

    postal_codes = POSTAL_RE.findall(address)
    if postal_codes:
        keys.add(f"Z|{country}|{postal_codes[0]}")

    house_match = HOUSE_NUMBER_RE.search(address)
    address_tokens = _distinctive_tokens(address, ADDRESS_GENERIC_WORDS, 1)
    if house_match and address_tokens:
        keys.add(f"A|{country}|{house_match.group(1)}|{address_tokens[0]}")

    return keys


def _partition_for(key):
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % PARTITION_COUNT


def _remove_stale_artifacts():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (
        CANDIDATE_DB,
        Path(str(CANDIDATE_DB) + "-wal"),
        Path(str(CANDIDATE_DB) + "-shm"),
        OUTPUT_DIR / "blocking_index.sqlite",
        OUTPUT_DIR / "blocking_index.sqlite-wal",
        OUTPUT_DIR / "blocking_index.sqlite-shm",
    ):
        if path.exists():
            path.unlink()
    if PARTITION_DIR.exists():
        shutil.rmtree(PARTITION_DIR)
    if CANDIDATE_PATH.exists():
        CANDIDATE_PATH.unlink()


def _source_row(row):
    return (
        (row.get("entity_id") or "").strip(),
        row.get("business_name", ""),
        row.get("business_address", ""),
        row.get("country", ""),
    )


def partition_targets(test_dir, sample_records=None, sample_source1=None):
    PARTITION_DIR.mkdir(parents=True, exist_ok=True)
    target_handles = [
        (PARTITION_DIR / f"targets_{index:03d}.tsv").open("w", encoding="utf-8", newline="")
        for index in range(PARTITION_COUNT)
    ]
    source1_handles = [
        (PARTITION_DIR / f"source1_{index:03d}.tsv").open("w", encoding="utf-8", newline="")
        for index in range(PARTITION_COUNT)
    ]
    record_counts = Counter()
    try:
        source1_path = test_dir / "test_source1.tsv"
        print(f"Partitioning {source1_path.name} lookup keys...", flush=True)
        with source1_path.open("r", encoding="utf-8", newline="") as source_file:
            reader = csv.DictReader(source_file, delimiter="\t")
            for row in reader:
                entity_id, name, address, country = _source_row(row)
                if not entity_id.startswith("S1-"):
                    continue
                if sample_source1 is not None and record_counts[1] >= sample_source1:
                    break
                for key in generate_block_keys(name, address, country):
                    source1_handles[_partition_for(key)].write(f"{key}\t{entity_id}\n")
                record_counts[1] += 1

        for source_number in (2, 3):
            source_path = test_dir / f"test_source{source_number}.tsv"
            expected_prefix = f"S{source_number}-"
            print(f"Partitioning {source_path.name}...", flush=True)
            with source_path.open("r", encoding="utf-8", newline="") as source_file:
                reader = csv.DictReader(source_file, delimiter="\t")
                for row in reader:
                    entity_id, name, address, country = _source_row(row)
                    if not entity_id.startswith(expected_prefix):
                        continue
                    if sample_records is not None and record_counts[source_number] >= sample_records:
                        break
                    for key in generate_block_keys(name, address, country):
                        target_handles[_partition_for(key)].write(f"{key}\t{entity_id}\n")
                    record_counts[source_number] += 1
                    if record_counts[source_number] % 500_000 == 0:
                        print(
                            f"  {source_path.name}: {record_counts[source_number]:,} records",
                            flush=True,
                        )
    finally:
        for handle in target_handles + source1_handles:
            handle.close()
    return record_counts


def _load_selective_partition(path):
    if not path.exists():
        return {}
    frequencies = Counter()
    with path.open("r", encoding="utf-8", newline="") as partition_file:
        for line in partition_file:
            key, _ = line.rstrip("\n").split("\t", 1)
            frequencies[key] += 1

    selective = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as partition_file:
        for line in partition_file:
            key, entity_id = line.rstrip("\n").split("\t", 1)
            if frequencies[key] <= MAX_BLOCK_SIZE:
                selective[key].append(entity_id)
    return selective


def _initialize_candidate_db():
    connection = sqlite3.connect(CANDIDATE_DB)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(f"PRAGMA cache_size=-{MAX_PARTITION_CACHE_KIB}")
    connection.execute(
        "CREATE TABLE pairs ("
        "source1_entity_id TEXT NOT NULL, candidate_entity_id TEXT NOT NULL, "
        "score REAL NOT NULL, "
        "PRIMARY KEY (source1_entity_id, candidate_entity_id)) WITHOUT ROWID"
    )
    connection.commit()
    return connection


def _process_source1_partition(connection, index, counts):
    target_path = PARTITION_DIR / f"targets_{index:03d}.tsv"
    source1_path = PARTITION_DIR / f"source1_{index:03d}.tsv"
    new_pairs = 0
    selective = _load_selective_partition(target_path)
    if not selective:
        return 0

    def flush_source1(source1_id, keys):
        nonlocal new_pairs
        scores = defaultdict(float)
        for key in keys:
            for candidate_id in selective.get(key, ()):
                scores[candidate_id] += KEY_WEIGHTS[key[0]]

        current_count = counts.get(source1_id, 0)
        cursor = connection.cursor()
        for candidate_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
            cursor.execute(
                "INSERT OR IGNORE INTO pairs VALUES (?, ?, ?)",
                (source1_id, candidate_id, score),
            )
            if cursor.rowcount:
                if current_count < MAX_CANDIDATES_PER_SOURCE1:
                    current_count += 1
                    new_pairs += 1
                    counts[source1_id] = current_count
                else:
                    cursor.execute(
                        "DELETE FROM pairs WHERE source1_entity_id = ? AND candidate_entity_id = ?",
                        (source1_id, candidate_id),
                    )
            else:
                cursor.execute(
                    "UPDATE pairs SET score = score + ? "
                    "WHERE source1_entity_id = ? AND candidate_entity_id = ?",
                    (score, source1_id, candidate_id),
                )

    with source1_path.open("r", encoding="utf-8", newline="") as query_file:
        current_id = None
        current_keys = []
        for line in query_file:
            key, source1_id = line.rstrip("\n").split("\t", 1)
            if source1_id != current_id:
                if current_id is not None:
                    flush_source1(current_id, current_keys)
                current_id = source1_id
                current_keys = []
            if key in selective:
                current_keys.append(key)
        if current_id is not None:
            flush_source1(current_id, current_keys)
    connection.commit()
    return new_pairs


def generate_test_candidates(sample_records=None, sample_source1=None):
    test_dir = DATA_DIR / "test"
    _remove_stale_artifacts()
    free_bytes = shutil.disk_usage(BASE_DIR).free
    if free_bytes < 2 * 1024**3:
        raise RuntimeError(
            f"Refusing to generate partitions with only {free_bytes / 1024**3:.2f} GB free."
        )

    source_counts = partition_targets(test_dir, sample_records, sample_source1)
    connection = _initialize_candidate_db()
    candidates_per_source1 = {}
    candidate_count = 0
    source1_count = source_counts[1]
    try:
        for partition_index in range(PARTITION_COUNT):
            candidate_count += _process_source1_partition(
                connection, partition_index, candidates_per_source1
            )
            if partition_index % 16 == 15:
                print(
                    f"  processed partitions: {partition_index + 1}/{PARTITION_COUNT}; "
                    f"unique candidate pairs: {candidate_count:,}",
                    flush=True,
                )

        connection.commit()
        with CANDIDATE_PATH.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", "candidate_entity_id"])
            for source1_id, candidate_id in connection.execute(
                "SELECT source1_entity_id, candidate_entity_id FROM pairs "
                "ORDER BY source1_entity_id, candidate_entity_id"
            ):
                writer.writerow([source1_id, candidate_id])
    finally:
        connection.close()
        if PARTITION_DIR.exists():
            shutil.rmtree(PARTITION_DIR)
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(CANDIDATE_DB) + suffix)
            if path.exists():
                path.unlink()

    average = candidate_count / source1_count if source1_count else 0.0
    maximum = max(candidates_per_source1.values(), default=0)
    print("Test candidate generation complete")
    print(f"  Source 2 records indexed: {source_counts[2]:,}")
    print(f"  Source 3 records indexed: {source_counts[3]:,}")
    print(f"  Source 1 records processed: {source1_count:,}")
    print(f"  Candidate pairs: {candidate_count:,}")
    print(f"  Average candidates per Source 1: {average:.2f}")
    print(f"  Maximum candidates for one Source 1: {maximum:,}")
    print(f"  Output: {CANDIDATE_PATH}")
    print(f"  Output size: {CANDIDATE_PATH.stat().st_size:,} bytes")


def main():
    parser = argparse.ArgumentParser(
        description="Generate bounded-memory candidate pairs using disk partitions."
    )
    parser.add_argument(
        "--sample-records", type=int,
        help="Index only this many records from each target source for a smoke test.",
    )
    parser.add_argument(
        "--sample-source1", type=int,
        help="Process only this many Source-1 rows for a smoke test.",
    )
    args = parser.parse_args()
    generate_test_candidates(args.sample_records, args.sample_source1)


if __name__ == "__main__":
    main()