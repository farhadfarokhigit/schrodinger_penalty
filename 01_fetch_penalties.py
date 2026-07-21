"""
Stage 1 of 2: download every penalty kick from StatsBomb's open data.

What this does:
  1. Fetches the list of all competitions/seasons StatsBomb has released.
  2. For each one, fetches its match list and collects every match_id.
  3. For each match, fetches the full event log and keeps only the shot
     events tagged as penalties (both in-game spot-kicks and shootouts).
  4. Saves the raw penalty records (still in StatsBomb's native coordinate
     units) to penalties_raw.json.

Requirements: internet access to github.com / raw.githubusercontent.com.
No API key needed -- this is public data.

Expect roughly 3,900 matches to process in total across all competitions,
which took several minutes when we ran it. The script is deliberately
conservative (small thread pool, retries with backoff) to avoid hammering
GitHub's raw-content servers; do not increase max_workers much beyond 12
or you risk rate-limiting.

Run this once. Its output feeds into build_data_xz.py (Stage 2), which
does not need the network at all.
"""

import json
import time
import urllib.request
import concurrent.futures
import threading

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def fetch_json(url, retries=6, timeout=25):
    """GET a JSON file with exponential backoff on failure."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            time.sleep(min(2 ** attempt, 20))
    return None


def get_all_competitions():
    comps = fetch_json(f"{BASE}/competitions.json")
    if comps is None:
        raise RuntimeError("Could not reach StatsBomb open-data on GitHub -- "
                            "check your internet connection / that github.com "
                            "and raw.githubusercontent.com are reachable.")
    print(f"found {len(comps)} competition-season entries")
    return comps


def get_all_match_ids(competitions):
    match_ids = []
    for c in competitions:
        url = f"{BASE}/matches/{c['competition_id']}/{c['season_id']}.json"
        data = fetch_json(url)
        if not data:
            print(f"  [skipped, unreachable] {c['competition_name']} {c['season_name']}")
            continue
        for m in data:
            match_ids.append(m['match_id'])
    print(f"total matches across all competitions: {len(match_ids)}")
    return match_ids


def extract_penalties_from_match(match_id, results, errors, lock):
    """Fetch one match's events and keep only penalty shot events."""
    url = f"{BASE}/events/{match_id}.json"
    data = fetch_json(url)
    if data is None:
        with lock:
            errors.append(match_id)
        return

    penalties = []
    for ev in data:
        if ev.get('type', {}).get('name') == 'Shot':
            shot = ev.get('shot', {})
            if shot.get('type', {}).get('name') == 'Penalty':
                penalties.append({
                    'match_id': match_id,
                    'player': ev.get('player', {}).get('name'),
                    'team': ev.get('team', {}).get('name'),
                    'location': ev.get('location'),          # [x, y] at the spot
                    'end_location': shot.get('end_location'),  # [x, y, z] at the goal plane
                    'outcome': shot.get('outcome', {}).get('name'),
                    'period': ev.get('period'),  # 1-4 = in-game/extra-time, 5 = shootout
                })
    with lock:
        results.extend(penalties)


def get_all_penalties(match_ids, max_workers=12):
    lock = threading.Lock()
    results, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(extract_penalties_from_match, mid, results, errors, lock)
                   for mid in match_ids]
        done = 0
        for _ in concurrent.futures.as_completed(futures):
            done += 1
            if done % 200 == 0:
                print(f"  processed {done}/{len(match_ids)} matches, "
                      f"{len(results)} penalties found so far")
    print(f"finished: {len(results)} penalty kicks found, {len(errors)} matches "
          f"could not be fetched after retries")
    return results, errors


if __name__ == '__main__':
    comps = get_all_competitions()
    match_ids = get_all_match_ids(comps)
    penalties, failed_matches = get_all_penalties(match_ids)

    with open('penalties_raw.json', 'w') as f:
        json.dump(penalties, f)
    print(f"\nsaved {len(penalties)} raw penalty records to penalties_raw.json")
    if failed_matches:
        print(f"note: {len(failed_matches)} matches failed after retries and were "
              f"skipped (transient network issues); re-run if you want to retry them")
