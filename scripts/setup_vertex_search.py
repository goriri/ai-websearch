"""Provisions the Vertex AI Search (Agent Search) image index for Prism.

Basic website search allows <= 50 URL patterns per data store and one data store
per app, so the 100 sites in app/vertex_search_sites.txt are split across two
PUBLIC_WEBSITE data stores, each with its own Enterprise-tier search engine.

Usage (needs Discovery Engine admin on the project):
  GOOGLE_CLOUD_PROJECT=<project> python -m scripts.setup_vertex_search
Prints the engine IDs to put in VERTEX_SEARCH_ENGINES (comma-separated).
Idempotent: existing data stores / engines / sites are reused.
"""

import os
import pathlib
import subprocess
import sys
import time

import httpx

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or sys.exit("Set GOOGLE_CLOUD_PROJECT")
PREFIX = os.environ.get("VERTEX_SEARCH_PREFIX", "prism-images")
BASE = (f"https://discoveryengine.googleapis.com/v1/projects/{PROJECT}"
        "/locations/global/collections/default_collection")
SITES_FILE = pathlib.Path(__file__).resolve().parents[1] / "app" / "vertex_search_sites.txt"


def token() -> str:
  return subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()


def api(method: str, url: str, **kw) -> httpx.Response:
  headers = {"Authorization": f"Bearer {token()}", "X-Goog-User-Project": PROJECT,
             "Content-Type": "application/json"}
  return httpx.request(method, url, headers=headers, timeout=60, **kw)


def wait_op(op: dict) -> None:
  name = op.get("name")
  while name and not op.get("done"):
    time.sleep(3)
    op = api("GET", f"https://discoveryengine.googleapis.com/v1/{name}").json()
  if op.get("error"):
    raise RuntimeError(op["error"])


def sites() -> list[str]:
  return [l.strip() for l in SITES_FILE.read_text().splitlines()
          if l.strip() and not l.startswith("#")]


def ensure_datastore(ds_id: str) -> None:
  r = api("POST", f"{BASE}/dataStores?dataStoreId={ds_id}", json={
      "displayName": ds_id, "industryVertical": "GENERIC",
      "solutionTypes": ["SOLUTION_TYPE_SEARCH"], "contentConfig": "PUBLIC_WEBSITE"})
  if r.status_code == 409:
    print(f"  data store {ds_id} exists")
    return
  r.raise_for_status()
  wait_op(r.json())
  print(f"  created data store {ds_id}")


def ensure_sites(ds_id: str, domains: list[str]) -> None:
  url = f"{BASE}/dataStores/{ds_id}/siteSearchEngine/targetSites"
  existing = {t.get("providedUriPattern") for t in api("GET", url + "?pageSize=100").json().get("targetSites", [])}
  todo = [d for d in domains if f"{d}/*" not in existing and d not in existing]
  for i in range(0, len(todo), 20):  # batchCreate accepts <= 20 sites.
    batch = todo[i:i + 20]
    r = api("POST", url + ":batchCreate", json={"requests": [
        {"targetSite": {"providedUriPattern": f"{d}/*", "type": "INCLUDE", "exactMatch": False}}
        for d in batch]})
    r.raise_for_status()
    wait_op(r.json())
    print(f"  added {len(batch)} sites to {ds_id}")
  if not todo:
    print(f"  {ds_id}: all {len(domains)} sites present")


def ensure_engine(engine_id: str, ds_id: str) -> None:
  r = api("POST", f"{BASE}/engines?engineId={engine_id}", json={
      "displayName": engine_id, "dataStoreIds": [ds_id],
      "solutionType": "SOLUTION_TYPE_SEARCH", "industryVertical": "GENERIC",
      "searchEngineConfig": {"searchTier": "SEARCH_TIER_ENTERPRISE", "searchAddOns": []}})
  if r.status_code == 409:
    print(f"  engine {engine_id} exists")
    return
  r.raise_for_status()
  wait_op(r.json())
  print(f"  created engine {engine_id}")


def main() -> None:
  domains = sites()
  assert len(domains) <= 100, "two basic data stores hold at most 100 sites"
  engines = []
  for n, chunk in enumerate((domains[:50], domains[50:]), 1):
    if not chunk:
      continue
    ds_id, engine_id = f"{PREFIX}-ds-{n}", f"{PREFIX}-{n}"
    print(f"[{n}] {len(chunk)} sites")
    ensure_datastore(ds_id)
    ensure_sites(ds_id, chunk)
    ensure_engine(engine_id, ds_id)
    engines.append(engine_id)
  print("\nVERTEX_SEARCH_ENGINES=" + ",".join(engines))
  print("Basic website search uses Google's public index; allow up to ~1 hour before results appear.")


if __name__ == "__main__":
  main()
