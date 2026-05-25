"""
Load and cache intersphinx objects.inv files for PennyLane and Catalyst.

Tries to load from disk first; if missing or unusable, fetches from the web
and persists to disk for next time.
"""
import os
import time
from pathlib import Path
from typing import Any

import sphobjinv as soi
import requests
from requests import exceptions as requests_exceptions

REQUEST_TIMEOUT = 5  # seconds
USER_AGENT = "Mozilla/5.0 (compatible; Demos-Pandoc-Filter; +https://github.com/PennyLaneAI/demos)"


def _load_inventory_from_path(path: Path) -> soi.Inventory | None:
    """Load an inventory from a local file. Returns None if file missing or invalid."""
    if not path.is_file():
        return None
    try:
        return soi.Inventory(fname_zlib=path)
    except Exception:
        return None


def _load_inventory_from_url(
    url: str, retries: int = 3, delay: float = 1.0
) -> tuple[soi.Inventory, bytes]:
    """Download objects.inv from URL and load with retry logic. Returns (inventory, raw_bytes)."""
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            break
        except requests_exceptions.RequestException:
            if attempt == retries:
                raise
            time.sleep(delay * attempt)
    # sphobjinv accepts zlib-compressed bytes from objects.inv
    raw = response.content
    return soi.Inventory(zlib=raw), raw


def _make_named_inventory(inv: soi.Inventory) -> dict[str, Any]:
    """Build a name -> item dict for easy lookup of inventory items."""
    return {item.name: item for item in inv.objects}


class IntersphinxInventories:
    """
    Load and cache PennyLane and Catalyst objects.inv inventories.

    On initialization, each inventory is loaded from disk if present in
    cache_dir; otherwise it is fetched from the web and then saved to disk.
    The parsed contents are exposed as .pennylane and .catalyst (name -> item
    dicts). Base URLs for building doc links are in .pennylane_base_url
    and .catalyst_base_url.
    """

    @property
    def dev(self) -> bool:
        """True if the DEV environment variable is the string \"True\"; False otherwise."""
        return os.environ.get("DEV", "") == "True"

    @property
    def pennylane_base_url(self) -> str:
        """Base URL for PennyLane documentation links."""
        return "https://docs.pennylane.ai/en/" + ("latest/" if self.dev else "stable/")

    @property
    def catalyst_base_url(self) -> str:
        """Base URL for Catalyst documentation links."""
        return "https://docs.pennylane.ai/projects/catalyst/en/" + ("latest/" if self.dev else "stable/")

    def __init__(
        self,
        cache_dir: Path | str | None = None,
    ):
        if cache_dir is None:
            cache_dir = Path.cwd() / ".intersphinx_cache"
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._pennylane: dict[str, Any] = {}
        self._catalyst: dict[str, Any] = {}

        self._load_pennylane()
        self._load_catalyst()

    def _load_pennylane(self) -> None:
        cache_path = self._cache_dir / f"pennylane_objects{'_dev' if self.dev else ''}.inv"
        inv = _load_inventory_from_path(cache_path)
        if inv is None:
            URL = self.pennylane_base_url + "objects.inv"
            inv, raw = _load_inventory_from_url(URL)
            cache_path.write_bytes(raw)
        self._pennylane = _make_named_inventory(inv)

    def _load_catalyst(self) -> None:
        cache_path = self._cache_dir / f"catalyst_objects{'_dev' if self.dev else ''}.inv"
        inv = _load_inventory_from_path(cache_path)
        if inv is None:
            URL = self.catalyst_base_url + "objects.inv"
            inv, raw = _load_inventory_from_url(URL)
            cache_path.write_bytes(raw)
        self._catalyst = _make_named_inventory(inv)

    @property
    def pennylane(self) -> dict[str, Any]:
        """PennyLane inventory as name -> item dict for link resolution."""
        return self._pennylane

    @property
    def catalyst(self) -> dict[str, Any]:
        """Catalyst inventory as name -> item dict for link resolution."""
        return self._catalyst
