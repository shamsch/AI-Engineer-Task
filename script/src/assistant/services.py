import asyncio
import json
from pathlib import Path
from statistics import median
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent

from assistant.config import DATA_DIR, MODEL

CACHE = Path(__file__).parents[2] / ".cache" / "services.json"

Equipment = Literal[
    "ColumnCell", "Concorde Cell", "CellStation", "Orion pump", "MD pump",
    "Larox PF", "Larox PF-DS", "Filter Optimizer", "SALA VSP pump", "other",
]
Task = Literal[
    "cloth_replacement", "hydraulic_service", "seal_replacement", "impeller_or_rotor_replacement",
    "bearing_service", "inspection", "cleaning", "calibration", "software_update", "preventive_maintenance",
]

class Part(BaseModel):
    name: str
    quantity: float
    unit_cost: float | None


class ServiceRecord(BaseModel):
    id: str
    country: str
    equipment: Equipment
    tasks: list[Task]
    onsite_hours: float | None
    travel_hours: float | None
    labour_rate: float | None
    travel_rate: float | None
    parts: list[Part]
    total_cost: float | None
    currency: Literal["USD", "EUR", "CAD", "AUD", "SEK"] | None


extractor = Agent(
    MODEL,
    output_type=ServiceRecord,
    instructions=(
        "Extract the field service report into the schema. Use only values written in the report. "
        "If a value is missing or vague (e.g. 'invoice to follow', 'as per agreement'), use null. "
        "Never calculate or guess. Country is the full English country name. "
        "travel_hours is the total travel time for the visit."
    ),
)


async def extract(path: Path, limit: asyncio.Semaphore) -> dict:
    async with limit:
        record = (await extractor.run(path.read_text(encoding="utf-8"))).output.model_dump()
    print(f"{path.stem} done", flush=True)
    return record


async def extract_all() -> list[dict]:
    limit = asyncio.Semaphore(10)
    paths = sorted((DATA_DIR / "HistoricalServices").glob("*.txt"))
    return await asyncio.gather(*(extract(p, limit) for p in paths))


def ingest() -> None:
    """Extract every report once into structured records (offline step), 10 at a time."""
    records = asyncio.run(extract_all())
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} records to {CACHE}")


def stats(values: list[float]) -> dict | None:
    return {"median": round(median(values), 1), "min": min(values), "max": max(values), "n": len(values)} if values else None


def estimate_service_cost(equipment: Equipment, tasks: list[Task] | None = None, country: str | None = None) -> dict:
    """Estimate a field service visit cost from historical service reports. The only source for costs.
    Only call this when the user has named the equipment; never pick it yourself.

    Filters past visits by equipment, then by tasks and country. If nothing matches, the country and then the
    task filters are dropped (reported in 'relaxed_filters'). Costs are grouped by currency, not converted.

    Args:
        equipment: Equipment type serviced.
        tasks: Work to be done, e.g. ['cloth_replacement'].
        country: Customer country, used for matching and travel time.
    """
    if equipment == "other":
        return {"error": "Equipment not specified. Ask the user which equipment needs service."}

    all_records = json.loads(CACHE.read_text())
    # Filter by equipment
    records = [r for r in all_records if r["equipment"] == equipment]

    relaxed: list[str] = []
    # Filter by task and country
    match = [r for r in records if (not tasks or set(tasks) & set(r["tasks"])) and (not country or r["country"] == country)]
    # Nothing matches: loosen country first, and report it
    if not match and country:
        relaxed.append("country")
        match = [r for r in records if not tasks or set(tasks) & set(r["tasks"])]
    # Still nothing: loosen task, and report it
    if not match and tasks:
        relaxed.append("tasks")
        match = records
    # Never loosen equipment: a different machine says nothing about this one's cost
    if not match:
        return {"error": f"No historical service reports for {equipment}."}

    costed = [r for r in match if r["total_cost"] and r["currency"]]
    in_country = [r["travel_hours"] for r in all_records if country and r["country"] == country and r["travel_hours"]]
    # Return the cost range per currency, with the evidence (reports used) and the gaps (relaxed filters, missing costs)
    return {
        "matched_reports": [r["id"] for r in match],
        "relaxed_filters": relaxed,
        "total_cost_by_currency": {c: stats([r["total_cost"] for r in costed if r["currency"] == c]) for c in {r["currency"] for r in costed}},
        "reports_without_cost": [r["id"] for r in match if r not in costed],
        "onsite_hours": stats([r["onsite_hours"] for r in match if r["onsite_hours"]]),
        "travel_hours": stats([r["travel_hours"] for r in match if r["travel_hours"]]),
        "travel_hours_to_country_any_equipment": stats(in_country),
        "records": match,
    }
