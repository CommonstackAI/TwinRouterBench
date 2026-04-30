# English public tier labels for open releases (no vendor model IDs).
TIER_LOW = "low"
TIER_MID = "mid"
TIER_MID_HIGH = "mid_high"
TIER_HIGH = "high"

PUBLIC_TIERS: tuple[str, ...] = (TIER_LOW, TIER_MID, TIER_MID_HIGH, TIER_HIGH)

# Numeric id: 0 = cheapest tier ... 3 = strongest tier (for labels / training).
TIER_TO_ID: dict[str, int] = {
    TIER_LOW: 0,
    TIER_MID: 1,
    TIER_MID_HIGH: 2,
    TIER_HIGH: 3,
}
ID_TO_TIER: dict[int, str] = {v: k for k, v in TIER_TO_ID.items()}

# Pipeline / doc Chinese labels -> public English (open dataset targets only).
CN_TIER_TO_PUBLIC: dict[str, str] = {
    "低": TIER_LOW,
    "中": TIER_MID,
    "中高": TIER_MID_HIGH,
    "高": TIER_HIGH,
}


def public_tier_from_cn(value: str) -> str:
    """Map Chinese tier label from internal annotations to public English label."""
    if value not in CN_TIER_TO_PUBLIC:
        raise ValueError(f"Unknown Chinese tier label: {value!r}; expected one of {list(CN_TIER_TO_PUBLIC)}")
    return CN_TIER_TO_PUBLIC[value]


def public_tier_to_id(tier: str) -> int:
    if tier not in TIER_TO_ID:
        raise ValueError(f"Unknown public tier: {tier!r}; expected one of {list(TIER_TO_ID)}")
    return TIER_TO_ID[tier]


def public_tier_from_id(tier_id: int) -> str:
    if tier_id not in ID_TO_TIER:
        raise ValueError(f"Unknown target_tier_id: {tier_id!r}; expected 0..3")
    return ID_TO_TIER[tier_id]
