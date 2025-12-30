from __future__ import annotations

from io import BytesIO
from typing import List, Tuple, Literal, Optional

import numpy as np
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, HttpUrl, Field
from PIL import Image, ImageOps
from sklearn.cluster import MiniBatchKMeans

app = FastAPI(title="Palette API (clustered + Roblox)", version="3.0.0")

# --- Safety / limits ---
MAX_BYTES = 8 * 1024 * 1024          # 8 MB
TIMEOUT_S = 12.0
MAX_SIDE = 512                       # resize for speed
TOP_K = 5

# --- Clustering tunables ---
PIXEL_SAMPLE_MAX = 120_000
KMEANS_BATCH = 4096
KMEANS_RANDOM_STATE = 0

# Roblox endpoints (documented domains)
ROBLOX_THUMBNAILS = "https://thumbnails.roblox.com"
ROBLOX_USERS = "https://users.roblox.com"


# ------------------ Models ------------------

class PaletteRequest(BaseModel):
    url: HttpUrl = Field(..., description="Direct URL to an image file")


class RobloxPaletteRequest(BaseModel):
    userId: int = Field(..., ge=1)
    kind: Literal["headshot", "bust", "full"] = "bust"
    size: Literal["48x48", "50x50", "60x60", "75x75", "100x100", "110x110",
                  "150x150", "180x180", "352x352", "420x420", "720x720"] = "420x420"
    format: Literal["Png", "Jpeg", "Webp"] = "Png"
    isCircular: bool = False


class RobloxUsernamePaletteRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=20)
    kind: Literal["headshot", "bust", "full"] = "bust"
    size: Literal["48x48", "50x50", "60x60", "75x75", "100x100", "110x110",
                  "150x150", "180x180", "352x352", "420x420", "720x720"] = "420x420"
    format: Literal["Png", "Jpeg", "Webp"] = "Png"
    isCircular: bool = False


class PaletteColor(BaseModel):
    hex: str
    rgb: Tuple[int, int, int]
    percent: float


class PaletteResponse(BaseModel):
    source: str
    colors: List[PaletteColor]


# ------------------ Helpers ------------------

def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02X}{g:02X}{b:02X}"


async def _http_get_bytes(url: str, max_bytes: int = MAX_BYTES) -> bytes:
    async with httpx.AsyncClient(follow_redirects=True, timeout=TIMEOUT_S) as client:
        try:
            r = await client.get(url)
        except httpx.RequestError as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch: {e}") from e

    if r.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"URL returned HTTP {r.status_code}")

    data = r.content
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Content too large (>{max_bytes} bytes)")
    return data


def _open_and_prepare(data: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(data))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=415, detail=f"Unsupported/invalid image: {e}") from e

    w, h = img.size
    scale = min(1.0, MAX_SIDE / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def _clustered_palette_percent(img: Image.Image, k: int = TOP_K) -> list[tuple[tuple[int, int, int], float]]:
    """
    Cluster pixels in LAB (perceptual-ish) so shades/tints get grouped.
    Returns representative RGB colors + % coverage.
    """
    arr = np.asarray(img, dtype=np.uint8)          # (H,W,3)
    pixels_rgb = arr.reshape(-1, 3)                # (N,3)

    n = pixels_rgb.shape[0]
    sample_n = min(PIXEL_SAMPLE_MAX, n)
    if n > sample_n:
        rng = np.random.default_rng(KMEANS_RANDOM_STATE)
        idx = rng.choice(n, sample_n, replace=False)
        sample_rgb = pixels_rgb[idx]
    else:
        sample_rgb = pixels_rgb

    # RGB -> LAB using PIL
    sample_lab = np.asarray(
        Image.fromarray(sample_rgb.reshape(-1, 1, 3), "RGB").convert("LAB"),
        dtype=np.float32
    ).reshape(-1, 3)

    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=KMEANS_RANDOM_STATE,
        batch_size=KMEANS_BATCH,
        n_init="auto",
    )
    km.fit(sample_lab)

    lab_all = np.asarray(Image.fromarray(arr, "RGB").convert("LAB"), dtype=np.float32).reshape(-1, 3)
    labels = km.predict(lab_all)

    counts = np.bincount(labels, minlength=k)
    total = int(counts.sum())
    if total == 0:
        return []

    # Center LAB -> representative RGB
    centers_lab = km.cluster_centers_.clip(0, 255).astype(np.uint8)  # (k, 3)

    centers_rgb_list = []
    for i in range(k):
        lab_pixel = centers_lab[i].reshape((1, 1, 3))  # (1,1,3)
        rgb_pixel = Image.fromarray(lab_pixel, mode="LAB").convert("RGB")
        rgb_arr = np.asarray(rgb_pixel, dtype=np.uint8).reshape(3,)
        centers_rgb_list.append(rgb_arr)

    centers_rgb = np.stack(centers_rgb_list, axis=0)  # (k, 3)

    order = np.argsort(counts)[::-1]
    out: list[tuple[tuple[int, int, int], float]] = []
    for i in order:
        rgb = tuple(int(x) for x in centers_rgb[i])
        pct = (float(counts[i]) * 100.0) / float(total)
        out.append((rgb, pct))

    return out[:k]


def _build_response(source: str, img: Image.Image) -> PaletteResponse:
    top = _clustered_palette_percent(img, TOP_K)
    colors = [
        PaletteColor(hex=_rgb_to_hex(rgb), rgb=rgb, percent=round(pct, 3))
        for rgb, pct in top
    ]
    return PaletteResponse(source=source, colors=colors)


async def _roblox_resolve_userid(username: str) -> int:
    """
    Resolve username -> userId using users.roblox.com.
    Uses POST /v1/usernames/users (documented).
    """
    payload = {
        "usernames": [username],
        "excludeBannedUsers": False,
    }
    async with httpx.AsyncClient(timeout=TIMEOUT_S, follow_redirects=True) as client:
        try:
            r = await client.post(f"{ROBLOX_USERS}/v1/usernames/users", json=payload)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Roblox users API request failed: {e}") from e

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Roblox users API HTTP {r.status_code}")

    j = r.json()
    data = j.get("data") or []
    if not data:
        raise HTTPException(status_code=404, detail="Username not found on Roblox")

    uid = data[0].get("id")
    if not isinstance(uid, int) or uid <= 0:
        raise HTTPException(status_code=502, detail="Roblox users API returned invalid userId")
    return uid


async def _roblox_thumbnail_image_url(
    user_id: int,
    kind: str,
    size: str,
    fmt: str,
    is_circular: bool,
) -> str:
    """
    Calls thumbnails.roblox.com and returns the resolved imageUrl.
    """
    if kind == "headshot":
        path = "/v1/users/avatar-headshot"
    elif kind == "bust":
        path = "/v1/users/avatar-bust"
    else:
        path = "/v1/users/avatar"  # full body thumbnail

    params = {
        "userIds": str(user_id),
        "size": size,
        "format": fmt,
        "isCircular": "true" if is_circular else "false",
    }

    async with httpx.AsyncClient(timeout=TIMEOUT_S, follow_redirects=True) as client:
        try:
            r = await client.get(f"{ROBLOX_THUMBNAILS}{path}", params=params)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Roblox thumbnails API request failed: {e}") from e

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Roblox thumbnails API HTTP {r.status_code}")

    j = r.json()
    data = j.get("data") or []
    if not data:
        raise HTTPException(status_code=502, detail="Roblox thumbnails API returned no data")

    item = data[0]
    state = item.get("state")
    image_url = item.get("imageUrl") or ""

    # Roblox sometimes returns Pending with empty URL (known behavior in practice).
    if state != "Completed" or not image_url:
        raise HTTPException(status_code=503, detail=f"Roblox thumbnail not ready (state={state})")

    return image_url


# ------------------ Endpoints ------------------

@app.post("/palette", response_model=PaletteResponse)
async def palette_from_url(req: PaletteRequest) -> PaletteResponse:
    data = await _http_get_bytes(str(req.url))
    img = _open_and_prepare(data)
    return _build_response(str(req.url), img)


@app.post("/palette/file", response_model=PaletteResponse)
async def palette_from_upload(file: UploadFile = File(...)) -> PaletteResponse:
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"Image too large (>{MAX_BYTES} bytes)")
    img = _open_and_prepare(data)
    return _build_response(file.filename or "upload", img)


@app.post("/palette/roblox", response_model=PaletteResponse)
async def palette_from_roblox(req: RobloxPaletteRequest) -> PaletteResponse:
    thumb_url = await _roblox_thumbnail_image_url(
        user_id=req.userId,
        kind=req.kind,
        size=req.size,
        fmt=req.format,
        is_circular=req.isCircular,
    )
    data = await _http_get_bytes(thumb_url)
    img = _open_and_prepare(data)
    return _build_response(f"roblox:userId={req.userId}:{req.kind}", img)


@app.post("/palette/roblox/username", response_model=PaletteResponse)
async def palette_from_roblox_username(req: RobloxUsernamePaletteRequest) -> PaletteResponse:
    user_id = await _roblox_resolve_userid(req.username)
    thumb_url = await _roblox_thumbnail_image_url(
        user_id=user_id,
        kind=req.kind,
        size=req.size,
        fmt=req.format,
        is_circular=req.isCircular,
    )
    data = await _http_get_bytes(thumb_url)
    img = _open_and_prepare(data)
    return _build_response(f"roblox:username={req.username}:userId={user_id}:{req.kind}", img)
