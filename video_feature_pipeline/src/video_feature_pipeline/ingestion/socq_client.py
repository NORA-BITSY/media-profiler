from __future__ import annotations

from typing import Any

import httpx

from video_feature_pipeline.core.config import get_settings
from video_feature_pipeline.core.logging import get_logger

logger = get_logger(__name__)


class SocQClient:
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.harvest.socq_api_url
        self.api_key = self.settings.harvest.socq_api_key
        self.client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                timeout=30.0,
            )
        return self.client

    async def search_profiles(
        self,
        query: str,
        platform: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search for profiles matching query."""
        client = await self._get_client()
        params = {"q": query, "limit": limit}
        if platform:
            params["platform"] = platform

        try:
            response = await client.get("/v1/profiles/search", params=params)
            response.raise_for_status()
            return response.json().get("results", [])
        except httpx.HTTPError as e:
            logger.exception("socq_search_failed", error=str(e))
            return []

    async def get_profile_videos(
        self,
        profile_id: str,
        platform: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get video URLs for a profile."""
        client = await self._get_client()

        try:
            response = await client.get(
                f"/v1/profiles/{platform}/{profile_id}/videos",
                params={"limit": limit}
            )
            response.raise_for_status()
            return response.json().get("videos", [])
        except httpx.HTTPError as e:
            logger.exception("socq_get_videos_failed", profile_id=profile_id, error=str(e))
            return []

    async def get_profile_info(
        self,
        profile_id: str,
        platform: str,
    ) -> dict[str, Any] | None:
        """Get detailed profile information."""
        client = await self._get_client()

        try:
            response = await client.get(f"/v1/profiles/{platform}/{profile_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.exception("socq_get_profile_failed", profile_id=profile_id, error=str(e))
            return None

    async def close(self) -> None:
        if self.client:
            await self.client.aclose()
            self.client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


_socq_client: SocQClient | None = None


def get_socq_client() -> SocQClient:
    global _socq_client
    if _socq_client is None:
        _socq_client = SocQClient()
    return _socq_client