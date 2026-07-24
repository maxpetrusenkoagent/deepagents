"""Tests for ModelProvider and CommandPalette integration."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from textual.app import App
from textual.command import DiscoveryHit, Hit

from deepagents_code.app import DeepAgentsApp
from deepagents_code.tui.command_palette import ModelProvider


class CommandPaletteTestApp(App):
    """Test app for CommandPalette registration and functionality."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.switched_model: str | None = None
        self.current_spec: str | None = "openai:gpt-4"

    def _effective_model_spec(self) -> str | None:
        """Mock _effective_model_spec."""
        return self.current_spec

    async def _switch_model(
        self,
        model_spec: str,
        *,
        extra_kwargs: dict[str, Any] | None = None,
        announce_unchanged: bool = True,
        persist: bool = True,
        from_resume: bool = False,
    ) -> None:
        """Mock _switch_model for testing."""
        del extra_kwargs, announce_unchanged, persist, from_resume
        self.switched_model = model_spec


@pytest.mark.asyncio
async def test_model_provider_logic() -> None:
    """Test that ModelProvider correctly discovers and searches models."""
    mock_available = {
        "anthropic": ["claude-3-sonnet", "claude-3-opus"],
        "openai": ["gpt-4"],
    }
    mock_profiles = {
        "anthropic:claude-3-sonnet": {"profile": {"name": "Claude 3 Sonnet"}},
        "anthropic:claude-3-opus": {"profile": {"name": "Claude 3 Opus"}},
        "openai:gpt-4": {"profile": {"name": "GPT-4"}},
    }

    # Patch model_config functions that ModelProvider calls.
    with (
        patch(
            "deepagents_code.tui.command_palette.get_available_models",
            return_value=mock_available,
        ),
        patch(
            "deepagents_code.tui.command_palette.get_model_profiles",
            return_value=mock_profiles,
        ),
    ):
        app = CommandPaletteTestApp()
        from textual.app import active_app

        active_app.set(app)
        # Provider requires a screen; we mock one that points to our test app.
        mock_screen = MagicMock()
        mock_screen.app = app
        provider = ModelProvider(mock_screen)

        # Test discover()
        discovery_hits = [hit async for hit in provider.discover()]
        assert len(discovery_hits) == 3
        assert all(isinstance(hit, DiscoveryHit) for hit in discovery_hits)

        # DiscoveryHit uses 'display' for the label
        display_names = {str(hit.display) for hit in discovery_hits}
        assert "Claude 3 Sonnet" in display_names
        assert "Claude 3 Opus" in display_names
        # Check that current model is indicated
        assert "GPT-4 (current)" in display_names

        # Test search() with a query
        search_hits = [hit async for hit in provider.search("claude")]
        assert len(search_hits) == 2
        assert all(isinstance(hit, Hit) for hit in search_hits)

        # Verify help text includes provider
        assert any("anthropic" in (hit.help or "").lower() for hit in search_hits)


def test_command_palette_registration() -> None:
    """Test that the command palette is enabled and provider is registered."""
    assert DeepAgentsApp.ENABLE_COMMAND_PALETTE is True
    # The provider is registered via a lazy-load function.
    from deepagents_code.app import _get_model_provider

    assert _get_model_provider in DeepAgentsApp.COMMANDS


@pytest.mark.asyncio
async def test_command_palette_interaction() -> None:
    """Test that the command palette opens and doesn't crash with our provider."""
    # We use DeepAgentsApp directly to test the real COMMANDS registration.
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        # Open the command palette.
        await pilot.press("ctrl+p")
        # Ensure it's open (it should have a CommandPalette screen).
        from textual.command import CommandPalette

        assert any(isinstance(screen, CommandPalette) for screen in app.screen_stack)
        # Close it.
        await pilot.press("escape")
