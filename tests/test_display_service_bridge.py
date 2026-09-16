"""Tests for PyDisplayServiceBridge — verifies that setting coordinator.display_system
creates a Rust-side DisplayService bridge and is reflected in to_dict()."""

import pytest
from unittest.mock import AsyncMock, patch

from amplifier_core import AmplifierSession
from amplifier_core._engine import RustCoordinator, RustSession
from amplifier_core.models import HookResult


class RecordingDisplay:
    def __init__(self):
        self.messages = []

    def show_message(self, message, level="info", source=None):
        self.messages.append((message, level, source))


def test_display_system_sets_has_display_service():
    try:
        from amplifier_core._engine import RustCoordinator
    except ImportError:
        pytest.skip("Rust engine not available")
    coord = RustCoordinator()
    d = coord.to_dict()
    assert d.get("has_display_service") is False or d.get("has_display_service") is None

    class FakeDisplay:
        def __init__(self):
            self.messages = []

        def show_message(self, message, level, source):
            self.messages.append((message, level, source))

    coord.display_system = FakeDisplay()
    d = coord.to_dict()
    assert d.get("has_display_service") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("session_class", [AmplifierSession, RustSession])
async def test_session_constructor_wires_notification_sink(session_class):
    display = RecordingDisplay()
    session = session_class(
        {"session": {"orchestrator": "test", "context": "test"}},
        display_system=display,
    )

    async def handler(event, data):
        return HookResult(
            user_message="session notice", user_message_level="warning",
            user_message_source="teamwork",
        )

    session.coordinator.hooks.register("test:notice", handler)
    await session.coordinator.hooks.emit("test:notice", {})
    assert display.messages == [("session notice", "warning", "teamwork")]
    assert session.coordinator.to_dict()["has_display_service"] is True


@pytest.mark.asyncio
async def test_rust_session_direct_lifecycle_emit_delivers_notice():
    """RustSession.execute bypasses the Python registry's emit method."""
    display = RecordingDisplay()
    session = RustSession(
        {"session": {"orchestrator": "test", "context": "test"}},
        display_system=display,
    )

    async def handler(event, data):
        return HookResult(user_message="starting")

    session.coordinator.hooks.register("session:start", handler, name="lifecycle")
    with patch("amplifier_core._session_init.initialize_session", AsyncMock()):
        await session.initialize()
    session.coordinator.mount_points["orchestrator"] = AsyncMock(
        execute=AsyncMock(return_value="done")
    )
    session.coordinator.mount_points["context"] = AsyncMock()
    session.coordinator.mount_points["providers"] = {"test": AsyncMock()}
    assert await session.execute("synthetic test") == "done"
    assert display.messages == [("starting", "info", "lifecycle")]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["continue", "modify", "inject_context", "deny"])
async def test_notices_preserve_metadata_and_legacy_processing_does_not_duplicate(action):
    display = RecordingDisplay()
    coord = RustCoordinator(display_system=display)
    context = AsyncMock()
    await coord.mount("context", context)

    async def first(event, data):
        return HookResult(user_message="first", user_message_source="explicit-source")

    async def second(event, data):
        return HookResult(
            action=action, user_message="second", user_message_level="error",
            data={"changed": True} if action == "modify" else None,
            context_injection="injected" if action == "inject_context" else None,
            reason="blocked" if action == "deny" else None,
        )

    coord.hooks.register("test:event", first, priority=0, name="first-hook")
    coord.hooks.register("test:event", second, priority=1, name="second-hook")
    result = await coord.hooks.emit("test:event", {})
    assert display.messages == [
        ("first", "info", "explicit-source"),
        ("second", "error", "second-hook"),
    ]
    context.add_message.assert_not_called()
    if action == "modify":
        assert result.action == "continue"
        assert result.data == {"changed": True}
    else:
        assert result.action == action
    if action == "deny":
        assert result.reason == "blocked"
    if action == "inject_context":
        assert result.context_injection == "injected"

    await coord.process_hook_result(result, "test:event")
    assert len(display.messages) == 2
    assert context.add_message.call_count == (1 if action == "inject_context" else 0)


@pytest.mark.asyncio
async def test_notifications_do_not_request_approval():
    display = RecordingDisplay()
    approval = AsyncMock()
    coord = RustCoordinator(display_system=display, approval_system=approval)

    async def handler(event, data):
        return HookResult(
            action="ask_user", user_message="approval notice", approval_prompt="Allow?"
        )

    coord.hooks.register("test:event", handler)
    result = await coord.hooks.emit("test:event", {})
    assert result.action == "ask_user"
    assert result.user_message == "approval notice"
    assert result.approval_prompt == "Allow?"
    assert display.messages == []
    approval.request_approval.assert_not_called()


@pytest.mark.asyncio
async def test_display_can_be_replaced_and_cleared():
    first, second = RecordingDisplay(), RecordingDisplay()
    coord = RustCoordinator()

    async def handler(event, data):
        return HookResult(user_message="notice")

    coord.hooks.register("test:event", handler, name="source")
    coord.display_system = first
    await coord.hooks.emit("test:event", {})
    coord.display_system = second
    await coord.hooks.emit("test:event", {})
    coord.display_system = None
    await coord.hooks.emit("test:event", {})
    assert first.messages == second.messages == [("notice", "info", "source")]
    assert coord.to_dict()["has_display_service"] is False


@pytest.mark.asyncio
async def test_broken_display_does_not_prevent_deny():
    class BrokenDisplay:
        def show_message(self, message, level, source):
            raise RuntimeError("synthetic display failure")

    coord = RustCoordinator(display_system=BrokenDisplay())

    async def handler(event, data):
        return HookResult(action="deny", reason="blocked", user_message="notice")

    later = AsyncMock(return_value=HookResult())
    coord.hooks.register("test:event", handler, priority=0)
    coord.hooks.register("test:event", later, priority=1)
    result = await coord.hooks.emit("test:event", {})
    assert result.action == "deny"
    assert result.reason == "blocked"
    later.assert_not_called()
