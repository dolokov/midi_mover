"""UI helpers for the new-highscore countdown screen."""

from __future__ import annotations

import time
from typing import Any


def show_highscore_countdown_screen(
    *,
    pygame_module: Any,
    window: Any,
    countdown_seconds: float,
) -> bool:
    """Render a visible highscore countdown sequence (3 → 2 → 1 → 0).

    Returns ``False`` when the user closes the window or presses ESC.
    """

    start_value = max(0, int(round(float(countdown_seconds))))
    zero_hold_seconds = 0.5
    start_time = time.monotonic()
    end_time = start_time + float(start_value) + zero_hold_seconds

    title_font = pygame_module.font.SysFont(None, 58, bold=True)
    message_font = pygame_module.font.SysFont(None, 34)
    countdown_font = pygame_module.font.SysFont(None, 220, bold=True)

    title_surface = title_font.render("New Highscore!", True, (241, 245, 249))
    message_surface = message_font.render("Get ready for your headshot", True, (191, 219, 254))

    while time.monotonic() < end_time:
        for event in pygame_module.event.get():
            if event.type == pygame_module.QUIT:
                return False
            if event.type == pygame_module.KEYDOWN and event.key == pygame_module.K_ESCAPE:
                return False

        elapsed = time.monotonic() - start_time
        if elapsed < float(start_value):
            countdown_value = start_value - int(elapsed)
        else:
            countdown_value = 0

        window.fill((5, 12, 28))
        panel_width = min(window.get_width() - 120, 820)
        panel_height = min(window.get_height() - 120, 560)
        panel = pygame_module.Rect(
            (window.get_width() - panel_width) // 2,
            (window.get_height() - panel_height) // 2,
            panel_width,
            panel_height,
        )
        pygame_module.draw.rect(window, (15, 23, 42), panel, border_radius=18)
        pygame_module.draw.rect(window, (96, 165, 250), panel, width=2, border_radius=18)

        title_rect = title_surface.get_rect(centerx=panel.centerx, y=panel.y + 30)
        window.blit(title_surface, title_rect)
        message_rect = message_surface.get_rect(centerx=panel.centerx, y=title_rect.bottom + 10)
        window.blit(message_surface, message_rect)

        countdown_surface = countdown_font.render(str(countdown_value), True, (248, 250, 252))
        countdown_rect = countdown_surface.get_rect(center=panel.center)
        window.blit(countdown_surface, countdown_rect)

        pygame_module.display.flip()
        pygame_module.time.delay(16)

    return True
