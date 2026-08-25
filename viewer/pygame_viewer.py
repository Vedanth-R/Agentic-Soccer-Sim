"""A read-only Pygame view of a :class:`sim.state.WorldState`."""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import pygame

from sim import Action, ScenarioStatus, SoccerWorld, WorldState

Color = Tuple[int, int, int]
ActionProvider = Callable[[WorldState], Dict[int, Action]]


@dataclass(frozen=True)
class ViewerConfig:
    window_size: Tuple[int, int] = (1100, 760)
    margin: int = 55
    hud_height: int = 65
    frames_per_second: int = 60
    title: str = "PitchLab - 2D World"


class PitchRenderer:
    """Converts pitch coordinates to pixels and draws state without changing it."""

    BACKGROUND: Color = (24, 31, 36)
    GRASS: Color = (48, 139, 79)
    GRASS_ALT: Color = (44, 132, 74)
    LINE: Color = (235, 240, 230)
    TEAM_COLORS = ((54, 133, 255), (238, 73, 78), (247, 190, 51))

    def __init__(self, world: SoccerWorld, config: ViewerConfig) -> None:
        self.world = world
        self.config = config
        self.font = pygame.font.Font(None, 25)
        self.small_font = pygame.font.Font(None, 20)
        width, height = config.window_size
        available_width = width - 2 * config.margin
        available_height = height - 2 * config.margin - config.hud_height
        self.scale = min(
            available_width / world.config.pitch_length,
            available_height / world.config.pitch_width,
        )
        pitch_width = round(world.config.pitch_length * self.scale)
        pitch_height = round(world.config.pitch_width * self.scale)
        left = (width - pitch_width) // 2
        top = config.margin + config.hud_height
        self.pitch_rect = pygame.Rect(left, top, pitch_width, pitch_height)

    def to_screen(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """Map meters from the top-left of the pitch to screen pixels."""
        return (
            round(self.pitch_rect.left + position[0] * self.scale),
            round(self.pitch_rect.top + position[1] * self.scale),
        )

    def draw(
        self,
        surface: pygame.Surface,
        state: WorldState,
        paused: bool,
        playback_speed: float,
    ) -> None:
        surface.fill(self.BACKGROUND)
        self._draw_pitch(surface)
        self._draw_players(surface, state)
        self._draw_ball(surface, state)
        self._draw_hud(surface, state, paused, playback_speed)

    def _draw_pitch(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, self.GRASS, self.pitch_rect)
        stripe_width = self.pitch_rect.width / 10
        for index in range(10):
            if index % 2:
                stripe = pygame.Rect(
                    round(self.pitch_rect.left + index * stripe_width),
                    self.pitch_rect.top,
                    round(stripe_width + 1),
                    self.pitch_rect.height,
                )
                pygame.draw.rect(surface, self.GRASS_ALT, stripe)
        pygame.draw.rect(surface, self.LINE, self.pitch_rect, 3)
        center = self.to_screen(
            (self.world.config.pitch_length / 2, self.world.config.pitch_width / 2)
        )
        pygame.draw.line(
            surface,
            self.LINE,
            (center[0], self.pitch_rect.top),
            (center[0], self.pitch_rect.bottom),
            2,
        )
        pygame.draw.circle(surface, self.LINE, center, round(9.15 * self.scale), 2)
        pygame.draw.circle(surface, self.LINE, center, 4)
        self._draw_penalty_area(surface, left_side=True)
        self._draw_penalty_area(surface, left_side=False)
        self._draw_goals(surface)

    def _draw_penalty_area(self, surface: pygame.Surface, left_side: bool) -> None:
        depth = round(16.5 * self.scale)
        width = round(40.32 * self.scale)
        y = self.pitch_rect.centery - width // 2
        x = self.pitch_rect.left if left_side else self.pitch_rect.right - depth
        pygame.draw.rect(surface, self.LINE, pygame.Rect(x, y, depth, width), 2)

    def _draw_goals(self, surface: pygame.Surface) -> None:
        width = round(self.world.config.goal_width * self.scale)
        depth = max(round(2.0 * self.scale), 8)
        y = self.pitch_rect.centery - width // 2
        pygame.draw.rect(
            surface,
            self.LINE,
            pygame.Rect(self.pitch_rect.left - depth, y, depth, width),
            2,
        )
        pygame.draw.rect(
            surface,
            self.LINE,
            pygame.Rect(self.pitch_rect.right, y, depth, width),
            2,
        )

    def _draw_players(self, surface: pygame.Surface, state: WorldState) -> None:
        radius = max(round(1.25 * self.scale), 10)
        for player in sorted(state.players.values(), key=lambda item: item.player_id):
            center = self.to_screen(player.position)
            color = self.TEAM_COLORS[player.team % len(self.TEAM_COLORS)]
            if player.has_ball:
                pygame.draw.circle(surface, (255, 211, 74), center, radius + 5, 3)
            pygame.draw.circle(surface, color, center, radius)
            pygame.draw.circle(surface, (245, 245, 245), center, radius, 2)
            label = self.small_font.render(str(player.player_id), True, (255, 255, 255))
            surface.blit(label, label.get_rect(center=center))

    def _draw_ball(self, surface: pygame.Surface, state: WorldState) -> None:
        center = self.to_screen(state.ball.position)
        radius = max(round(0.45 * self.scale), 5)
        pygame.draw.circle(surface, (248, 248, 240), center, radius)
        pygame.draw.circle(surface, (30, 30, 30), center, radius, 2)

    def _draw_hud(
        self,
        surface: pygame.Surface,
        state: WorldState,
        paused: bool,
        playback_speed: float,
    ) -> None:
        seconds = state.tick / self.world.config.ticks_per_second
        mode = "PAUSED" if paused else "PLAYING"
        title = (
            f"{mode}   |   {playback_speed:g}x   |   "
            f"{seconds:05.1f}s   |   {state.status.value.upper()}"
        )
        help_text = "Space: pause   R: reset   -/+: speed   Esc: quit"
        surface.blit(self.font.render(title, True, (245, 245, 245)), (self.config.margin, 24))
        surface.blit(
            self.small_font.render(help_text, True, (180, 190, 195)),
            (self.config.margin, 50),
        )


class PygameViewer:
    """Runs display timing while delegating all soccer changes to the world."""

    SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)

    def __init__(
        self,
        world: SoccerWorld,
        action_provider: ActionProvider,
        config: ViewerConfig = ViewerConfig(),
        seed: int = 0,
    ) -> None:
        self.world = world
        self.action_provider = action_provider
        self.config = config
        self.seed = seed
        self.paused = False
        self.speed_index = self.SPEEDS.index(1.0)

    def run(self) -> None:
        pygame.init()
        screen = pygame.display.set_mode(self.config.window_size)
        pygame.display.set_caption(self.config.title)
        clock = pygame.time.Clock()
        renderer = PitchRenderer(self.world, self.config)
        self.world.reset(self.seed)
        accumulated_sim_seconds = 0.0
        running = True
        while running:
            real_seconds = clock.tick(self.config.frames_per_second) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running, accumulated_sim_seconds = self._handle_key(
                        event.key, running, accumulated_sim_seconds
                    )
            if not self.paused and self.world.state.status is ScenarioStatus.RUNNING:
                accumulated_sim_seconds += real_seconds * self.playback_speed
                while accumulated_sim_seconds >= self.world.config.dt:
                    self.world.step(self.action_provider(self.world.state))
                    accumulated_sim_seconds -= self.world.config.dt
            renderer.draw(screen, self.world.state, self.paused, self.playback_speed)
            pygame.display.flip()
        pygame.quit()

    @property
    def playback_speed(self) -> float:
        return self.SPEEDS[self.speed_index]

    def _handle_key(
        self, key: int, running: bool, accumulated_seconds: float
    ) -> Tuple[bool, float]:
        if key == pygame.K_ESCAPE:
            return False, accumulated_seconds
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_r:
            self.world.reset(self.seed)
            accumulated_seconds = 0.0
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed_index = max(0, self.speed_index - 1)
        elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.speed_index = min(len(self.SPEEDS) - 1, self.speed_index + 1)
        return running, accumulated_seconds

