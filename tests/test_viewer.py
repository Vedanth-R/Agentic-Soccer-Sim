import pygame

from sim import SoccerWorld
from viewer.pygame_viewer import PitchRenderer, ViewerConfig


def test_pitch_coordinates_map_to_pitch_corners():
    pygame.font.init()
    world = SoccerWorld()
    renderer = PitchRenderer(world, ViewerConfig())
    assert renderer.to_screen((0.0, 0.0)) == renderer.pitch_rect.topleft
    assert renderer.to_screen(
        (world.config.pitch_length, world.config.pitch_width)
    ) == renderer.pitch_rect.bottomright


def test_renderer_draws_without_mutating_world():
    pygame.font.init()
    world = SoccerWorld()
    world.reset(seed=4)
    before = repr(world.state)
    config = ViewerConfig()
    surface = pygame.Surface(config.window_size)
    PitchRenderer(world, config).draw(surface, world.state, paused=False, playback_speed=1.0)
    assert repr(world.state) == before

