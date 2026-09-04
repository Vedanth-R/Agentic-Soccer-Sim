"""Watch the trained 3v2 swarm. This file only draws and controls playback."""

import argparse

import pygame

from neural_network import load_model
from swarm_3v2 import Swarm3v2, model_actions


WINDOW = (1100, 760)
MARGIN = 55
TOP = 120
FIELD_COLOR = (48, 139, 79)
LINE_COLOR = (240, 240, 230)
TEAM_COLORS = ((54, 133, 255), (238, 73, 78))


def screen_position(env, position):
    """Convert pitch metres into screen pixels."""

    scale = min((WINDOW[0] - 2 * MARGIN) / env.width, (WINDOW[1] - TOP - MARGIN) / env.height)
    field_width = env.width * scale
    left = (WINDOW[0] - field_width) / 2
    return round(left + position[0] * scale), round(TOP + position[1] * scale)


def draw(screen, env, font, paused, speed, seed):
    screen.fill((24, 31, 36))
    top_left = screen_position(env, (0, 0))
    bottom_right = screen_position(env, (env.width, env.height))
    field = pygame.Rect(
        top_left[0], top_left[1], bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]
    )
    pygame.draw.rect(screen, FIELD_COLOR, field)

    # Alternating grass stripes.
    for stripe in range(10):
        if stripe % 2:
            x1 = screen_position(env, (stripe * env.width / 10, 0))[0]
            x2 = screen_position(env, ((stripe + 1) * env.width / 10, 0))[0]
            pygame.draw.rect(screen, (44, 132, 74), (x1, field.top, x2 - x1, field.height))

    pygame.draw.rect(screen, LINE_COLOR, field, 3)
    center = screen_position(env, (env.width / 2, env.height / 2))
    pygame.draw.line(screen, LINE_COLOR, (center[0], field.top), (center[0], field.bottom), 2)
    pygame.draw.circle(screen, LINE_COLOR, center, round(9.15 * field.width / env.width), 2)

    # The yellow line is the attackers' objective.
    progress_x = screen_position(env, (env.progression_x, 0))[0]
    pygame.draw.line(screen, (255, 210, 70), (progress_x, field.top), (progress_x, field.bottom), 3)

    for player in env.players.values():
        center = screen_position(env, player.position)
        if player.has_ball:
            pygame.draw.circle(screen, (255, 210, 70), center, 16, 3)
        pygame.draw.circle(screen, TEAM_COLORS[player.team], center, 11)
        pygame.draw.circle(screen, LINE_COLOR, center, 11, 2)
        label = font.render(str(player.number), True, (255, 255, 255))
        screen.blit(label, label.get_rect(center=center))

    ball = screen_position(env, env.ball.position)
    pygame.draw.circle(screen, (250, 250, 240), ball, 6)
    pygame.draw.circle(screen, (25, 25, 25), ball, 6, 2)

    seconds = env.tick / env.ticks_per_second
    mode = "PAUSED" if paused else "PLAYING"
    status = f"{mode}   {speed:g}x   {seconds:.1f}s   SEED {seed}   {env.result.upper()}"
    screen.blit(font.render(status, True, (245, 245, 245)), (MARGIN, 28))
    help_text = "Space: pause   R: replay   N: new positions   -/+: speed   Esc: quit"
    screen.blit(font.render(help_text, True, (180, 190, 195)), (MARGIN, 58))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/ppo_swarm_3v2.pt")
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument(
        "--jitter",
        type=float,
        default=2.0,
        help="maximum random position change in metres",
    )
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode(WINDOW)
    pygame.display.set_caption("PitchLab 3v2 Swarm")
    font = pygame.font.Font(None, 23)
    clock = pygame.time.Clock()
    model = load_model(args.model)
    env = Swarm3v2(starting_jitter=args.jitter)
    current_seed = args.seed
    observations = env.reset(current_seed)
    speeds = (0.25, 0.5, 1.0, 2.0, 4.0)
    speed_index = 2
    paused = False
    accumulated_time = 0.0
    running = True

    while running:
        real_seconds = clock.tick(60) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                paused = not paused
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                observations = env.reset(current_seed)
                accumulated_time = 0.0
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                current_seed += 1
                observations = env.reset(current_seed)
                accumulated_time = 0.0
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                speed_index = max(0, speed_index - 1)
            elif event.type == pygame.KEYDOWN and event.key in (
                pygame.K_EQUALS,
                pygame.K_PLUS,
                pygame.K_KP_PLUS,
            ):
                speed_index = min(len(speeds) - 1, speed_index + 1)

        if not paused and env.result == "running":
            accumulated_time += real_seconds * speeds[speed_index]
            while accumulated_time >= 1 / env.ticks_per_second:
                observations, _, _, _ = env.step(model_actions(model, observations))
                accumulated_time -= 1 / env.ticks_per_second

        draw(screen, env, font, paused, speeds[speed_index], current_seed)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
