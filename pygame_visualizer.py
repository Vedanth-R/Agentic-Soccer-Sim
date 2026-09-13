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


def pitch_position(env, mouse_position):
    """Convert screen pixels back into pitch metres for mouse dragging."""

    scale = min((WINDOW[0] - 2 * MARGIN) / env.width, (WINDOW[1] - TOP - MARGIN) / env.height)
    field_width = env.width * scale
    left = (WINDOW[0] - field_width) / 2
    x = min(max((mouse_position[0] - left) / scale, 0), env.width)
    y = min(max((mouse_position[1] - TOP) / scale, 0), env.height)
    return x, y


def capture_setup(env):
    """Remember a starting arrangement so R can replay it exactly."""

    return {
        "positions": {number: player.position.copy() for number, player in env.players.items()},
        "owner": env.ball.owner,
    }


def restore_setup(env, setup, seed):
    """Reset episode counters, then restore the saved custom positions."""

    env.reset(seed)
    for number, position in setup["positions"].items():
        env.players[number].position = position.copy()
        env.players[number].velocity[:] = 0
        env.players[number].has_ball = False
    owner = setup["owner"]
    env.players[owner].has_ball = True
    env.ball.owner = owner
    env.ball.position = env.players[owner].position.copy()
    env.ball.velocity[:] = 0
    env.ball.target_player = None
    env.ball.possession_ticks = 0
    return env.observations()


def choose_ball_owner(env, number):
    for player in env.players.values():
        player.has_ball = False
    env.players[number].has_ball = True
    env.ball.owner = number
    env.ball.position = env.players[number].position.copy()
    env.ball.velocity[:] = 0
    env.ball.target_player = None


def draw(screen, env, font, paused, speed, seed, editing, selected_player):
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

    # Draw the goal that the attackers must actually score in.
    goal_top = screen_position(env, (env.width, env.goal_center_y - env.goal_width / 2))[1]
    goal_bottom = screen_position(env, (env.width, env.goal_center_y + env.goal_width / 2))[1]
    pygame.draw.rect(
        screen,
        LINE_COLOR,
        (field.right, goal_top, 16, goal_bottom - goal_top),
        3,
    )

    for player in env.players.values():
        center = screen_position(env, player.position)
        if player.number == selected_player:
            pygame.draw.circle(screen, (255, 165, 50), center, 19, 3)
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
    mode = "EDIT START" if editing else ("PAUSED" if paused else "PLAYING")
    status = f"{mode}   {speed:g}x   {seconds:.1f}s   SEED {seed}   {env.result.upper()}"
    screen.blit(font.render(status, True, (245, 245, 245)), (MARGIN, 28))
    if editing:
        help_text = "Drag blue players   1/2/3: choose ball   Enter: start   Esc: quit"
    else:
        help_text = "E: edit start   Space: pause   R: replay   N: new layout   -/+: speed"
    screen.blit(font.render(help_text, True, (180, 190, 195)), (MARGIN, 58))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/goal_swarm_3v2.pt")
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument(
        "--jitter",
        type=float,
        default=8.0,
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
    starting_setup = capture_setup(env)
    speeds = (0.25, 0.5, 1.0, 2.0, 4.0)
    speed_index = 2
    paused = False
    editing = False
    selected_player = None
    accumulated_time = 0.0
    running = True

    while running:
        real_seconds = clock.tick(60) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                observations = restore_setup(env, starting_setup, current_seed)
                editing = True
                paused = True
                selected_player = None
                accumulated_time = 0.0
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN and editing:
                starting_setup = capture_setup(env)
                observations = env.observations()
                editing = False
                paused = False
                accumulated_time = 0.0
            elif event.type == pygame.KEYDOWN and editing and event.key in (
                pygame.K_1,
                pygame.K_2,
                pygame.K_3,
            ):
                choose_ball_owner(env, event.key - pygame.K_0)
                observations = env.observations()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and not editing:
                paused = not paused
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                observations = restore_setup(env, starting_setup, current_seed)
                editing = False
                paused = False
                accumulated_time = 0.0
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                current_seed += 1
                observations = env.reset(current_seed)
                starting_setup = capture_setup(env)
                editing = False
                paused = False
                accumulated_time = 0.0
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and editing:
                for number in env.attacker_ids:
                    center = screen_position(env, env.players[number].position)
                    if (event.pos[0] - center[0]) ** 2 + (event.pos[1] - center[1]) ** 2 <= 18 ** 2:
                        selected_player = number
                        break
            elif event.type == pygame.MOUSEMOTION and selected_player is not None:
                env.players[selected_player].position[:] = pitch_position(env, event.pos)
                if env.ball.owner == selected_player:
                    env.ball.position = env.players[selected_player].position.copy()
                observations = env.observations()
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                selected_player = None
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                speed_index = max(0, speed_index - 1)
            elif event.type == pygame.KEYDOWN and event.key in (
                pygame.K_EQUALS,
                pygame.K_PLUS,
                pygame.K_KP_PLUS,
            ):
                speed_index = min(len(speeds) - 1, speed_index + 1)

        if not paused and not editing and env.result == "running":
            accumulated_time += real_seconds * speeds[speed_index]
            while accumulated_time >= 1 / env.ticks_per_second:
                observations, _, _, _ = env.step(model_actions(model, observations))
                accumulated_time -= 1 / env.ticks_per_second

        draw(
            screen,
            env,
            font,
            paused,
            speeds[speed_index],
            current_seed,
            editing,
            selected_player,
        )
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
