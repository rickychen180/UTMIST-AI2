# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
# 
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
# 
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
# 
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here
# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!

import os
import gdown
from typing import Optional
from environment.agent import Agent
from stable_baselines3 import PPO, A2C # Sample RL Algo imports
from sb3_contrib import RecurrentPPO # Importing an LSTM

# To run the sample TTNN model, you can uncomment the 2 lines below: 
# import ttnn
# from user.my_agent_tt import TTMLPPolicy

from enum import Enum
import numpy as np

class X_SECTION(Enum):
    LEFT_EDGE = "left_edge"
    LEFT_PLATFORM = "left_platform"
    RIGHT_EDGE = "right_edge"
    RIGHT_PLATFORM = "right_platform"
    MIDDLE = "middle"


class Y_SECTION(Enum):
    BOTTOM = "bottom" # (> 0.85)
    MIDDLE = "middle"
    TOP = "top" # (<= 2.85)

class SubmittedAgent(Agent):
    '''
    Better BasedAgent
    '''
    HORIZONTAL_THRESHOLD = 2.0
    VERTICAL_THRESHOLD = 2.0
    CHARACTER_HEIGHT = 0.4
    CHARACTER_WIDTH = 0.4
    HEIGHT_DIFFERENCE_RANGE = 0.15
    X_DIFFERENCE_RANGE = 0.8

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0
        self.x_section: X_SECTION = X_SECTION.LEFT_PLATFORM
        self.y_section: Y_SECTION = Y_SECTION.MIDDLE
        self.taunted_once: bool = False

    def get_section(self, pos: tuple[float, float]):
        if pos[0] < -7.0 - self.CHARACTER_WIDTH + self.HORIZONTAL_THRESHOLD:
            x_section = X_SECTION.LEFT_EDGE
        elif pos[0] < -2.0 + self.CHARACTER_WIDTH - self.HORIZONTAL_THRESHOLD:
            x_section = X_SECTION.LEFT_PLATFORM
        elif pos[0] < 2.0 - self.CHARACTER_WIDTH + self.HORIZONTAL_THRESHOLD:
            x_section = X_SECTION.MIDDLE
        elif pos[0] < 7.0 + self.CHARACTER_WIDTH - self.HORIZONTAL_THRESHOLD:
            x_section = X_SECTION.RIGHT_PLATFORM
        else:
            x_section = X_SECTION.RIGHT_EDGE

        if pos[1] < 0.85 - self.CHARACTER_HEIGHT:
            y_section = Y_SECTION.TOP
        elif pos[1] < 2.85 - self.CHARACTER_HEIGHT + self.VERTICAL_THRESHOLD:
            y_section = Y_SECTION.MIDDLE
        else:
            y_section = Y_SECTION.BOTTOM

        return x_section, y_section


    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        vel = self.obs_helper.get_section(obs, 'player_vel')
        self_state = self.obs_helper.get_section(obs, 'player_state')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [11]
        action = self.act_helper.zeros()
        prev_vel_y = 0
        self_jumps_left = self.obs_helper.get_section(obs, 'player_jumps_left')
        self_recoveries_left = self.obs_helper.get_section(obs, 'player_recoveries_left')
        platform_pos = self.obs_helper.get_section(obs, 'player_moving_platform_pos')
        spawners = np.array([self.obs_helper.get_section(obs, f'player_spawner_{i+1}') for i in range(4)])
        self_weapon_type = self.obs_helper.get_section(obs, 'player_weapon_type')
        self_dodge_timer = self.obs_helper.get_section(obs, 'player_dodge_timer')
        self_grounded = self.obs_helper.get_section(obs, 'player_grounded')

        self.x_section, self.y_section = self.get_section(pos)
        opp_x_section, opp_y_section = self.get_section(opp_pos)

        # Choose target (weapon if none, then opponent)
        target_pos = None
        active_spawners = spawners[np.where(np.logical_and(spawners[:, 2] != 0, spawners[:, 2] != 2))][:,0:2]
        # Prefer hammers
        if self_weapon_type == 0 or self_weapon_type == 1:
            if len(active_spawners) > 0:
                spawner_distances = np.pow(pos[0] - active_spawners[:,0], 2) + np.pow(pos[1] - active_spawners[:,1], 2)
                target_pos = active_spawners[np.argmin(spawner_distances)]
            else:
                target_pos = opp_pos
        elif not opp_KO:
            target_pos = opp_pos

        # Horizontal Movement
        if self.x_section == X_SECTION.LEFT_EDGE:
            action = self.act_helper.press_keys(['d'])
        elif self.x_section == X_SECTION.RIGHT_EDGE:
            action = self.act_helper.press_keys(['a'])
        elif (
            self.x_section == X_SECTION.MIDDLE and 
            pos[1] > platform_pos[1] + self.CHARACTER_HEIGHT
        ):
            action = self.act_helper.press_keys(['a'])
        elif target_pos is not None:
            # Head towards target
            if (target_pos[0] > pos[0] + self.X_DIFFERENCE_RANGE):
                action = self.act_helper.press_keys(['d'])
            elif (target_pos[0] < pos[0] - self.X_DIFFERENCE_RANGE):
                action = self.act_helper.press_keys(['a'])

        # Vertical Movement
        if (
            self.y_section == Y_SECTION.BOTTOM or
            (self.x_section == X_SECTION.RIGHT_EDGE and self.y_section == Y_SECTION.MIDDLE)
        ):
            if self.time % 2 == 0:
                action = self.act_helper.press_keys(['space'], action)
            if vel[1] > prev_vel_y and self_recoveries_left == 1 and self.time % 2 == 1:
                action = self.act_helper.press_keys(['k'], action)
        elif target_pos is not None and pos[1] > target_pos[1] + self.HEIGHT_DIFFERENCE_RANGE and self_jumps_left == 0 and self.time % 2 == 0:
            # Jump towards target
            action = self.act_helper.press_keys(['space'], action)

        # Attack if near and not recovering or below platform
        if (
            self.x_section == X_SECTION.LEFT_PLATFORM or
            self.x_section == X_SECTION.RIGHT_PLATFORM or
            (
                self.x_section == X_SECTION.MIDDLE and 
                pos[1] < platform_pos[1] - self.CHARACTER_HEIGHT - self.VERTICAL_THRESHOLD
            ) or (
                self.x_section == X_SECTION.MIDDLE and 
                pos[1] < platform_pos[1] - self.CHARACTER_HEIGHT and
                self_grounded
            )
        ):
            # Engagement range based on weapon
            if self_weapon_type == 0:
                if pos[1] < opp_pos[1] - self.HEIGHT_DIFFERENCE_RANGE:
                    squared_engagement_range = 9
                else:
                    squared_engagement_range = 4
            elif self_weapon_type == 1:
                if pos[1] < opp_pos[1] - self.HEIGHT_DIFFERENCE_RANGE:
                    if vel[1] < 0:
                        squared_engagement_range = 4
                    else:
                        squared_engagement_range = 9
                else:
                    squared_engagement_range = 6.25
            else:
                if pos[1] < opp_pos[1] - self.HEIGHT_DIFFERENCE_RANGE:
                    if vel[1] < 0:
                        squared_engagement_range = 4
                    else:
                        squared_engagement_range = 12.25
                else:
                    squared_engagement_range = 6.25

            if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < squared_engagement_range:
                # Weights for holding s,w,a,d
                directions = ['', 's', 'w']
                direction_weights = np.array([10, 5, 5])
                if pos[1] < opp_pos[1] - self.HEIGHT_DIFFERENCE_RANGE:
                    direction_weights[1] += 25
                    direction_weights[2] -= 5
                elif pos[1] > opp_pos[1] + self.HEIGHT_DIFFERENCE_RANGE:
                    direction_weights[1] -= 5
                    direction_weights[2] += 25
                # Hammer, prefer s
                if self_weapon_type == 2 and self_grounded:
                    direction_weights[1] += 35
                # No s in middle section
                if self.x_section == X_SECTION.MIDDLE and not self_grounded:
                    direction_weights[1] = 0
                direction_choice = np.random.choice(directions, p=(direction_weights / np.sum(direction_weights)))
                
                x_directions = ['', 's', 'w']
                x_direction_weights = np.array([40, 5, 5])
                x_direction_choice = np.random.choice(x_directions, p=(x_direction_weights / np.sum(x_direction_weights)))

                # Weights for holding j, k, l, space
                attacks = ['', 'k', 'j', 'space', 'l',]
                attack_weights = np.zeros(5)
                attack_weights[0], attack_weights[2] = 5, 30
                if self_grounded:
                    if self_weapon_type == 0:
                        attack_weights[1] = 0
                else:
                    if self_weapon_type == 2:
                        attack_weights[1] = 10
                if self_jumps_left == 0 and pos[1] >= opp_pos[1] - self.HEIGHT_DIFFERENCE_RANGE:
                    if self_weapon_type == 2:
                        attack_weights[3] = 5
                    else:
                        attack_weights[3] = 10
                if self_dodge_timer == 0:
                    attack_weights[4] = 10
                attack_choice = np.random.choice(attacks, p=(attack_weights / np.sum(attack_weights)))

                if direction_choice:
                    action = self.act_helper.press_keys([direction_choice], action)
                if x_direction_choice:
                    action = self.act_helper.press_keys([x_direction_choice], action)
                if attack_choice:
                    action = self.act_helper.press_keys([attack_choice], action)

        # Pickup
        # Check for nearby spawner
        if np.any(
            np.logical_and(
                np.pow(pos[0] - spawners[:, 0], 2) + np.pow(pos[1] - spawners[:, 1], 2) < 1.5,
                np.logical_and(spawners[:, 2] != 0, spawners[:, 2] != 2)
            )
        # Prefer hammers
        ) and self.time % 2 == 0 and self_weapon_type in [0, 1]:
            action = self.act_helper.press_keys(['h'], action)

        # Taunting
        if (
            target_pos is None and
            not self.taunted_once and 
            self.time % 2 == 0 and
            self_grounded and
            self_state not in [9, 10] and
            (
                self.x_section in [X_SECTION.LEFT_PLATFORM, X_SECTION.RIGHT_PLATFORM] or
                (self.x_section == X_SECTION.MIDDLE and pos[1] < platform_pos[1] - self.CHARACTER_HEIGHT)
            )
        ):
            action = self.act_helper.press_keys(['g'], action)
        elif (self_state == 12):
            self.taunted_once = True
        if (not opp_KO):
            self.taunted_once = False

        prev_vel_y = vel[1]
        return action