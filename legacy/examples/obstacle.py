import pymc as pm
from pymc.math import switch, eq, and_, or_, ge, lt, minimum
from actual_cause.environment import Environment
import numpy as np


class Mover1D(Environment):
    def __init__(self, n_samples_per_obs=1):
        super().__init__(n_samples_per_obs)

    def build_model(self):
        self.description = """
        1D mover
        """

        with self.model:
            line_length = 4
            concentration_params = np.ones(line_length)
            u_mover = pm.Dirichlet("u_mover", concentration_params)
            u_obstacle = pm.Dirichlet("u_obstacle", concentration_params)
            current_mover_pos = pm.Categorical("mover", u_mover)
            obstacle_pos = pm.Categorical("obstacle", u_obstacle)
            next_mover_pos = pm.Deterministic(
                "next_mover_pos",
                switch(
                    or_(
                        eq(current_mover_pos + 1, obstacle_pos),
                        eq(current_mover_pos, line_length - 1),
                    ),
                    current_mover_pos,
                    current_mover_pos + 1,
                ),
            )


class BlockPusher1D(Environment):
    def __init__(self, n_samples_per_obs=1):
        super().__init__(n_samples_per_obs)

    def build_model(self):
        self.description = """
        1D block pusher
        """

        with self.model:
            line_length = 100
            push_length = 2
            u_current_pusher_pos = pm.Uniform(
                "u_current_pusher_pos", 0, line_length - 1
            )
            u_current_block_pos = pm.Deterministic(
                "u_current_block_pos", u_current_pusher_pos + 1
            )
            u_obstacle_pos = pm.Uniform("u_obstacle_pos", 0, line_length)
            current_pusher_pos = pm.Deterministic(
                "current_pusher_pos", u_current_pusher_pos
            )
            current_block_pos = pm.Deterministic(
                "current_block_pos", u_current_block_pos
            )
            obstacle_pos = pm.Deterministic("obstacle_pos", u_obstacle_pos)
            next_block_pos = pm.Deterministic(
                "next_block_pos",
                switch(
                    and_(
                        ge(obstacle_pos, current_pusher_pos),
                        lt(obstacle_pos, current_block_pos + push_length),
                    ),
                    current_block_pos,
                    minimum(current_block_pos + push_length, line_length),
                ),
            )
            next_pusher_pos = pm.Deterministic(
                "next_pusher_pos",
                switch(
                    eq(next_block_pos, current_block_pos + push_length),
                    minimum(next_block_pos - 1, line_length - 1),
                    current_pusher_pos,
                ),
            )
