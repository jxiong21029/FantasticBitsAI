import copy

import numpy as np
import torch

from ppo import PPOTrainer, RolloutBuffer


class VectorizedPPOTrainer(PPOTrainer):
    def __init__(
        self,
        agents,
        env_kwargs=None,
        lr=1e-3,
        weight_decay=1e-4,
        rollout_steps=4096,
        gamma=0.99,
        gae_lambda=0.95,
        minibatch_size=64,
        epochs=3,
        ppo_clip_coeff=0.2,
        grad_clipping=10.0,
        entropy_reg=1e-5,
        vectorized_envs=1,
        rollout_device=torch.device("cpu"),
        train_device=torch.device("cpu"),
        seed=None,
    ):
        super().__init__(
            agents=agents,
            env_kwargs=env_kwargs,
            lr=lr,
            weight_decay=weight_decay,
            rollout_steps=rollout_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            minibatch_size=minibatch_size,
            epochs=epochs,
            ppo_clip_coeff=ppo_clip_coeff,
            grad_clipping=grad_clipping,
            entropy_reg=entropy_reg,
            rollout_device=rollout_device,
            train_device=train_device,
            seed=seed,
        )
        self.rollout_steps = rollout_steps
        self.buf = [
            RolloutBuffer(
                size=rollout_steps // vectorized_envs,
                gamma=gamma,
                lam=gae_lambda,
                logger=self.logger,
                device=train_device,
            )
            for _ in range(vectorized_envs)
        ]
        self.env = [copy.deepcopy(self.env) for _ in range(vectorized_envs)]
        self.env_obs = [env.reset() for env in self.env]
        self.env_idx = 0

    def collect_rollout(self):
        self.agents.to(self.rollout_device)
        self.agents.eval()
        with torch.no_grad():
            actions, logps = None, None
            for t in range(self.rollout_steps):
                if self.env_idx == 0:
                    actions, logps = self.agents.vectorized_step(self.env_obs)
                next_obs, reward, done = self.env[self.env_idx].step(
                    actions[self.env_idx]
                )

                buf = self.buf[self.env_idx]
                buf.store(
                    self.env_obs[self.env_idx],
                    actions[self.env_idx],
                    reward,
                    logps[self.env_idx],
                )
                self.env_obs[self.env_idx] = next_obs

                epoch_ended = t >= self.rollout_steps - len(self.env)
                if epoch_ended or done:
                    if done:
                        value = (0, 0)
                    else:
                        value = self.agents.predict_value(
                            self.env_obs[self.env_idx], None
                        )

                    buf.val_buf[
                        buf.path_start_idx : buf.ptr
                    ] = self.agents.value_forward(
                        {
                            "obs": {
                                k: torch.tensor(v[buf.path_start_idx : buf.ptr])
                                for k, v in buf.obs_buf.items()
                            }
                        },
                        np.arange(buf.ptr - buf.path_start_idx),
                    ).numpy()
                    buf.finish_path(value)

                    if done:
                        self.env_obs[self.env_idx] = self.env[self.env_idx].reset()

                self.env_idx = (self.env_idx + 1) % len(self.env)

        rollouts = [buf.get() for buf in self.buf]

        def rcat(objs):
            ret = {}
            for k in objs[0].keys():
                if isinstance(objs[0][k], dict):
                    ret[k] = rcat([obj[k] for obj in objs])
                else:
                    ret[k] = torch.cat([obj[k] for obj in objs], dim=0)
            return ret

        self.rollout = rcat(rollouts)


def main():
    import tqdm

    from architectures import VonMisesAgents

    trainer = VectorizedPPOTrainer(
        VonMisesAgents(),
        rollout_steps=4096,
        lr=10**-3.5,
        weight_decay=1e-4,
        minibatch_size=512,
        gae_lambda=0.975,
        epochs=3,
        env_kwargs={
            "reward_shaping_snaffle_goal_dist": True,
            "reward_own_goal": 3,
        },
        vectorized_envs=8,
    )

    for i in tqdm.trange(101):
        trainer.train()
        if i % 20 == 0:
            trainer.evaluate()
            trainer.logger.generate_plots()


if __name__ == "__main__":
    main()
