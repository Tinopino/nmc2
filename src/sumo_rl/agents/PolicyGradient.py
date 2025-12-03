
import numpy as np


class PolicyGradientAgent:
    def __init__(
        self,
        obs_dim,
        action_space,
        lr=1e-3,
        beta_rew=0.01,
        gamma=0.99,
        grad_clip=5.0,
    ):
        self.obs_dim = obs_dim
        self.n_actions = action_space.n
        self.lr = lr
        self.beta_rew = beta_rew
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.R_bar = 0.0

        rng = np.random.default_rng()
        # θ has shape (n_actions, obs_dim)
        self.theta = rng.normal(0.0, 0.01, size=(self.n_actions, self.obs_dim))

        self.last_phi = None
        self.last_probs = None
        self.last_action = None
        self.trajectory = []

    def _phi(self, obs):
        x = np.array(obs, dtype=np.float32).flatten()
        # optional normalisation
        return x

    def act(self, obs):
        phi = self._phi(obs)
        logits = self.theta @ phi  # shape (n_actions,)
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        action = np.random.choice(self.n_actions, p=probs)

        self.last_phi = phi
        self.last_probs = probs
        self.last_action = action
        return action

    def _finish_episode(self):
        if not self.trajectory:
            return

        # Compute discounted returns (reward-to-go)
        returns = []
        G = 0.0
        for (_, _, _, r) in reversed(self.trajectory):
            G = r + self.gamma * G
            returns.insert(0, G)

        # Normalize for numerical stability
        returns = np.array(returns, dtype=np.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Update running baseline with the episode mean return
        self.R_bar = (1 - self.beta_rew) * self.R_bar + self.beta_rew * float(returns.mean())

        # Policy gradient update for each time step
        for (phi, probs, action, _), Gt in zip(self.trajectory, returns):
            advantage = Gt - self.R_bar
            grad_log_pi = -np.outer(probs, phi)
            grad_log_pi[action] += phi

            step_grad = advantage * grad_log_pi
            grad_norm = np.linalg.norm(step_grad)
            if self.grad_clip is not None and grad_norm > self.grad_clip:
                step_grad *= self.grad_clip / (grad_norm + 1e-8)

            self.theta += self.lr * step_grad

        self.trajectory.clear()

    def learn(self, reward, done=False, next_state=None):
        if self.last_phi is None:
            return

        # Store the transition for the episode
        self.trajectory.append((self.last_phi, self.last_probs, self.last_action, reward))

        if done:
            self._finish_episode()
            self.last_phi = None
            self.last_probs = None
            self.last_action = None
