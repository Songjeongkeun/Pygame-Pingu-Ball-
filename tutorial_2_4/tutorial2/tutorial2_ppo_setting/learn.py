import numpy as np
import torch
import torch.nn.functional as F
from .network import action_distribution


def build_batch(net, streams, *, gamma_per_second=1.0, gae_lambda=.98):
    """Each stream holds consecutive steps from ONE environment.

    Bootstrap across a rollout boundary without leaking GAE between envs.
    A done transition has zero bootstrap even if that environment is reset.
    """
    flat = [tr for stream in streams for tr in stream]
    if not flat:
        raise ValueError('Empty rollout')
    obs = torch.as_tensor(np.stack([tr['obs'] for tr in flat]), dtype=torch.float32)
    next_obs = torch.as_tensor(np.stack([tr['next_obs'] for tr in flat]), dtype=torch.float32)
    actions = torch.tensor([tr['action'] for tr in flat], dtype=torch.int64)
    old_lp = torch.tensor([tr['logp'] for tr in flat], dtype=torch.float32)
    with torch.no_grad():
        _, v = net(obs)
        _, nv = net(next_obs)
    values = v.detach().numpy().astype(np.float64)
    next_values = nv.detach().numpy().astype(np.float64)
    advantages = np.empty(len(flat), dtype=np.float64)
    position = 0
    for stream in streams:
        length = len(stream)
        gae = 0.
        for i in range(length - 1, -1, -1):
            tr = stream[i]
            idx = position + i
            discount = gamma_per_second ** (tr['frames'] / 120.)
            mask = 0. if tr['done'] else 1.
            delta = tr['reward'] / 25. + discount * next_values[idx] * mask - values[idx]
            gae = delta + discount * gae_lambda * mask * gae
            advantages[idx] = gae
        position += length
    returns = advantages + values
    return obs, actions, old_lp, torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def ppo_update(net, optimizer, streams, *, gamma_per_second=1., gae_lambda=.98,
               epochs=5, batch_size=256, clip=.15, entropy_coef=.02,
               value_coef=.5, max_grad_norm=.5, target_kl=.03,
               greedy_mix=.05, normalize_advantage=True):
    if not any(streams):
        return {}
    obs, actions, oldlp, adv, returns = build_batch(
        net, streams, gamma_per_second=gamma_per_second, gae_lambda=gae_lambda)
    if normalize_advantage and len(adv)>1:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-7)
    N = len(obs)
    logs = []
    epochs_used = 0
    stopped_early = False
    for epoch in range(epochs):
        epochs_used += 1
        epoch_kls = []
        for ids in torch.randperm(N).split(batch_size):
            logits, value = net(obs[ids])
            dist = action_distribution(logits, greedy_mix=greedy_mix)
            lp = dist.log_prob(actions[ids])
            log_ratio = lp - oldlp[ids]
            ratio = log_ratio.exp()
            surrogate = torch.minimum(ratio * adv[ids], ratio.clamp(1.-clip, 1.+clip) * adv[ids])
            actor_loss = -surrogate.mean()
            critic_loss = F.smooth_l1_loss(value, returns[ids])
            entropy = dist.entropy().mean()
            loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                kl = ((ratio - 1.) - log_ratio).mean().item()
            epoch_kls.append(kl)
            logs.append((actor_loss.item(), critic_loss.item(), entropy.item(), kl))


        if target_kl > 0 and float(np.mean(epoch_kls)) > target_kl:
            stopped_early = True
            break
    arr = np.asarray(logs)
    return {'policy_loss':round(float(arr[:,0].mean()),4),
            'value_loss':round(float(arr[:,1].mean()),4),
            'entropy':round(float(arr[:,2].mean()),4),
            'approx_kl':round(float(arr[:,3].mean()),6),
            'epochs_used':epochs_used,
            'early_kl_stop':stopped_early,
            'batch_transitions':N,
            'entropy_coef':round(entropy_coef,6)}
