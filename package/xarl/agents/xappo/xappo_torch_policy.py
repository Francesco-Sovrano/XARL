"""
PyTorch policy class used for APPO.

Adapted from VTraceTFPolicy to use the PPO surrogate loss.
Keep in sync with changes to VTraceTFPolicy.
"""
from ray.rllib.agents.ppo.appo_torch_policy import *
from ray.rllib.agents.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.agents.ppo.ppo_tf_policy import vf_preds_fetches

def xappo_surrogate_loss(policy, model, dist_class, train_batch):
	"""Constructs the loss for APPO.

	With IS modifications and V-trace for Advantage Estimation.

	Args:
		policy (Policy): The Policy to calculate the loss for.
		model (ModelV2): The Model to calculate the loss for.
		dist_class (Type[ActionDistribution]): The action distr. class.
		train_batch (SampleBatch): The training data.

	Returns:
		Union[TensorType, List[TensorType]]: A single loss tensor or a list
			of loss tensors.
	"""
	target_model = policy.target_models[model]

	model_out, _ = model(train_batch)
	action_dist = dist_class(model_out, model)

	if isinstance(policy.action_space, gym.spaces.Discrete):
		is_multidiscrete = False
		output_hidden_shape = [policy.action_space.n]
	elif isinstance(policy.action_space, gym.spaces.multi_discrete.MultiDiscrete):
		is_multidiscrete = True
		output_hidden_shape = policy.action_space.nvec.astype(np.int32)
	else:
		is_multidiscrete = False
		output_hidden_shape = 1

	def _make_time_major(*args, **kwargs):
		return make_time_major(
			policy, train_batch.get(SampleBatch.SEQ_LENS), *args, **kwargs
		)

	actions = train_batch[SampleBatch.ACTIONS]
	dones = train_batch[SampleBatch.DONES]
	rewards = train_batch[SampleBatch.REWARDS]
	behaviour_logits = train_batch[SampleBatch.ACTION_DIST_INPUTS]

	target_model_out, _ = target_model(train_batch)

	prev_action_dist = dist_class(behaviour_logits, model)
	values = model.value_function()
	values_time_major = _make_time_major(values)

	drop_last = policy.config["vtrace"] and policy.config["vtrace_drop_last_ts"]

	if policy.is_recurrent():
		max_seq_len = torch.max(train_batch[SampleBatch.SEQ_LENS])
		mask = sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
		mask = torch.reshape(mask, [-1])
		mask = _make_time_major(mask, drop_last=drop_last)
		num_valid = torch.sum(mask)

		def reduce_mean_valid(t):
			return torch.sum(t[mask]) / num_valid

	else:
		reduce_mean_valid = torch.mean

	weights = _make_time_major(train_batch[PRIO_WEIGHTS], drop_last=drop_last)

	if policy.config["vtrace"]:
		logger.debug(
			"Using V-Trace surrogate loss (vtrace=True; " f"drop_last={drop_last})"
		)

		old_policy_behaviour_logits = target_model_out.detach()
		old_policy_action_dist = dist_class(old_policy_behaviour_logits, model)

		if isinstance(output_hidden_shape, (list, tuple, np.ndarray)):
			unpacked_behaviour_logits = torch.split(
				behaviour_logits, list(output_hidden_shape), dim=1
			)
			unpacked_old_policy_behaviour_logits = torch.split(
				old_policy_behaviour_logits, list(output_hidden_shape), dim=1
			)
		else:
			unpacked_behaviour_logits = torch.chunk(
				behaviour_logits, output_hidden_shape, dim=1
			)
			unpacked_old_policy_behaviour_logits = torch.chunk(
				old_policy_behaviour_logits, output_hidden_shape, dim=1
			)

		# Prepare actions for loss.
		loss_actions = actions if is_multidiscrete else torch.unsqueeze(actions, dim=1)

		# Prepare KL for loss.
		action_kl = _make_time_major(
			old_policy_action_dist.kl(action_dist), drop_last=drop_last
		)

		# Compute vtrace on the CPU for better perf.
		vtrace_returns = vtrace.multi_from_logits(
			behaviour_policy_logits=_make_time_major(
				unpacked_behaviour_logits, drop_last=drop_last
			),
			target_policy_logits=_make_time_major(
				unpacked_old_policy_behaviour_logits, drop_last=drop_last
			),
			actions=torch.unbind(
				_make_time_major(loss_actions, drop_last=drop_last), dim=2
			),
			discounts=(1.0 - _make_time_major(dones, drop_last=drop_last).float())
			* policy.config["gamma"],
			rewards=_make_time_major(rewards, drop_last=drop_last),
			values=values_time_major[:-1] if drop_last else values_time_major,
			bootstrap_value=values_time_major[-1],
			dist_class=TorchCategorical if is_multidiscrete else dist_class,
			model=model,
			clip_rho_threshold=policy.config["vtrace_clip_rho_threshold"],
			clip_pg_rho_threshold=policy.config["vtrace_clip_pg_rho_threshold"],
		)

		actions_logp = _make_time_major(action_dist.logp(actions), drop_last=drop_last)
		prev_actions_logp = _make_time_major(
			prev_action_dist.logp(actions), drop_last=drop_last
		)
		old_policy_actions_logp = _make_time_major(
			old_policy_action_dist.logp(actions), drop_last=drop_last
		)
		is_ratio = torch.clamp(
			torch.exp(prev_actions_logp - old_policy_actions_logp), 0.0, 2.0
		)
		logp_ratio = is_ratio * torch.exp(actions_logp - prev_actions_logp)
		policy._is_ratio = is_ratio

		advantages = vtrace_returns.pg_advantages.to(logp_ratio.device)
		surrogate_loss = torch.min(
			advantages * logp_ratio,
			advantages
			* torch.clamp(
				logp_ratio,
				1 - policy.config["clip_param"],
				1 + policy.config["clip_param"],
			),
		)

		if PRIO_WEIGHTS in train_batch:
			surrogate_loss *= weights
			action_kl *= weights
		mean_kl_loss = reduce_mean_valid(action_kl)
		mean_policy_loss = -reduce_mean_valid(surrogate_loss)

		# The value function loss.
		value_targets = vtrace_returns.vs.to(values_time_major.device)
		if drop_last:
			delta = values_time_major[:-1] - value_targets
		else:
			delta = values_time_major - value_targets
		if PRIO_WEIGHTS in train_batch:
		    delta *= weights
		mean_vf_loss = 0.5 * reduce_mean_valid(torch.pow(delta, 2.0))

		# The entropy loss.
		entropy = _make_time_major(action_dist.entropy(), drop_last=True)
		if PRIO_WEIGHTS in train_batch:
			entropy *= weights
		mean_entropy = reduce_mean_valid(entropy)

	else:
		logger.debug("Using PPO surrogate loss (vtrace=False)")

		# Prepare KL for Loss
		action_kl = _make_time_major(prev_action_dist.kl(action_dist))

		actions_logp = _make_time_major(action_dist.logp(actions))
		prev_actions_logp = _make_time_major(prev_action_dist.logp(actions))
		logp_ratio = torch.exp(actions_logp - prev_actions_logp)

		advantages = _make_time_major(train_batch[Postprocessing.ADVANTAGES])
		surrogate_loss = torch.min(
			advantages * logp_ratio,
			advantages
			* torch.clamp(
				logp_ratio,
				1 - policy.config["clip_param"],
				1 + policy.config["clip_param"],
			),
		)

		if PRIO_WEIGHTS in train_batch:
			surrogate_loss *= weights
			action_kl *= weights
		mean_kl_loss = reduce_mean_valid(action_kl)
		mean_policy_loss = -reduce_mean_valid(surrogate_loss)

		# The value function loss.
		value_targets = _make_time_major(train_batch[Postprocessing.VALUE_TARGETS])
		delta = values_time_major - value_targets
		if PRIO_WEIGHTS in train_batch:
			delta *= weights
		mean_vf_loss = 0.5 * reduce_mean_valid(torch.pow(delta, 2.0))

		# The entropy loss.
		entropy = _make_time_major(action_dist.entropy())
		if PRIO_WEIGHTS in train_batch:
			entropy *= weights
		mean_entropy = reduce_mean_valid(entropy)

	# The summed weighted loss
	total_loss = (
		mean_policy_loss
		+ mean_vf_loss * policy.config["vf_loss_coeff"]
		- mean_entropy * policy.entropy_coeff
	)

	# Optional additional KL Loss
	if policy.config["use_kl_loss"]:
		total_loss += policy.kl_coeff * mean_kl_loss

	# Store values for stats function in model (tower), such that for
	# multi-GPU, we do not override them during the parallel loss phase.
	model.tower_stats["total_loss"] = total_loss
	model.tower_stats["mean_policy_loss"] = mean_policy_loss
	model.tower_stats["mean_kl_loss"] = mean_kl_loss
	model.tower_stats["mean_vf_loss"] = mean_vf_loss
	model.tower_stats["mean_entropy"] = mean_entropy
	model.tower_stats["value_targets"] = value_targets
	model.tower_stats["vf_explained_var"] = explained_variance(
		torch.reshape(value_targets, [-1]),
		torch.reshape(values_time_major[:-1] if drop_last else values_time_major, [-1]),
	)

	return total_loss

# Han, Seungyul, and Youngchul Sung. "Dimension-Wise Importance Sampling Weight Clipping for Sample-Efficient Reinforcement Learning." arXiv preprint arXiv:1905.02363 (2019).
def gae_v(gamma, lambda_, last_value, reversed_reward, reversed_value, reversed_importance_weight):
	def generalized_advantage_estimator_with_vtrace(gamma, lambd, last_value, reversed_reward, reversed_value, reversed_rho):
		reversed_rho = np.minimum(1.0, reversed_rho)
		def get_return(last_gae, last_value, last_rho, reward, value, rho):
			new_gae = reward + gamma*last_value - value + gamma*lambd*last_gae
			return new_gae, value, rho, last_rho*new_gae
		reversed_cumulative_advantage, _, _, _ = zip(*accumulate(
			iterable=zip(reversed_reward, reversed_value, reversed_rho), 
			func=lambda cumulative_value,reward_value_rho: get_return(
				last_gae=cumulative_value[3], 
				last_value=cumulative_value[1], 
				last_rho=cumulative_value[2], 
				reward=reward_value_rho[0], 
				value=reward_value_rho[1],
				rho=reward_value_rho[2],
			),
			initial_value=(0.,last_value,1.,0.) # initial cumulative_value
		))
		reversed_cumulative_return = tuple(map(lambda adv,val,rho: rho*adv+val, reversed_cumulative_advantage, reversed_value, reversed_rho))
		return reversed_cumulative_return, reversed_cumulative_advantage
	return generalized_advantage_estimator_with_vtrace(
		gamma=gamma, 
		lambd=lambda_, 
		last_value=last_value, 
		reversed_reward=reversed_reward, 
		reversed_value=reversed_value,
		reversed_rho=reversed_importance_weight
	)

def compute_gae_v_advantages(rollout: SampleBatch, last_r: float, gamma: float = 0.9, lambda_: float = 1.0):
	rollout_size = len(rollout[SampleBatch.ACTIONS])
	assert SampleBatch.VF_PREDS in rollout, "values not found"
	reversed_cumulative_return, reversed_cumulative_advantage = gae_v(
		gamma, 
		lambda_, 
		last_r, 
		rollout[SampleBatch.REWARDS][::-1], 
		rollout[SampleBatch.VF_PREDS][::-1], 
		rollout["action_importance_ratio"][::-1]
	)
	rollout[Postprocessing.ADVANTAGES] = np.array(reversed_cumulative_advantage, dtype=np.float32)[::-1]
	rollout[Postprocessing.VALUE_TARGETS] = np.array(reversed_cumulative_return, dtype=np.float32)[::-1]
	assert all(val.shape[0] == rollout_size for key, val in rollout.items()), "Rollout stacked incorrectly!"
	return rollout

# TODO: (sven) Experimental method.
def get_single_step_input_dict(self, view_requirements, index="last"):
	"""Creates single ts SampleBatch at given index from `self`.

	For usage as input-dict for model calls.

	Args:
		sample_batch (SampleBatch): A single-trajectory SampleBatch object
			to generate the compute_actions input dict from.
		index (Union[int, str]): An integer index value indicating the
			position in the trajectory for which to generate the
			compute_actions input dict. Set to "last" to generate the dict
			at the very end of the trajectory (e.g. for value estimation).
			Note that "last" is different from -1, as "last" will use the
			final NEXT_OBS as observation input.

	Returns:
		SampleBatch: The (single-timestep) input dict for ModelV2 calls.
	"""
	last_mappings = {
		SampleBatch.OBS: SampleBatch.NEXT_OBS,
		SampleBatch.PREV_ACTIONS: SampleBatch.ACTIONS,
		SampleBatch.PREV_REWARDS: SampleBatch.REWARDS,
	}

	input_dict = {}
	for view_col, view_req in view_requirements.items():
		# Create batches of size 1 (single-agent input-dict).
		data_col = view_req.data_col or view_col
		if index == "last":
			data_col = last_mappings.get(data_col, data_col)
			# Range needed.
			if view_req.shift_from is not None:
				data = self[view_col][-1]
				traj_len = len(self[data_col])
				missing_at_end = traj_len % view_req.batch_repeat_value
				obs_shift = -1 if data_col in [
					SampleBatch.OBS, SampleBatch.NEXT_OBS
				] else 0
				from_ = view_req.shift_from + obs_shift
				to_ = view_req.shift_to + obs_shift + 1
				if to_ == 0:
					to_ = None
				input_dict[view_col] = np.array([
					np.concatenate(
						[data,
						 self[data_col][-missing_at_end:]])[from_:to_]
				])
			# Single index.
			else:
				data = self[data_col][-1]
				input_dict[view_col] = np.array([data])
		else:
			# Index range.
			if isinstance(index, tuple):
				data = self[data_col][index[0]:index[1] +
									  1 if index[1] != -1 else None]
				input_dict[view_col] = np.array([data])
			# Single index.
			else:
				input_dict[view_col] = self[data_col][
					index:index + 1 if index != -1 else None]

	return SampleBatch(input_dict, seq_lens=np.array([1], dtype=np.int32))

def xappo_postprocess_trajectory(policy, sample_batch, other_agent_batches=None, episode=None):
	# Add PPO's importance weights
	action_logp = policy.compute_log_likelihoods(
		actions=sample_batch[SampleBatch.ACTIONS],
		obs_batch=sample_batch[SampleBatch.CUR_OBS],
		state_batches=None, # missing, needed for RNN-based models
		prev_action_batch=None,
		prev_reward_batch=None,
	)
	old_action_logp = sample_batch[SampleBatch.ACTION_LOGP]
	sample_batch["action_importance_ratio"] = np.exp(action_logp - old_action_logp)
	if policy.config["buffer_options"]["prioritization_importance_beta"] and 'weights' not in sample_batch:
		sample_batch['weights'] = np.ones_like(sample_batch[SampleBatch.REWARDS])
	# sample_batch[Postprocessing.VALUE_TARGETS] = sample_batch[Postprocessing.ADVANTAGES] = np.ones_like(sample_batch[SampleBatch.REWARDS])
	# Add advantages, do it after computing action_importance_ratio (used by gae-v)
	if policy.config["update_advantages_when_replaying"] or Postprocessing.ADVANTAGES not in sample_batch:
		if sample_batch[SampleBatch.DONES][-1]:
			last_r = 0.0
		# Trajectory has been truncated -> last r=VF estimate of last obs.
		else:
			# Input dict is provided to us automatically via the Model's
			# requirements. It's a single-timestep (last one in trajectory)
			# input_dict.
			# Create an input dict according to the Model's requirements.
			input_dict = get_single_step_input_dict(sample_batch, policy.model.view_requirements, index="last")
			last_r = policy._value(**input_dict)

		# Adds the policy logits, VF preds, and advantages to the batch,
		# using GAE ("generalized advantage estimation") or not.
		
		if not policy.config["vtrace"] and policy.config["gae_with_vtrace"]:
			sample_batch = compute_gae_v_advantages(
				sample_batch, 
				last_r, 
				policy.config["gamma"], 
				policy.config["lambda"]
			)
		else:
			sample_batch = compute_advantages(
				sample_batch,
				last_r,
				policy.config["gamma"],
				policy.config["lambda"],
				use_gae=policy.config["use_gae"],
				use_critic=policy.config.get("use_critic", True)
			)
	# Add gains
	sample_batch['gains'] = sample_batch['action_importance_ratio']*sample_batch[Postprocessing.ADVANTAGES]
	return sample_batch

XAPPOTorchPolicy = AsyncPPOTorchPolicy.with_updates(
	name="XAPPOTorchPolicy",
	extra_action_out_fn=vf_preds_fetches,
	postprocess_fn=xappo_postprocess_trajectory,
	loss_fn=xappo_surrogate_loss,
)