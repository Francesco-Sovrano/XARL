from ray.rllib.agents.impala.impala import *
from ray.rllib.agents.ppo.appo import *
from ray.rllib.agents.ppo.appo_tf_policy import *
from ray.rllib.agents.ppo.ppo_tf_policy import vf_preds_fetches
from ray.rllib.agents.ppo.appo_torch_policy import torch, AsyncPPOTorchPolicy, appo_surrogate_loss as torch_appo_surrogate_loss
from ray.rllib.agents.impala.vtrace_tf_policy import _make_time_major as tf_make_time_major
from ray.rllib.agents.impala.vtrace_torch_policy import make_time_major as torch_make_time_major
from experience_buffers.replay_ops import MixInReplay

PRIO_WEIGHTS = "weights"
# DEFAULT_CONFIG = DEFAULT_CONFIG.copy()
DEFAULT_CONFIG["worker_side_prioritization"] = True

########################
# XAPPO's Trajectory Post-Processing
########################

def xappo_postprocess_trajectory(policy, sample_batch, other_agent_batches=None, episode=None):
    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                               sample_batch[SampleBatch.ACTIONS][-1],
                               sample_batch[SampleBatch.REWARDS][-1],
                               *next_state)
    sample_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
        use_critic=policy.config["use_critic"]
    )
    # Add PPO's importance weights
    _,_,actions_info = policy.compute_actions_from_input_dict(sample_batch, episodes=[episode])
    action_logp = actions_info['action_logp']
    old_action_logp = sample_batch[SampleBatch.ACTION_LOGP]
    logp_ratio = np.exp(action_logp - old_action_logp)
    advantages = sample_batch[Postprocessing.ADVANTAGES]
    new_priorities = advantages * logp_ratio
    if policy.config["worker_side_prioritization"]:
        sample_batch.data[PRIO_WEIGHTS] = new_priorities
    else:
        sample_batch[PRIO_WEIGHTS] = new_priorities
    del sample_batch.data["new_obs"]  # not used, so save some bandwidth
    return sample_batch

XAPPOTFPolicy = AsyncPPOTFPolicy.with_updates(
    extra_action_fetches_fn=vf_preds_fetches,
    postprocess_fn=xappo_postprocess_trajectory,
)
XAPPOTorchPolicy = AsyncPPOTorchPolicy.with_updates(
    extra_action_out_fn=vf_preds_fetches,
    postprocess_fn=xappo_postprocess_trajectory,
)

########################
# XAPPO's Execution Plan
########################

def xappo_get_policy_class(config):
    if config["framework"] == "torch":
        return XAPPOTorchPolicy
    # return XAPPOTFPolicy

def gather_experiences_directly(workers, config):
    rollouts = ParallelRollouts(workers, mode="async", num_async=config["max_sample_requests_in_flight_per_worker"])
    # get_batch_infos = lambda batch: batch[SampleBatch.INFOS]

    # Augment with replay and concat to desired train batch size.
    train_batches = rollouts \
        .for_each(lambda batch: batch.decompress_if_needed()) \
        .for_each(MixInReplay(
            num_slots=config["replay_buffer_num_slots"],
            replay_proportion=config["replay_proportion"])) \
        .flatten() \
        .combine(ConcatBatches(min_batch_size=config["train_batch_size"]))

    return train_batches

def xappo_execution_plan(workers, config):
    train_batches = gather_experiences_directly(workers, config)

    # Start the learner thread.
    learner_thread = make_learner_thread(workers.local_worker(), config)
    learner_thread.start()

    # This sub-flow sends experiences to the learner.
    enqueue_op = train_batches \
        .for_each(Enqueue(learner_thread.inqueue))
    # Only need to update workers if there are remote workers.
    if workers.remote_workers():
        enqueue_op = enqueue_op.zip_with_source_actor() \
            .for_each(BroadcastUpdateLearnerWeights(learner_thread, workers, broadcast_interval=config["broadcast_interval"]))

    # This sub-flow updates the steps trained counter based on learner output.
    dequeue_op = Dequeue(learner_thread.outqueue, check=learner_thread.is_alive) \
        .for_each(record_steps_trained)

    merged_op = Concurrently([enqueue_op, dequeue_op], mode="async", output_indexes=[1])

    # Callback for APPO to use to update KL, target network periodically.
    # The input to the callback is the learner fetches dict.
    if config["after_train_step"]:
        merged_op = merged_op.for_each(lambda t: t[1]).for_each(config["after_train_step"](workers, config))

    return StandardMetricsReporting(merged_op, workers, config) \
        .for_each(learner_thread.add_learner_metrics)

XAPPOTrainer = APPOTrainer.with_updates(
    default_policy=XAPPOTFPolicy,
    get_policy_class=xappo_get_policy_class,
    execution_plan=xappo_execution_plan,
)