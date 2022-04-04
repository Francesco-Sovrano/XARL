from ray.tune.registry import register_env
######### Add new environment below #########

### XA Atari
from environments.special_atari.special_atari import SpecialAtariEnv
for game in [
	"adventure",
	"air_raid",
	"alien",
	"amidar",
	"assault",
	"asterix",
	"asteroids",
	"atlantis",
	"bank_heist",
	"battle_zone",
	"beam_rider",
	"berzerk",
	"bowling",
	"boxing",
	"breakout",
	"carnival",
	"centipede",
	"chopper_command",
	"crazy_climber",
	"defender",
	"demon_attack",
	"double_dunk",
	"elevator_action",
	"enduro",
	"fishing_derby",
	"freeway",
	"frostbite",
	"gopher",
	"gravitar",
	"hero",
	"ice_hockey",
	"jamesbond",
	"journey_escape",
	"kangaroo",
	"krull",
	"kung_fu_master",
	"montezuma_revenge",
	"ms_pacman",
	"name_this_game",
	"phoenix",
	"pitfall",
	"pong",
	"pooyan",
	"private_eye",
	"qbert",
	"riverraid",
	"road_runner",
	"robotank",
	"seaquest",
	"skiing",
	"solaris",
	"space_invaders",
	"star_gunner",
	"tennis",
	"time_pilot",
	"tutankham",
	"up_n_down",
	"venture",
	"video_pinball",
	"wizard_of_wor",
	"yars_revenge",
	"zaxxon",
]:
	for obs_type in ["image", "ram"]:
		for explanation_fn in ['rewards_only_explanation','rewards_n_lives_explanation']:
			# space_invaders should yield SpaceInvaders-v0 and SpaceInvaders-ram-v0
			name = "".join([g.capitalize() for g in game.split("_")])
			if obs_type == "ram":
				name = "{}-ram".format(name)

			nondeterministic = False
			if game == "elevator_action" and obs_type == "ram":
				# ElevatorAction-ram-v0 seems to yield slightly
				# non-deterministic observations about 10% of the time. We
				# should track this down eventually, but for now we just
				# mark it as nondeterministic.
				nondeterministic = True

			if explanation_fn == 'rewards_n_lives_explanation':
				name = '2'+name

			register_env(
				"XA{}-v0".format(name),
				lambda config: SpecialAtariEnv(**{
					"game": game,
					"obs_type": obs_type,
					"repeat_action_probability": 0.25,
					'explanation_fn': explanation_fn,
				})
			)

			register_env(
				"XA{}-v4".format(name),
				lambda config: SpecialAtariEnv(**{"game": game, "obs_type": obs_type, 'explanation_fn': explanation_fn,})
			)

			# Standard Deterministic (as in the original DeepMind paper)
			if game == "space_invaders":
				frameskip = 3
			else:
				frameskip = 4

			# Use a deterministic frame skip.
			register_env(
				"XA{}Deterministic-v0".format(name),
				lambda config: SpecialAtariEnv(**{
					"game": game,
					"obs_type": obs_type,
					"frameskip": frameskip,
					"repeat_action_probability": 0.25,
					'explanation_fn': explanation_fn,
				})
			)

			register_env(
				"XA{}Deterministic-v4".format(name),
				lambda config: SpecialAtariEnv(**{"game": game, "obs_type": obs_type, "frameskip": frameskip, 'explanation_fn': explanation_fn,})
			)

			register_env(
				"XA{}NoFrameskip-v0".format(name),
				lambda config: SpecialAtariEnv(**{
					"game": game,
					"obs_type": obs_type,
					"frameskip": 1,
					"repeat_action_probability": 0.25,
					'explanation_fn': explanation_fn,
				})
			)

			# No frameskip. (Atari has no entropy source, so these are
			# deterministic environments.)
			register_env(
				"XA{}NoFrameskip-v4".format(name),
				lambda config: SpecialAtariEnv(**{
					"game": game,
					"obs_type": obs_type,
					"frameskip": 1,
					'explanation_fn': explanation_fn,
				})
			)
