from ray.tune.registry import register_env
######### Add new environment below #########

### CescoDrive
from .env.cesco_drive_v0 import CescoDriveV0
register_env("CescoDrive-V0", lambda config: CescoDriveV0(config))

from .env.cesco_drive_v1 import CescoDriveV1
register_env("CescoDrive-V1", lambda config: CescoDriveV1(config))

### GridDrive
from .env.grid_drive import GridDrive
culture_level_list = ["Easy","Medium","Hard"]
for culture_level in culture_level_list:
	register_env(f"GridDrive-{culture_level}", lambda config: GridDrive({"reward_fn": 'frequent_reward_default', "culture_level": culture_level}))
	register_env(f"GridDrive-{culture_level}-ExplanationEngineering-V1", lambda config: GridDrive({"reward_fn": 'frequent_reward_explanation_engineering_v1', "culture_level": culture_level}))
	register_env(f"GridDrive-{culture_level}-ExplanationEngineering-V2", lambda config: GridDrive({"reward_fn": 'frequent_reward_explanation_engineering_v2', "culture_level": culture_level}))
	register_env(f"GridDrive-{culture_level}-S*J", lambda config: GridDrive({"reward_fn": 'frequent_reward_step_multiplied_by_junctions', "culture_level": culture_level}))
	register_env(f"GridDrive-{culture_level}-FullStep", lambda config: GridDrive({"reward_fn": 'frequent_reward_full_step', "culture_level": culture_level}))

### GraphDrive
from .env.graph_drive import GraphDrive
culture_level_list = ["Easy","Medium","Hard"]
for culture_level in culture_level_list:
	register_env(f"GraphDrive-{culture_level}", lambda config: GraphDrive({"reward_fn": 'frequent_reward_default', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-ExplanationEngineering-V1", lambda config: GraphDrive({"reward_fn": 'frequent_reward_explanation_engineering_v1', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-ExplanationEngineering-V2", lambda config: GraphDrive({"reward_fn": 'frequent_reward_explanation_engineering_v2', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-ExplanationEngineering-V3", lambda config: GraphDrive({"reward_fn": 'frequent_reward_explanation_engineering_v3', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-S*J", lambda config: GraphDrive({"reward_fn": 'frequent_reward_step_multiplied_by_junctions', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-FullStep", lambda config: GraphDrive({"reward_fn": 'frequent_reward_full_step', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-Sparse", lambda config: GraphDrive({"reward_fn": 'sparse_reward_default', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-Sparse-ExplanationEngineering-V1", lambda config: GraphDrive({"reward_fn": 'sparse_reward_explanation_engineering_v1', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-Sparse-ExplanationEngineering-V2", lambda config: GraphDrive({"reward_fn": 'sparse_reward_explanation_engineering_v2', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-Sparse-ExplanationEngineering-V3", lambda config: GraphDrive({"reward_fn": 'sparse_reward_explanation_engineering_v3', "culture_level": culture_level}))
	register_env(f"GraphDrive-{culture_level}-Sparse-S*J", lambda config: GraphDrive({"reward_fn": 'sparse_reward_step_multiplied_by_junctions', "culture_level": culture_level}))

###########################################################################
### Multi-Agent GraphDrive
from .env.multi_agent_graph_delivery.full_world_all_agents_env import FullWorldAllAgents_GraphDelivery
from .env.multi_agent_graph_delivery.part_world_some_agents_env import PartWorldSomeAgents_GraphDelivery
from .env.multi_agent_graph_delivery.full_world_some_agents_env import FullWorldSomeAgents_GraphDelivery
culture_list = ["Heterogeneity"]

for culture in culture_list:
	register_env(f"MAGraphDelivery-FullWorldAllAgents-{culture}", lambda config: FullWorldAllAgents_GraphDelivery({"culture": culture, **config}))
register_env(f"MAGraphDelivery-FullWorldAllAgents", lambda config: FullWorldAllAgents_GraphDelivery({"culture": None, **config}))

for culture in culture_list:
	register_env(f"MAGraphDelivery-PartWorldSomeAgents-{culture}", lambda config: PartWorldSomeAgents_GraphDelivery({"culture": culture, **config}))
register_env(f"MAGraphDelivery-PartWorldSomeAgents", lambda config: PartWorldSomeAgents_GraphDelivery({"culture": None, **config}))

for culture in culture_list:
	register_env(f"MAGraphDelivery-FullWorldSomeAgents-{culture}", lambda config: FullWorldSomeAgents_GraphDelivery({"culture": culture, **config}))
register_env(f"MAGraphDelivery-FullWorldSomeAgents", lambda config: FullWorldSomeAgents_GraphDelivery({"culture": None, **config}))
