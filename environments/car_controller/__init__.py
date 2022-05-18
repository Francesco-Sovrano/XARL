from ray.tune.registry import register_env
######### Add new environment below #########

### CescoDrive
from environments.car_controller.cesco_drive.cesco_drive_v0 import CescoDriveV0
register_env("CescoDrive-V0", lambda config: CescoDriveV0(config))

from environments.car_controller.cesco_drive.cesco_drive_v1 import CescoDriveV1
register_env("CescoDrive-V1", lambda config: CescoDriveV1(config))

### GridDrive
from environments.car_controller.grid_drive.grid_drive import GridDrive
culture_level_list = ["Easy","Medium","Hard"]
for culture_level in culture_level_list:
	register_env(f"GridDrive-{culture_level}", lambda config: GridDrive({"reward_fn": 'frequent_reward_default', "culture_level": culture_level}))
	register_env(f"GridDrive-{culture_level}-ExplanationEngineering-V1", lambda config: GridDrive({"reward_fn": 'frequent_reward_explanation_engineering_v1', "culture_level": culture_level}))
	register_env(f"GridDrive-{culture_level}-ExplanationEngineering-V2", lambda config: GridDrive({"reward_fn": 'frequent_reward_explanation_engineering_v2', "culture_level": culture_level}))
	register_env(f"GridDrive-{culture_level}-S*J", lambda config: GridDrive({"reward_fn": 'frequent_reward_step_multiplied_by_junctions', "culture_level": culture_level}))
	register_env(f"GridDrive-{culture_level}-FullStep", lambda config: GridDrive({"reward_fn": 'frequent_reward_full_step', "culture_level": culture_level}))

### GraphDrive
from environments.car_controller.graph_drive.graph_drive import GraphDrive
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

### MultiAgentGraphDrive
from environments.car_controller.food_delivery_multi_agent_graph_drive.global_view_env import MultiAgentGraphDrive
culture_level_list = ["Easy","Medium","Hard"]
for culture_level in culture_level_list:
	register_env(f"MAGraphDrive-{culture_level}", lambda config: MultiAgentGraphDrive({"reward_fn": 'frequent_reward_default', "culture_level": culture_level, **config}))
	register_env(f"MAGraphDrive-{culture_level}-Sparse", lambda config: MultiAgentGraphDrive({"reward_fn": 'sparse_reward_default', "culture_level": culture_level, **config}))
register_env(f"MAGraphDrive", lambda config: MultiAgentGraphDrive({"reward_fn": 'frequent_reward_no_culture', "culture_level": "Easy", **config}))
register_env(f"MAGraphDrive-Sparse", lambda config: MultiAgentGraphDrive({"reward_fn": 'sparse_reward_no_culture', "culture_level": "Easy", **config}))

from environments.car_controller.food_delivery_multi_agent_graph_drive.partial_view_with_comm_env import PVCommMultiAgentGraphDrive
culture_level_list = ["Easy","Medium","Hard"]
for culture_level in culture_level_list:
	register_env(f"MAGraphDrive-PVComm-{culture_level}", lambda config: PVCommMultiAgentGraphDrive({"reward_fn": 'frequent_reward_default', "culture_level": culture_level, **config}))
	register_env(f"MAGraphDrive-PVComm-{culture_level}-Sparse", lambda config: PVCommMultiAgentGraphDrive({"reward_fn": 'sparse_reward_default', "culture_level": culture_level, **config}))
register_env(f"MAGraphDrive-PVComm", lambda config: PVCommMultiAgentGraphDrive({"reward_fn": 'frequent_reward_default', "culture_level": None, **config}))
register_env(f"MAGraphDrive-PVComm-Sparse", lambda config: PVCommMultiAgentGraphDrive({"reward_fn": 'sparse_reward_default', "culture_level": None, **config}))
