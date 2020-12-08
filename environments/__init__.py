from ray.tune.registry import register_env
######### Add new environment below #########

from environments.gym_env_example import Example_v0
register_env("ToyExample-v0", lambda config: Example_v0())

from environments.car_controller.cesco_drive_v0 import CescoDriveV0
register_env("CescoDrive-v0", lambda config: CescoDriveV0())

from environments.car_controller.cesco_drive_v1 import CescoDriveV1
register_env("CescoDrive-v1", lambda config: CescoDriveV1())

from environments.car_controller.cesco_drive_v2 import CescoDriveV2
register_env("CescoDrive-v2", lambda config: CescoDriveV2())

from environments.car_controller.alex_drive_v0 import AlexDriveV0
register_env("AlexDrive-v0", lambda config: AlexDriveV0())

from environments.car_controller.alex_drive_v1 import AlexDriveV1
register_env("AlexDrive-v1", lambda config: AlexDriveV1())

from environments.car_controller.alex_drive_v2 import AlexDriveV2
register_env("AlexDrive-v2", lambda config: AlexDriveV2())