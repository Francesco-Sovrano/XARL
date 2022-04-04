from ray.tune.registry import register_env
######### Add new environment below #########

### GFootball
from environments.gfootball.gfootball_env import RllibGFootball
register_env('gfootball', lambda config: RllibGFootball(config))
