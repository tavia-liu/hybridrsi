"""The smallest valid policy: takes one look through the cameras, then waits out the episode. Replace policy()
with your own.

A policy is a Python file defining policy(api); the methods of `api` are listed by

    PYTHONPATH=.:benchmark python -c "import api; print(api.spec())"

api.EpisodeOver is raised when the episode ends (success, safety stop or the time limit): let it propagate.
"""


def policy(api):
    print(api.instruction())                      # e.g. "Pick up the orange rod and insert it ... blue fixture."
    views = api.camera()                          # agentview / eye_in_hand: rgb, depth, K, cam_pos, cam_R
    obs = api.observe()                           # tcp pose, joints, gripper, external wrench
    assert views["agentview"]["rgb"].shape[2] == 3 and obs["tcp_pos"].shape == (3,)
    api.wait(1.0)
    api.wait(60.0)                                # raises api.EpisodeOver at the time limit
