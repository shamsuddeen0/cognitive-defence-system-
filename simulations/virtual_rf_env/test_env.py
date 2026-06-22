from rf_env import RFEnv
import numpy as np

def test_env():
    env = RFEnv()
    obs, _ = env.reset()
    print(f"Initial Observation: {obs}")

    # Try to hop around
    for i in range(5):
        action = np.random.randint(0, 4)
        obs, reward, term, trunc, _ = env.step(action)
        print(f"Step {i}: Action={action}, Reward={reward:.2f}, Obs={obs}")

if __name__ == "__main__":
    test_env()
