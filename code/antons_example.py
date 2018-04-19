se = StudentEnv()
histories = []

for _ in range(10000):

  history = [se.reset()]
  done = False
  
  while done is False:
    a = se.action_space.sample()
    obs, reward, done, info = se.step(a)
    history.append(obs)
    
  histories.append(history)
