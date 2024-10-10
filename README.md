## dqn-assault
  Deep Q network experiments for the atari assault environment.

### Usage
  To execute run:
  ```
  docker compose up
  ```
  Once execution has ended, the container will exit automatically. The program writes output to `/apps/output` inside the container. To get the output directory to your system run:
  ```
  docker cp dqn-assault-assault-dqn-1:/app/output ./output
  ```
  The `output` directory will contain:
  - `output/checkpoints`
  - `output/videos`
  - `output/XX.png`
