sudo docker build . -t turbo:latest
sudo docker run --rm --gpus all --network host -v $(pwd):/code/turbo-codes -w /code/turbo-codes -u $(id -u):$(id -g) -it turbo:latest bash
# sudo docker run --rm --gpus all --network host -v $(pwd):/code/turbo-codes -u 0 -w /code/turbo-codes -it turbo:latest bash

