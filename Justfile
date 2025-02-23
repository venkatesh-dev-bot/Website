image_name := "resume_qa"
container_name := "resume_qa"

rebuild: rm-container build

rm-container:
  docker rm -f {{container_name}}

build: rm-container
  docker build -t {{image_name}} .

run: build
  docker run -d  -p 9090:9090 --name {{container_name}} -it --restart unless-stopped {{image_name}}

watch: run
  docker logs -f {{container_name}}