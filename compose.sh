sudo docker compose down -v
sudo docker compose build --no-cache
sudo docker compose up --build -d --remove-orphans --force-recreate