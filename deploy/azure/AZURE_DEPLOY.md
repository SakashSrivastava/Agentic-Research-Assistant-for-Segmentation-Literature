# Deploying to an Azure VM (free, no card) with HTTPS

Runs the self-contained Docker image on an Azure **B1s** VM (your free 750-hours
service), behind Caddy for automatic HTTPS, on a free DuckDNS subdomain. Admin stays
locked to you (server-side only). Email verification is off for the demo.

## Free accounts you'll need (all no-card)
- **Docker Hub** — https://hub.docker.com (to host the image).
- **DuckDNS** — https://www.duckdns.org (free subdomain; log in with GitHub/Google).
- Azure for Students (you have it).

---

## Step 1 - Build and push the image to Docker Hub (run on your PC)
Rebuild so the image has the latest code, then tag and push it (replace `USER` with
your Docker Hub username):
```bash
docker compose build
docker login
docker tag aimedicalliteraturechatbot-web:latest USER/seg-lit-assistant:latest
docker push USER/seg-lit-assistant:latest
```

## Step 2 - Create a DuckDNS subdomain
On duckdns.org, create a subdomain, e.g. `seglit` -> `seglit.duckdns.org`. Leave the IP
blank for now; you'll set it in Step 4.

## Step 3 - Create the Azure VM
Azure Portal -> Virtual Machines -> Create -> Azure virtual machine:
- **Image:** Ubuntu Server 22.04 LTS
- **Size:** B1s (free-eligible)
- **Authentication:** SSH public key (let Azure generate one and download it, or use your own)
- **Inbound ports:** allow **SSH (22)**, **HTTP (80)**, **HTTPS (443)**
- Create, then copy the VM's **public IP**.

## Step 4 - Point DuckDNS at the VM
Back on duckdns.org, set your subdomain's IP to the VM's public IP and Save.

## Step 5 - SSH in, install Docker, add swap
```bash
ssh azureuser@VM_PUBLIC_IP           # use the key from Step 3

curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER        # then: exit and SSH back in so the group applies

# 2 GB swap so the ~1.5 GB app fits on a 1 GB B1s
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Step 6 - Copy the config files and create .env
From your PC (project root), copy the two files to the VM:
```bash
scp deploy/azure/docker-compose.yml deploy/azure/Caddyfile azureuser@VM_PUBLIC_IP:~/
```
Then on the VM, create `.env` (same folder) with:
```
IMAGE=USER/seg-lit-assistant:latest
DOMAIN=seglit.duckdns.org
BASE_URL=https://seglit.duckdns.org
GROQ_API_KEY=gsk_your_key
FLASK_SECRET_KEY=paste_a_long_random_string
ADMIN_EMAIL=sakashsrivastava06@gmail.com
```
(Generate a secret with `openssl rand -hex 32`.)

## Step 7 - Launch
```bash
docker compose up -d
docker compose logs -f caddy        # watch it obtain the HTTPS certificate (Ctrl-C to stop)
```
First boot pulls the image (~a few minutes) and loads the model. Then open
`https://seglit.duckdns.org`.

## Step 8 - Make yourself admin (only you can do this)
```bash
docker compose exec web python -m src.manage_admin grant sakashsrivastava06@gmail.com
```
Log in with that email -> the Admin dashboard appears. No one else can be granted admin.

## Redeploying after code changes
On your PC: rebuild + push (`docker compose build` then `docker push USER/seg-lit-assistant:latest`).
On the VM: `docker compose pull && docker compose up -d`.

## Cost control
B1s is free for 12 months. Stop the VM from the Azure portal when you don't need it to
save the free-hours/credit. Your $100 credit covers a bigger VM (B1ms/B2s) if B1s feels slow.
