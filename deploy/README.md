# Deploy on one EC2 instance

The whole stack (Caddy, web, API, worker, Postgres, Redis) runs with Docker Compose on one
instance, from [`docker-compose.prod.yml`](../docker-compose.prod.yml). Caddy is the only service
reachable from the internet. It serves HTTPS with a certificate it gets from Let's Encrypt and
renews itself. This costs about $20 a month and has no redundancy: if the instance dies, the site
is down until you replace it, and only your backups survive.

## 1. Launch the instance (AWS console, EC2)

- **AMI:** Ubuntu Server 24.04 LTS, x86_64.
- **Type:** `t3.small` (2 vCPU, 2 GB). The user-data script adds 2 GB of swap for the image build.
- **Key pair:** create or pick one; you need it for SSH.
- **Network:** a security group allowing inbound TCP 22 from *My IP*, and TCP 80 and 443 from
  anywhere (UDP 443 too if you want HTTP/3).
- **Storage:** 20 GB gp3.
- **Advanced details → User data:** paste [`ec2-user-data.sh`](ec2-user-data.sh). It installs
  Docker and clones the repo into `/home/ubuntu/DocPilot`.

Then allocate an **Elastic IP** and associate it with the instance, so the address (and the domain
below) survives a stop and start.

## 2. Pick a domain

Caddy needs a hostname that resolves to the instance. Either:
- point an A record of your own domain at the Elastic IP, or
- use `<ip-with-dashes>.sslip.io` (e.g. `203-0-113-7.sslip.io`), which resolves to that IP with
  no setup.

## 3. Configure

```bash
ssh -i <key>.pem ubuntu@<elastic-ip>
cloud-init status --wait          # wait for the user-data script to finish
cd DocPilot
cp .env.example .env
nano .env
```

Set at least:

| variable | value |
|---|---|
| `DOMAIN` | the hostname from step 2 |
| `POSTGRES_PASSWORD`, `API_KEY` | `python3 -c "import secrets; print(secrets.token_hex(32))"`, one each |
| `WEB_PASSWORD` | the password visitors type (any username) |
| `LLM_PROVIDER` and its key | e.g. `anthropic` + `ANTHROPIC_API_KEY`, or `openai` + `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `CHAT_MODEL` for DeepSeek or the LLM Gateway |
| embeddings | a hosted endpoint; a local GPU server on your own machine won't be reachable from here. See below. |

For example, DeepSeek for chat and OpenAI for embeddings. `OPENAI_*` then points at DeepSeek, so
embeddings get their own endpoint:

```bash
LLM_PROVIDER=openai
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=<deepseek key>
CHAT_MODEL=deepseek-v4-flash
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=<openai key>
EMBEDDING_MODEL=text-embedding-3-small
```

`docker-compose.prod.yml` sets the rest itself (production mode, database and Redis URLs, queued
ingest, client IPs), overriding `.env`.

## 4. Start

```bash
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml ps        # every service should become healthy
```

The first build takes several minutes. Open `https://<DOMAIN>` and sign in with any username and
`WEB_PASSWORD`. To load the sample corpus:

```bash
docker compose -f docker-compose.prod.yml exec api docpilot ingest /corpus
```

## Operating it

```bash
alias dc='docker compose -f docker-compose.prod.yml'
dc logs -f api worker                      # logs (JSON)
git pull && dc up -d --build               # deploy a new version; db init applies new migrations
dc exec db pg_dump -U docpilot docpilot | gzip > backup-$(date +%F).sql.gz   # backup
```

Copy backups off the instance (e.g. `aws s3 cp`), since they're lost with the instance's disk.
Containers restart by themselves after a crash or a reboot.

If the certificate doesn't arrive, `dc logs caddy` says why. Usually port 80 is closed in the
security group, or the domain doesn't resolve to the instance yet.
